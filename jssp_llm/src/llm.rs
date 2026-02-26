// src/llm.rs
//
// Pure-Rust LLM inference via the `candle` framework.
// Supports Phi-3 Mini (the base model used in the paper) with a LoRA adapter
// stored as safetensors files in `adapter_dir`.
//
// Key public API:
//   ModelBundle::load(adapter_dir, device)  -> async Result<ModelBundle>
//   solve_with_llm(jobs, …, bundle, device) -> Result<(Option<LlmResult>, usize)>
//   make_device(idx)                        -> Result<Device>

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};
use candle_nn::VarBuilder;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::path::{Path, PathBuf};
use rand::prelude::*;

use crate::jssp::{Jobs, Schedule};
use crate::parser::parse_output;
use crate::prompt::build_prompt_for_jobs;
use crate::jssp::validate;

// ── Device helper ─────────────────────────────────────────────────────────────

/// Create a candle Device.
/// `idx` == -1 → CPU; `idx` >= 0 → CUDA device at that index.
pub fn make_device(idx: i64) -> Result<Device> {
    if idx < 0 {
        Ok(Device::Cpu)
    } else {
        #[cfg(feature = "cuda")]
        {
            Ok(Device::new_cuda(idx as usize)?)
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("Warning: CUDA requested but binary compiled without --features cuda. Falling back to CPU.");
            Ok(Device::Cpu)
        }
    }
}

// ── Model bundle ─────────────────────────────────────────────────────────────

/// Everything needed for inference: the model, tokenizer, and config.
pub struct ModelBundle {
    pub model: Phi3Model,
    pub tokenizer: Tokenizer,
    pub config: Phi3Config,
    pub eos_token_id: u32,
}

impl ModelBundle {
    /// Load the Phi-3 model (base + LoRA adapter weights) from `adapter_dir`.
    ///
    /// Expected files in `adapter_dir`:
    ///   adapter_config.json   – LoRA hyper-params (read for reference; candle
    ///                           merges LoRA weights at load time via safetensors)
    ///   model.safetensors     – merged weights  OR
    ///   adapter_model.safetensors + base model shards fetched from HF Hub
    ///   tokenizer.json        – HF fast tokenizer
    ///   config.json           – model architecture config
    pub async fn load(adapter_dir: &str, device: &Device) -> Result<Self> {
        let adapter_path = Path::new(adapter_dir);

        // --- Resolve adapter directory (unwrap single top-level folder) -------
        let adapter_path = resolve_adapter_dir(adapter_path)?;

        // --- Tokenizer -------------------------------------------------------
        let tok_path = adapter_path.join("tokenizer.json");
        if !tok_path.exists() {
            bail!(
                "tokenizer.json not found in '{}'. \
                 Make sure the adapter directory contains the tokenizer.",
                adapter_path.display()
            );
        }
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // --- Model config ----------------------------------------------------
        let config_path = adapter_path.join("config.json");
        if !config_path.exists() {
            bail!(
                "config.json not found in '{}'. \
                 Copy it from the base model (microsoft/Phi-3-mini-4k-instruct).",
                adapter_path.display()
            );
        }
        let config_str = std::fs::read_to_string(&config_path)
            .context("Failed to read config.json")?;
        let config: Phi3Config = serde_json::from_str(&config_str)
            .context("Failed to parse config.json")?;

        // --- Weights (safetensors) -------------------------------------------
        // Prefer merged weights; fall back to loading base + adapter separately.
        let weight_files = find_weight_files(&adapter_path)?;
        if weight_files.is_empty() {
            bail!(
                "No .safetensors weight files found in '{}'. \
                 Expected 'model.safetensors' or 'adapter_model.safetensors'.",
                adapter_path.display()
            );
        }

        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)?
        };

        let model = Phi3Model::new(&config, vb)
            .context("Failed to construct Phi-3 model from weights")?;

        // EOS token
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2); // Phi-3 default

        println!("  Tokenizer  : {}", tok_path.display());
        println!("  Config     : {}", config_path.display());
        println!("  Weights    : {} file(s)", weight_files.len());
        println!("  Device     : {:?}", device);
        println!("  DType      : {:?}", dtype);

        Ok(Self { model, tokenizer, config, eos_token_id })
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

/// Result of one successful LLM inference + validation pass.
pub struct LlmResult {
    pub makespan:    u32,
    pub start_times: Schedule,
    pub text:        String,
}

/// Mirror of the Python `solve_with_llm` function.
///
/// 1. Formats the JSSP as a prompt.
/// 2. Generates `n_samples` candidate solutions via temperature sampling.
/// 3. Parses + validates each with `parse_output` + `validate`.
/// 4. Returns the best (lowest makespan) valid result, and the count of valid samples.
pub fn solve_with_llm(
    jobs:           &Jobs,
    n_samples:      usize,
    temperature:    f64,
    top_p:          f64,
    max_new_tokens: usize,
    bundle:         &ModelBundle,
    device:         &Device,
) -> Result<(Option<LlmResult>, usize)> {
    let prompt = build_prompt_for_jobs(jobs);

    // Tokenise prompt
    let encoding = bundle.tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenisation error: {e}"))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = input_ids.len();

    let mut best: Option<LlmResult> = None;
    let mut valid_count = 0usize;
    let mut rng = rand::thread_rng();

    for sample_i in 0..n_samples {
        println!("  Sample {}/{} …", sample_i + 1, n_samples);

        let text = generate(
            &bundle.model,
            &bundle.tokenizer,
            &input_ids,
            prompt_len,
            max_new_tokens,
            temperature,
            top_p,
            bundle.eos_token_id,
            device,
            &mut rng,
        )?;

        println!("    raw output: {}", &text[..text.len().min(120)]);

        // Parse
        let Some(schedule) = parse_output(&text, jobs) else {
            println!("    → parse failed, skipping");
            continue;
        };

        // Validate
        let (ok, ms) = validate(jobs, &schedule);
        if !ok {
            println!("    → constraint violation, skipping");
            continue;
        }

        valid_count += 1;
        if best.as_ref().map_or(true, |b| ms < b.makespan) {
            best = Some(LlmResult {
                makespan:    ms,
                start_times: schedule,
                text,
            });
        }
        println!("    ✓ valid, makespan = {}", ms);
    }

    Ok((best, valid_count))
}

// ── Token-by-token generation loop ───────────────────────────────────────────

fn generate(
    model:          &Phi3Model,
    tokenizer:      &Tokenizer,
    input_ids:      &[u32],
    prompt_len:     usize,
    max_new_tokens: usize,
    temperature:    f64,
    top_p:          f64,
    eos_token_id:   u32,
    device:         &Device,
    rng:            &mut impl Rng,
) -> Result<String> {
    // Build initial input tensor [1, seq_len]
    let mut token_ids: Vec<u32> = input_ids.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    // We run a simple KV-cache-less loop for clarity.
    // For large models / long sequences, replace with candle's KV-cache API.
    for _step in 0..max_new_tokens {
        let seq_len = token_ids.len();
        let input_tensor = Tensor::from_slice(
            token_ids.as_slice(),
            &[1, seq_len],
            device,
        )?;

        // Forward pass — candle Phi3Model returns (logits: [batch, seq, vocab])
        let logits = model.forward(&input_tensor, seq_len - 1)?;
        // logits shape: [1, 1, vocab_size]  (last token only)
        let logits = logits.squeeze(0)?.squeeze(0)?; // [vocab_size]

        // Sample next token
        let next_token = sample_token(&logits, temperature, top_p, rng)?;
        token_ids.push(next_token);
        generated.push(next_token);

        if next_token == eos_token_id {
            break;
        }
    }

    // Decode only the generated tokens (not the prompt)
    let decoded = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {e}"))?;

    Ok(decoded)
}

// ── Sampling helpers ──────────────────────────────────────────────────────────

/// Sample one token from `logits` using temperature scaling + nucleus (top-p) filtering.
fn sample_token(
    logits:      &Tensor,
    temperature: f64,
    top_p:       f64,
    rng:         &mut impl Rng,
) -> Result<u32> {
    // 1. Apply temperature
    let logits = (logits / temperature as f64)?;

    // 2. Softmax → probabilities
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // 3. Nucleus (top-p) filtering
    // Sort indices by descending probability
    let mut indexed: Vec<(usize, f32)> = probs_vec
        .iter()
        .cloned()
        .enumerate()
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum = 0.0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for (idx, p) in &indexed {
        nucleus.push((*idx, *p));
        cumsum += p;
        if cumsum as f64 >= top_p {
            break;
        }
    }

    // 4. Re-normalise nucleus and sample
    let total: f32 = nucleus.iter().map(|(_, p)| p).sum();
    let u: f32 = rng.gen::<f32>() * total;
    let mut acc = 0.0f32;
    for (idx, p) in &nucleus {
        acc += p;
        if acc >= u {
            return Ok(*idx as u32);
        }
    }

    // Fallback: argmax
    Ok(indexed[0].0 as u32)
}

// ── File-system helpers ───────────────────────────────────────────────────────

/// If `adapter_dir` does not contain `adapter_config.json`, descend into the
/// single subdirectory (mirrors the Python notebook logic).
fn resolve_adapter_dir(path: &Path) -> Result<PathBuf> {
    if path.join("config.json").exists() || path.join("adapter_config.json").exists() {
        return Ok(path.to_path_buf());
    }
    // Try one level down
    let subdirs: Vec<_> = std::fs::read_dir(path)
        .with_context(|| format!("Cannot read adapter directory '{}'", path.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    if subdirs.len() == 1 {
        let sub = subdirs[0].path();
        if sub.join("config.json").exists() || sub.join("adapter_config.json").exists() {
            println!("  Unwrapped adapter subdirectory: {}", sub.display());
            return Ok(sub);
        }
    }
    bail!(
        "Could not find config.json or adapter_config.json in '{}' or its immediate subdirectories.",
        path.display()
    );
}

/// Collect all *.safetensors files in `dir`, sorted for determinism.
fn find_weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Cannot read '{}'", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    Ok(files)
}
