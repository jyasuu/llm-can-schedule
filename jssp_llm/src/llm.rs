// src/llm.rs
//
// Pure-Rust LLM inference via `candle`.
//
// Handles the real-world adapter directory layout produced by unsloth/PEFT:
//   adapter_config.json        — LoRA hyper-params (rank, alpha, target modules…)
//   adapter_model.safetensors  — LoRA delta weights only  (NOT a full model)
//   tokenizer.json             — Phi-3 sentencepiece/Unigram tokenizer
//
// Because only LoRA deltas are present, we:
//   1. Parse adapter_config.json to find the base model repo id.
//   2. Download base model weights + config.json from HF Hub.
//   3. Load all base tensors into memory.
//   4. Apply LoRA math in-place:  W += (lora_B @ lora_A) * (alpha / rank)
//   5. Build the Phi-3 model from the merged weight map.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};
use hf_hub::api::tokio::Api;
use rand::prelude::*;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::jssp::{validate, Schedule};
use crate::parser::parse_output;
use crate::prompt::build_prompt_for_jobs;

// ── Device helper ─────────────────────────────────────────────────────────────

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
            eprintln!("CUDA requested but not compiled in — falling back to CPU.");
            Ok(Device::Cpu)
        }
    }
}

// ── adapter_config.json schema ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct AdapterConfig {
    base_model_name_or_path: String,
    r: usize,
    lora_alpha: f64,
}

// ── ModelBundle ───────────────────────────────────────────────────────────────

pub struct ModelBundle {
    pub model: Phi3Model,
    pub tokenizer: Tokenizer,
    #[allow(dead_code)]
    pub config: Phi3Config,
    pub eos_token_id: u32,
}

impl ModelBundle {
    pub async fn load(adapter_dir: &str, device: &Device) -> Result<Self> {
        let adapter_path = resolve_adapter_dir(Path::new(adapter_dir))?;

        // ── 1. Parse adapter_config.json ──────────────────────────────────────
        let adapter_cfg_path = adapter_path.join("adapter_config.json");
        if !adapter_cfg_path.exists() {
            bail!("adapter_config.json not found in '{}'", adapter_path.display());
        }
        let adapter_cfg: AdapterConfig =
            serde_json::from_str(&std::fs::read_to_string(&adapter_cfg_path)?)
                .context("Failed to parse adapter_config.json")?;

        let base_repo  = &adapter_cfg.base_model_name_or_path;
        let lora_scale = adapter_cfg.lora_alpha / adapter_cfg.r as f64;
        println!("  Base model : {}", base_repo);
        println!("  LoRA rank={} alpha={} scale={:.4}",
            adapter_cfg.r, adapter_cfg.lora_alpha, lora_scale);

        // ── 2. Download base model from HF Hub ────────────────────────────────
        println!("  Fetching base model weights from HF Hub (cached after first run)…");
        let api  = Api::new().context("Failed to init HF Hub API")?;
        let repo = api.model(base_repo.clone());

        let config_path = repo.get("config.json").await
            .with_context(|| format!("Cannot fetch config.json from {}", base_repo))?;
        let config: Phi3Config =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)
                .context("Cannot parse config.json")?;

        let base_files = download_base_weights(&repo).await
            .with_context(|| format!("Cannot fetch weights from {}", base_repo))?;
        println!("  Base shards: {}", base_files.len());

        // ── 3. Load base tensors ───────────────────────────────────────────────
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let mut tensors = load_safetensors_map(&base_files, dtype, device)?;

        // ── 4. Load & apply LoRA deltas ───────────────────────────────────────
        let adapter_st = adapter_path.join("adapter_model.safetensors");
        if !adapter_st.exists() {
            bail!("adapter_model.safetensors not found in '{}'", adapter_path.display());
        }
        let lora_map = load_safetensors_map(&[&adapter_st], dtype, device)?;
        println!("  LoRA tensors: {}", lora_map.len());

        let mut merged = 0usize;
        let a_keys: Vec<String> = lora_map.keys()
            .filter(|k| k.ends_with(".lora_A.weight"))
            .cloned()
            .collect();

        for a_key in &a_keys {
            let b_key  = a_key.replace(".lora_A.weight", ".lora_B.weight");
            let Some(lora_a) = lora_map.get(a_key.as_str()) else { continue };
            let Some(lora_b) = lora_map.get(&b_key)          else { continue };

            // PEFT prefix: "base_model.model.<path>.lora_A.weight"
            // Base key:    "<path>.weight"
            let base_key = a_key
                .trim_start_matches("base_model.model.")
                .replace(".lora_A.weight", ".weight");

            let Some(w) = tensors.get(&base_key) else { continue };

            // delta = (lora_B @ lora_A) * scale
            let delta = lora_b.matmul(lora_a)?.affine(lora_scale, 0.0)?;
            tensors.insert(base_key, (w + delta)?);
            merged += 1;
        }
        println!("  Merged {} LoRA module(s)", merged);

        // ── 5. Build Phi-3 model ───────────────────────────────────────────────
        let vmap = tensors_to_varmap(tensors)?;
        let vb   = VarBuilder::from_varmap(&vmap, dtype, device);
        let model = Phi3Model::new(&config, vb)
            .context("Failed to construct Phi-3 model")?;

        // ── 6. Tokenizer ───────────────────────────────────────────────────────
        let tokenizer = load_tokenizer(&adapter_path, &repo).await?;
        let eos_token_id = find_eos(&tokenizer);
        println!("  EOS id     : {}", eos_token_id);

        Ok(Self { model, tokenizer, config, eos_token_id })
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

pub struct LlmResult {
    pub makespan:    u32,
    pub start_times: Schedule,
    pub text:        String,
}

pub fn solve_with_llm(
    jobs:           &[Vec<(u32, u32)>],
    n_samples:      usize,
    temperature:    f64,
    top_p:          f64,
    max_new_tokens: usize,
    bundle:         &mut ModelBundle,
    device:         &Device,
) -> Result<(Option<LlmResult>, usize)> {
    let prompt   = build_prompt_for_jobs(jobs);
    let encoding = bundle.tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenisation error: {e}"))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    let mut best:        Option<LlmResult> = None;
    let mut valid_count: usize = 0;
    let mut rng = rand::rng();

    for i in 0..n_samples {
        println!("  Sample {}/{} …", i + 1, n_samples);
        let text = generate(
            &mut bundle.model, &bundle.tokenizer, &input_ids,
            max_new_tokens, temperature, top_p, bundle.eos_token_id,
            device, &mut rng,
        )?;
        println!("    raw: {}", text.chars().take(120).collect::<String>());

        let Some(schedule) = parse_output(&text, jobs) else {
            println!("    → parse failed"); continue;
        };
        let (ok, ms) = validate(jobs, &schedule);
        if !ok { println!("    → constraint violation"); continue; }

        valid_count += 1;
        if best.as_ref().map_or(true, |b| ms < b.makespan) {
            best = Some(LlmResult { makespan: ms, start_times: schedule, text });
        }
        println!("    ✓ makespan = {}", ms);
    }
    Ok((best, valid_count))
}

// ── Generation loop ───────────────────────────────────────────────────────────

fn generate(
    model:          &mut Phi3Model,
    tokenizer:      &Tokenizer,
    input_ids:      &[u32],
    max_new_tokens: usize,
    temperature:    f64,
    top_p:          f64,
    eos_token_id:   u32,
    device:         &Device,
    rng:            &mut impl Rng,
) -> Result<String> {
    let mut ids  = input_ids.to_vec();
    let mut gen  = Vec::<u32>::new();

    for _ in 0..max_new_tokens {
        let n   = ids.len();
        let inp = Tensor::from_slice(ids.as_slice(), &[1usize, n], device)?;
        // forward(input_ids, start_pos) — start_pos selects which position's logits to return
        let logits = model.forward(&inp, n - 1)?
            .squeeze(0)?.squeeze(0)?;          // [vocab_size]
        let next = sample_token(&logits, temperature, top_p, rng)?;
        ids.push(next);
        gen.push(next);
        if next == eos_token_id { break; }
    }

    tokenizer.decode(&gen, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {e}"))
}

// ── Nucleus sampling ──────────────────────────────────────────────────────────

fn sample_token(
    logits:      &Tensor,
    temperature: f64,
    top_p:       f64,
    rng:         &mut impl Rng,
) -> Result<u32> {
    let logits = (logits / temperature)?;
    let probs  = candle_nn::ops::softmax(&logits, 0)?;
    let pv: Vec<f32> = probs.to_vec1()?;

    let mut idx: Vec<(usize, f32)> = pv.into_iter().enumerate().collect();
    idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cum = 0f32;
    let mut nucleus = Vec::new();
    for &(i, p) in &idx {
        nucleus.push((i, p));
        cum += p;
        if cum as f64 >= top_p { break; }
    }

    let total: f32 = nucleus.iter().map(|(_, p)| p).sum();
    let u: f32     = rng.random::<f32>() * total;
    let mut acc    = 0f32;
    for &(i, p) in &nucleus {
        acc += p;
        if acc >= u { return Ok(i as u32); }
    }
    Ok(idx[0].0 as u32)
}

// ── HF Hub helpers ────────────────────────────────────────────────────────────

async fn download_base_weights(
    repo: &hf_hub::api::tokio::ApiRepo,
) -> Result<Vec<PathBuf>> {
    // Single-file model
    if let Ok(p) = repo.get("model.safetensors").await {
        return Ok(vec![p]);
    }
    // Sharded model — read the index to discover shard filenames
    let idx_path = repo.get("model.safetensors.index.json").await
        .context("model.safetensors and model.safetensors.index.json both absent")?;
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(idx_path)?)?;
    let wmap = idx["weight_map"].as_object()
        .context("weight_map key missing in shard index")?;
    let mut shards: Vec<String> = wmap.values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shards.sort(); shards.dedup();

    let mut paths = Vec::new();
    for shard in shards {
        paths.push(repo.get(&shard).await
            .with_context(|| format!("Failed to download shard {}", shard))?);
    }
    Ok(paths)
}

async fn load_tokenizer(
    adapter_path: &Path,
    repo: &hf_hub::api::tokio::ApiRepo,
) -> Result<Tokenizer> {
    let local = adapter_path.join("tokenizer.json");
    let path = if local.exists() {
        println!("  Tokenizer  : local ({})", local.display());
        local
    } else {
        println!("  Tokenizer  : downloading from Hub…");
        repo.get("tokenizer.json").await
            .context("Cannot download tokenizer.json from Hub")?
    };
    Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!(
            "Tokenizer load failed: {e}\n\
             Make sure Cargo.toml has: tokenizers = {{ version = \"0.15\" }}\n\
             Then run: cargo clean && cargo build --release"
        ))
}

// ── Tensor utilities ──────────────────────────────────────────────────────────

fn load_safetensors_map(
    files:  &[impl AsRef<Path>],
    dtype:  DType,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut map = HashMap::new();
    for f in files {
        let loaded = candle_core::safetensors::load(f.as_ref(), device)?;
        for (name, t) in loaded {
            map.insert(name, t.to_dtype(dtype)?);
        }
    }
    Ok(map)
}

fn tensors_to_varmap(tensors: HashMap<String, Tensor>) -> Result<VarMap> {
    let vm = VarMap::new();
    {
        let mut data = vm.data().lock().unwrap();
        for (name, tensor) in tensors {
            data.insert(name, Var::from_tensor(&tensor)?);
        }
    }
    Ok(vm)
}

// ── FS helpers ────────────────────────────────────────────────────────────────

fn resolve_adapter_dir(path: &Path) -> Result<PathBuf> {
    if path.join("adapter_config.json").exists() || path.join("config.json").exists() {
        return Ok(path.to_path_buf());
    }
    let subs: Vec<_> = std::fs::read_dir(path)
        .with_context(|| format!("Cannot open '{}'", path.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    if subs.len() == 1 {
        let sub = subs[0].path();
        if sub.join("adapter_config.json").exists() || sub.join("config.json").exists() {
            println!("  Adapter dir: {}", sub.display());
            return Ok(sub);
        }
    }
    bail!("adapter_config.json not found in '{}' or subdirectories", path.display())
}

// ── EOS helpers ───────────────────────────────────────────────────────────────

fn find_eos(tok: &Tokenizer) -> u32 {
    for name in &["<|end|>", "<|endoftext|>", "</s>"] {
        if let Some(id) = tok.token_to_id(name) { return id; }
    }
    32000
}