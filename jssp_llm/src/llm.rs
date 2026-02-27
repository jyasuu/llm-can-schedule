// src/llm.rs

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};
use rand::prelude::*;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::jssp::{validate, Schedule};
use crate::parser::parse_output;
use crate::prompt::build_prompt_for_jobs;

// ── Device ───────────────────────────────────────────────────────────────────

pub fn make_device(idx: i64) -> Result<Device> {
    if idx < 0 {
        Ok(Device::Cpu)
    } else {
        #[cfg(feature = "cuda")]
        { Ok(Device::new_cuda(idx as usize)?) }
        #[cfg(not(feature = "cuda"))]
        { eprintln!("CUDA not compiled in — using CPU."); Ok(Device::Cpu) }
    }
}

// ── dtype helper ─────────────────────────────────────────────────────────────

pub fn parse_dtype(flag: Option<&str>, device: &Device) -> Result<DType> {
    match flag {
        Some("bf16") => Ok(DType::BF16),
        Some("f16")  => Ok(DType::F16),
        Some("f32")  => Ok(DType::F32),
        Some(other)  => bail!("Unknown dtype '{}'. Use f32, f16, or bf16.", other),
        None => Ok(if device.is_cuda() { DType::F16 } else { DType::F32 }),
    }
}

#[derive(Debug, Deserialize)]
struct AdapterConfig {
    base_model_name_or_path: String,
    r: usize,
    lora_alpha: f64,
}

#[derive(Debug, Deserialize)]
struct RawPhi3Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    hidden_act: candle_nn::Activation,
    max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    rope_theta: f64,
    #[serde(default)]
    rope_scaling: serde_json::Value,
    #[serde(default = "default_original_max_pos")]
    original_max_position_embeddings: usize,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<u32>,
}

fn default_rms_norm_eps() -> f64 { 1e-5 }
fn default_original_max_pos() -> usize { 4096 }

impl RawPhi3Config {
    // OOM FIX 1: max_seq_len parameter caps max_position_embeddings.
    //
    // Phi-3-mini-128k has max_position_embeddings=131072. Candle pre-allocates
    // the full RoPE cos/sin cache AND the KV cache at this size on the first
    // forward pass:
    //   32 layers × 2 (K+V) × 32 heads × 96 dim × 131072 × 2 bytes = ~12.9 GB
    // This OOMs even after the model weights fit in VRAM.
    //
    // Python avoids this via max_seq_length=2048 in FastLanguageModel.from_pretrained().
    // We mirror that here. Our prompts are ~800-1200 tokens + 256 new = well under 2048.
    fn into_phi3_config(self, max_seq_len: usize) -> Result<Phi3Config> {
        let rope_scaling: Option<String> = match &self.rope_scaling {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(m) => m.get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            serde_json::Value::Null => None,
            other => {
                println!("  Warning: unexpected rope_scaling shape: {other}, ignoring");
                None
            }
        };

        let capped = self.max_position_embeddings.min(max_seq_len);
        if capped < self.max_position_embeddings {
            println!("  max_position_embeddings capped: {} → {} (OOM prevention)",
                self.max_position_embeddings, capped);
        }

        let mut map = serde_json::json!({
            "vocab_size":                       self.vocab_size,
            "hidden_size":                      self.hidden_size,
            "intermediate_size":                self.intermediate_size,
            "num_hidden_layers":                self.num_hidden_layers,
            "num_attention_heads":              self.num_attention_heads,
            "num_key_value_heads":              self.num_key_value_heads,
            "hidden_act":                       self.hidden_act,
            "max_position_embeddings":          capped,   // ← capped here
            "rms_norm_eps":                     self.rms_norm_eps,
            "rope_theta":                       self.rope_theta,
            "original_max_position_embeddings": self.original_max_position_embeddings,
        });
        if let Some(rs) = rope_scaling {
            map["rope_scaling"] = serde_json::Value::String(rs);
        }

        serde_json::from_value(map).context("Failed to construct Phi3Config")
    }
}

fn parse_phi3_config(json: &str, max_seq_len: usize) -> Result<Phi3Config> {
    let raw: RawPhi3Config = serde_json::from_str(json)
        .context("Failed to parse config.json into RawPhi3Config")?;
    raw.into_phi3_config(max_seq_len)
}

// ── ModelBundle ───────────────────────────────────────────────────────────────

pub struct ModelBundle {
    pub model:        Phi3Model,
    pub tokenizer:    Tokenizer,
    #[allow(dead_code)]
    pub config:       Phi3Config,
    pub eos_token_id: u32,
}

impl ModelBundle {
    pub async fn load(adapter_dir: &str, device: &Device, dtype: DType) -> Result<Self> {
        // Mirrors Python's max_seq_length=2048. Caps max_position_embeddings
        // so the KV+RoPE cache is ~200 MB instead of ~12.9 GB on Phi-3-128k.
        let max_seq_len: usize = 2048;

        let adapter_path = resolve_adapter_dir(Path::new(adapter_dir))?;

        let cfg: AdapterConfig = serde_json::from_str(
            &std::fs::read_to_string(adapter_path.join("adapter_config.json"))
                .context("Cannot read adapter_config.json")?
        ).context("Cannot parse adapter_config.json")?;

        let lora_scale = cfg.lora_alpha / cfg.r as f64;
        println!("  Base model : {}", cfg.base_model_name_or_path);
        println!("  LoRA rank={} alpha={} scale={:.4}", cfg.r, cfg.lora_alpha, lora_scale);
        println!("  dtype      : {:?}", dtype);

        let base_files = find_or_download_base(&cfg.base_model_name_or_path).await?;

        println!("  Loading {} base shard(s)…", base_files.len());
        let mut tensors = load_safetensors_map(&base_files, dtype, device)?;
        println!("  Base tensors loaded: {}", tensors.len());

        // Pass max_seq_len so config caps max_position_embeddings
        let config = load_model_config(&base_files, &cfg.base_model_name_or_path, max_seq_len).await?;

        let adapter_st = adapter_path.join("adapter_model.safetensors");
        if !adapter_st.exists() {
            bail!("adapter_model.safetensors missing in '{}'", adapter_path.display());
        }
        let lora_map = load_safetensors_map(&[&adapter_st], dtype, device)?;
        println!("  LoRA tensors: {}", lora_map.len());

        let mut merged = 0usize;
        let a_keys: Vec<String> = lora_map.keys()
            .filter(|k| k.ends_with(".lora_A.weight"))
            .cloned().collect();

        for a_key in &a_keys {
            let b_key    = a_key.replace(".lora_A.weight", ".lora_B.weight");
            let Some(la) = lora_map.get(a_key.as_str()) else { continue };
            let Some(lb) = lora_map.get(&b_key)          else { continue };
            let base_key = a_key
                .trim_start_matches("base_model.model.")
                .replace(".lora_A.weight", ".weight");
            let Some(w) = tensors.get(&base_key) else { continue };
            let delta = lb.matmul(la)?.affine(lora_scale, 0.0)?;
            tensors.insert(base_key, (w + delta)?);
            merged += 1;
        }
        println!("  Merged {} LoRA modules", merged);

        // OOM FIX 2: free lora_map immediately — ~500 MB freed before model build
        drop(lora_map);

        // OOM FIX 3: scope VarMap so it's freed right after Phi3Model::new()
        // takes ownership of the weights. Without this, VarMap (a full copy of
        // all tensors) lives alongside the built model until end of load().
        let model = {
            let vmap = tensors_to_varmap(tensors)?;
            let vb   = VarBuilder::from_varmap(&vmap, dtype, device);
            Phi3Model::new(&config, vb).context("Failed to build Phi-3 model")?
            // vmap dropped here, freeing the duplicate tensor copies
        };

        let tokenizer    = load_tokenizer(&adapter_path, &cfg.base_model_name_or_path).await?;
        let eos_token_id = find_eos(&tokenizer);
        println!("  EOS token id : {}", eos_token_id);

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
        .map_err(|e| anyhow::anyhow!("Tokenise error: {e}"))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

    let mut best:  Option<LlmResult> = None;
    let mut valid: usize = 0;
    // OOM FIX 4 (also warning fix): rand 0.9 renamed thread_rng() → rng()
    let mut rng = rand::rng();

    for i in 0..n_samples {
        println!("  Sample {}/{} …", i + 1, n_samples);
        bundle.model.clear_kv_cache();

        let text = generate(
            &mut bundle.model, &bundle.tokenizer, &input_ids,
            max_new_tokens, temperature, top_p, bundle.eos_token_id,
            device, &mut rng,
        )?;
        println!("    raw: {}", text.chars().take(120).collect::<String>());

        let Some(sched) = parse_output(&text, jobs)
            else { println!("    → parse failed"); continue; };
        let (ok, ms) = validate(jobs, &sched);
        if !ok { println!("    → constraint violation"); continue; }

        valid += 1;
        if best.as_ref().map_or(true, |b| ms < b.makespan) {
            best = Some(LlmResult { makespan: ms, start_times: sched, text });
        }
        println!("    ✓ makespan = {}", ms);
    }
    Ok((best, valid))
}

// ── Generation ────────────────────────────────────────────────────────────────

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
    let mut gen = Vec::<u32>::new();
    let prompt_len = input_ids.len();

    for index in 0..max_new_tokens {
        let (context, start_pos): (Vec<u32>, usize) = if index == 0 {
            (input_ids.to_vec(), 0)
        } else {
            (vec![*gen.last().unwrap()], prompt_len + index - 1)
        };

        let input  = Tensor::new(context.as_slice(), device)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?
            .squeeze(0)?.squeeze(0)?
            .to_dtype(DType::F32)?;

        let next = sample_token(&logits, temperature, top_p, rng)?;
        gen.push(next);
        if next == eos_token_id { break; }
    }
    tokenizer.decode(&gen, true).map_err(|e| anyhow::anyhow!("Decode: {e}"))
}

fn sample_token(logits: &Tensor, temperature: f64, top_p: f64, rng: &mut impl Rng) -> Result<u32> {
    let logits = (logits / temperature)?;
    let probs  = candle_nn::ops::softmax(&logits, 0)?;
    let pv: Vec<f32> = probs.to_vec1()?;

    let mut idx: Vec<(usize, f32)> = pv.into_iter().enumerate().collect();
    idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cum = 0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for &(i, p) in &idx {
        nucleus.push((i, p)); cum += p;
        if cum as f64 >= top_p { break; }
    }
    let total: f32 = nucleus.iter().map(|(_, p)| p).sum();
    // OOM FIX 4 (also warning fix): rand 0.9 renamed gen() → random()
    let u: f32 = rng.random::<f32>() * total;
    let mut acc = 0f32;
    for &(i, p) in &nucleus { acc += p; if acc >= u { return Ok(i as u32); } }
    Ok(idx[0].0 as u32)
}

// ── HF cache resolution ───────────────────────────────────────────────────────

fn hf_cache_root() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h).join("hub");
    }
    if let Ok(h) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(h);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join(".cache/huggingface/hub")
}

fn repo_to_cache_dir(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

fn find_cached_snapshot(repo_id: &str) -> Option<PathBuf> {
    let cache_root = hf_cache_root();
    let model_dir  = cache_root.join(repo_to_cache_dir(repo_id));
    let snapshots  = model_dir.join("snapshots");
    if !snapshots.exists() { return None; }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snapshots).ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    entries.sort_by(|a, b| {
        let ma = a.metadata().and_then(|m| m.modified()).ok();
        let mb = b.metadata().and_then(|m| m.modified()).ok();
        mb.cmp(&ma)
    });
    entries.into_iter().next()
}

fn snapshot_weight_files(snap: &Path) -> Result<Vec<PathBuf>> {
    let single = snap.join("model.safetensors");
    if single.exists() { return Ok(vec![single]); }
    let index = snap.join("model.safetensors.index.json");
    if index.exists() {
        let txt  = std::fs::read_to_string(&index)?;
        let val: serde_json::Value = serde_json::from_str(&txt)?;
        let wmap = val["weight_map"].as_object().context("weight_map missing")?;
        let mut shards: Vec<String> = wmap.values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        shards.sort(); shards.dedup();
        let paths: Vec<PathBuf> = shards.iter().map(|s| snap.join(s)).collect();
        if paths.iter().all(|p| p.exists()) { return Ok(paths); }
    }
    bail!("No safetensors weights found in snapshot '{}'", snap.display())
}

async fn find_or_download_base(repo_id: &str) -> Result<Vec<PathBuf>> {
    if let Some(snap) = find_cached_snapshot(repo_id) {
        println!("  Cache hit  : {}", snap.display());
        match snapshot_weight_files(&snap) {
            Ok(files) => return Ok(files),
            Err(e)    => println!("  Cache incomplete ({e}), falling back to Hub…"),
        }
    } else {
        println!("  No local cache for '{}', downloading from Hub…", repo_id);
    }
    download_from_hub(repo_id).await
}

async fn download_from_hub(repo_id: &str) -> Result<Vec<PathBuf>> {
    use hf_hub::api::tokio::ApiBuilder;
    let api  = ApiBuilder::new().with_progress(true).build()
        .context("Failed to build HF Hub API client")?;
    let repo = api.model(repo_id.to_string());
    if let Ok(p) = repo.get("model.safetensors").await { return Ok(vec![p]); }
    let idx = repo.get("model.safetensors.index.json").await
        .with_context(|| format!("Cannot fetch model weights from Hub for '{}'", repo_id))?;
    let val: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(idx)?)?;
    let wmap = val["weight_map"].as_object().context("weight_map missing in shard index")?;
    let mut shards: Vec<String> = wmap.values()
        .filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
    shards.sort(); shards.dedup();
    let mut paths = Vec::new();
    for s in shards {
        paths.push(repo.get(&s).await
            .with_context(|| format!("Cannot download shard '{}'", s))?);
    }
    Ok(paths)
}

async fn load_model_config(base_files: &[PathBuf], repo_id: &str, max_seq_len: usize) -> Result<Phi3Config> {
    let json = if let Some(dir) = base_files.first().and_then(|p| p.parent()) {
        let cfg_path = dir.join("config.json");
        if cfg_path.exists() {
            println!("  Config     : local ({})", cfg_path.display());
            std::fs::read_to_string(&cfg_path)?
        } else {
            download_config_json(repo_id).await?
        }
    } else {
        download_config_json(repo_id).await?
    };
    parse_phi3_config(&json, max_seq_len)
}

async fn download_config_json(repo_id: &str) -> Result<String> {
    println!("  Config     : downloading config.json from Hub…");
    use hf_hub::api::tokio::ApiBuilder;
    let api  = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(repo_id.to_string());
    let p    = repo.get("config.json").await
        .with_context(|| format!("Cannot fetch config.json from '{}'", repo_id))?;
    std::fs::read_to_string(p).context("Cannot read downloaded config.json")
}

async fn load_tokenizer(adapter_path: &Path, repo_id: &str) -> Result<Tokenizer> {
    if let Some(snap) = find_cached_snapshot(repo_id) {
        let cached = snap.join("tokenizer.json");
        if cached.exists() {
            println!("  Tokenizer  : HF cache");
            match load_tok_file(&cached) {
                Ok(t) => return Ok(t),
                Err(e) => println!("  Tokenizer cache load failed ({e}), trying adapter dir…"),
            }
        }
    }
    let local = adapter_path.join("tokenizer.json");
    if local.exists() {
        println!("  Tokenizer  : adapter dir");
        match load_tok_file(&local) {
            Ok(t) => return Ok(t),
            Err(e) => println!("  Adapter tokenizer load failed ({e}), downloading…"),
        }
    }
    println!("  Tokenizer  : downloading from Hub…");
    use hf_hub::api::tokio::ApiBuilder;
    let api  = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(repo_id.to_string());
    let p    = repo.get("tokenizer.json").await
        .context("Cannot download tokenizer.json")?;
    load_tok_file(&p)
}

fn load_tok_file(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(
        "Tokenizer load failed for '{}': {e}", path.display()
    ))
}

// ── Tensor / VarMap ───────────────────────────────────────────────────────────

fn load_safetensors_map(
    files:  &[impl AsRef<Path>],
    dtype:  DType,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut map = HashMap::new();
    for f in files {
        for (name, t) in candle_core::safetensors::load(f.as_ref(), device)? {
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
        .filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).collect();
    if subs.len() == 1 {
        let sub = subs[0].path();
        if sub.join("adapter_config.json").exists() || sub.join("config.json").exists() {
            println!("  Adapter dir: {}", sub.display());
            return Ok(sub);
        }
    }
    bail!("adapter_config.json not found in '{}'", path.display())
}

fn find_eos(tok: &Tokenizer) -> u32 {
    for name in &["<|end|>", "<|endoftext|>", "</s>"] {
        if let Some(id) = tok.token_to_id(name) { return id; }
    }
    32000
}
