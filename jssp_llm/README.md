# JSSP LLM Solver — Pure Rust

A complete migration of the original Python/Colab notebook into Rust with **zero Python FFI**.
Uses [candle](https://github.com/huggingface/candle) (Hugging Face's ML framework) for LLM
inference and a hand-written greedy scheduler as the reference solver (replacing OR-Tools).

## Architecture

```
jssp_llm/
├── Cargo.toml
└── src/
    ├── main.rs        – CLI entry point, benchmark loop (mirrors cells 11c/11d)
    ├── jssp.rs        – Problem types, formatting, constraint validation
    ├── prompt.rs      – Prompt builder (Phi-3 chat template)
    ├── parser.rs      – Regex schedule parser (parse_output equivalent)
    ├── llm.rs         – Candle inference engine (model load, generate, sample)
    ├── solver.rs      – Pure-Rust greedy reference solver (replaces OR-Tools)
    └── benchmarks.rs  – ft06, la01, la02, la05, la10 benchmark instances
```

## Prerequisites

| Requirement | Notes |
|---|---|
| Rust ≥ 1.77 | `rustup update stable` |
| A fine-tuned adapter | Directory with `config.json`, `tokenizer.json`, `*.safetensors` |
| NVIDIA GPU (optional) | Build with `--features cuda` for acceleration |

## Building

```bash
# CPU-only (default)
cargo build --release

# CUDA (NVIDIA GPU) — requires CUDA toolkit ≥ 11.8
cargo build --release --features cuda

# Apple Silicon
cargo build --release --features metal
```

## Preparing the Adapter Directory

Export your LoRA adapter from Python **once**, then never need Python again:

```python
# Run this in your Colab/Python environment to produce merged safetensors
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base   = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model  = PeftModel.from_pretrained(base, "./jssp_phi3_lora")
merged = model.merge_and_unload()

merged.save_pretrained("./jssp_phi3_merged", safe_serialization=True)
AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct").save_pretrained("./jssp_phi3_merged")
```

Then point `--adapter-dir` at `./jssp_phi3_merged`.

The directory must contain:
```
jssp_phi3_merged/
├── config.json                  ← model architecture
├── tokenizer.json               ← HF fast tokenizer
├── tokenizer_config.json        ← (optional but recommended)
├── model.safetensors            ← merged weights (or shards model-00001-of-N.safetensors)
└── adapter_config.json          ← (optional, for reference)
```

## Usage

```bash
# Quick smoke test — 3×3 problem
cargo run --release -- smoke-test

# Full benchmark suite (ft06, la01, la02, la05, la10)
cargo run --release -- benchmark --samples 5 --temperature 0.8

# Solve a custom instance
# Format: semicolon-separated jobs, each job = comma-separated (machine,duration) pairs
cargo run --release -- solve \
    --jobs "0,3,1,5,2,2;2,4,0,2,1,3;1,6,2,1,0,4" \
    --samples 10

# Use a custom adapter directory and CUDA device 0
cargo run --release --features cuda -- \
    --adapter-dir /path/to/merged_model \
    --device 0 \
    benchmark --samples 10
```

## CLI Reference

```
jssp_llm [OPTIONS] <COMMAND>

OPTIONS:
  --adapter-dir <PATH>   Path to merged adapter directory [default: ./jssp_phi3_lora]
  --device <N>           CUDA device index; -1 = CPU [default: -1]

COMMANDS:
  smoke-test             Quick 3×3 sanity check
  benchmark              Run ft06, la01, la02, la05, la10
    --samples <N>        Candidate solutions per instance [default: 5]
    --temperature <F>    Sampling temperature [default: 0.8]
    --top-p <F>          Nucleus sampling p [default: 0.95]
    --max-new-tokens <N> Token budget per sample [default: 512]
  solve                  Solve a custom instance
    --jobs <STRING>      Problem encoding (see format above)
    --samples / --temperature / --top-p / --max-new-tokens  (same as benchmark)
```

## Expected Output

```
Loading model from './jssp_phi3_merged' …
  Tokenizer  : ./jssp_phi3_merged/tokenizer.json
  Config     : ./jssp_phi3_merged/config.json
  Weights    : 1 file(s)
  Device     : Cpu
  DType      : F32
✓ Model ready

=======================================================
  FT06  (6×6)   optimal makespan = 55
=======================================================
  Sample 1/5 …
    raw output: Job 0: 0 1 4 10 17 20
    ...
  LLM best         : 57  (gap 3.6%)
  Valid candidates : 4 / 5
  Time             : 12.3s

  Schedule preview:
    Job 0: 0 1 4 10 17 20
    ...

=======================================================
Average gap from optimal : 9.21%
Paper reports            : ~8.92%  (Phi-3, s=10, 10×10)
```

## Design Notes

### Why candle instead of Python/PyTorch?
- **Zero FFI**: no `pyo3`, no shared libraries, single static binary.
- **Safe**: Rust's ownership model prevents data races in the sampling loop.
- **Portable**: compile once, run anywhere (Linux, macOS, Windows).

### Replacing OR-Tools
OR-Tools is a C++ library with a Python wrapper. In this Rust port, the
*reference makespan* for custom instances is computed by `solver::solve_greedy`
(an SPT dispatching rule). For the paper's benchmark instances, the known
optimal values are hard-coded in `benchmarks.rs`—so OR-Tools is not needed.

If you need a tighter reference solver, integrate the
[`or-tools` crate](https://crates.io/crates/or-tools) or implement the
Shifting Bottleneck heuristic (scaffold in `solver.rs`).

### LoRA weight loading
`candle` loads `.safetensors` files directly via memory-mapped I/O. If you
have **separate** base + adapter weights (rather than merged), you will need to
implement LoRA weight merging in `llm.rs::ModelBundle::load`. The simplest
approach is to merge in Python once (see *Preparing the Adapter Directory*
above) and load the merged weights here.

### Quantisation
The Colab notebook uses 4-bit quantisation (bitsandbytes). Candle supports
GGUF/GGML quantisation. To use 4-bit weights:
1. Convert the merged model to GGUF with `llama.cpp`.
2. Switch `llm.rs` to use `candle_transformers::models::quantized_phi3`.

## License
MIT
