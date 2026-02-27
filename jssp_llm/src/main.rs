// src/main.rs
//
// JSSP LLM Solver — pure Rust, no Python FFI.
// Mirrors the original Colab notebook (cells 11c / 11d / benchmark loop).
//
// Usage:
//   cargo run --release -- --help
//   cargo run --release -- smoke-test
//   cargo run --release -- benchmark --samples 5
//   cargo run --release -- solve --jobs "0,3,1,5,2,2;2,4,0,2,1,3"

mod jssp;        // problem types, formatting, validation
mod solver;      // pure-Rust greedy / branch-and-bound reference solver
mod llm;         // candle-based LLM inference (Phi-3 + LoRA)
mod prompt;      // prompt builder (mirrors build_prompt / format_job_centric)
mod parser;      // regex schedule parser  (mirrors parse_output)
mod benchmarks;  // standard JSSP benchmark instances (ft06, la01, …)

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

use anyhow::Result;
use clap::{Parser, Subcommand};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "jssp_llm", about = "JSSP solver powered by a LoRA LLM (pure Rust)")]
struct Cli {
    /// Path to the LoRA adapter directory (contains adapter_config.json + safetensors)
    #[arg(long, default_value = "./jssp_phi3_lora")]
    adapter_dir: String,

    /// Run on CUDA device index (-1 = CPU)
    #[arg(long, default_value_t = -1)]
    device: i64,

    /// Model dtype: f32, f16, bf16
    /// Default: f16 on CUDA (safe for T4/V100), f32 on CPU.
    /// Use bf16 only on Ampere (A100) or newer.
    #[arg(long)]
    dtype: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Quick 3×3 smoke test
    SmokeTest,

    /// Full benchmark suite (ft06, la01, la02, la05, la10)
    Benchmark {
        #[arg(long, default_value_t = 5)]
        samples: usize,

        #[arg(long, default_value_t = 0.8)]
        temperature: f64,

        #[arg(long, default_value_t = 0.95)]
        top_p: f64,

        #[arg(long, default_value_t = 512)]
        max_new_tokens: usize,
    },

    /// Solve a custom instance supplied on the command line.
    /// Format: semicolon-separated jobs, each job is comma-separated (machine,duration) pairs.
    /// Example: "0,3,1,5,2,2;2,4,0,2,1,3"
    Solve {
        #[arg(long)]
        jobs: String,

        #[arg(long, default_value_t = 5)]
        samples: usize,

        #[arg(long, default_value_t = 0.8)]
        temperature: f64,

        #[arg(long, default_value_t = 0.95)]
        top_p: f64,

        #[arg(long, default_value_t = 512)]
        max_new_tokens: usize,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Resolve device
    let device = llm::make_device(cli.device)?;

    // Resolve dtype (F16 on CUDA by default — safe for T4; use --dtype bf16 for A100)
    let dtype = llm::parse_dtype(cli.dtype.as_deref(), &device)?;

    // Load model once; reuse across commands
    println!("Loading model from '{}' …", cli.adapter_dir);
    let mut model_bundle = llm::ModelBundle::load(&cli.adapter_dir, &device, dtype).await?;
    println!("✓ Model ready\n");

    match cli.command {
        Commands::SmokeTest => {
            let test_jobs = vec![
                vec![(0u32, 3u32), (1, 5), (2, 2)],
                vec![(2, 4), (0, 2), (1, 3)],
                vec![(1, 6), (2, 1), (0, 4)],
            ];
            run_single(
                "smoke-test 3×3",
                &test_jobs,
                None,
                5, 0.8, 0.95, 512,
                &mut model_bundle,
                &device,
            )?;
        }

        Commands::Benchmark { samples, temperature, top_p, max_new_tokens } => {
            let suite = benchmarks::standard_suite();
            let mut all_gaps: Vec<f64> = Vec::new();

            for inst in &suite {
                println!("\n{}", "=".repeat(55));
                println!("  {}  ({}×{})   optimal = {}",
                    inst.name.to_uppercase(),
                    inst.jobs.len(),
                    inst.n_machines(),
                    inst.optimal,
                );
                println!("{}", "=".repeat(55));

                if let Some(gap) = run_single(
                    &inst.name,
                    &inst.jobs,
                    Some(inst.optimal),
                    samples, temperature, top_p, max_new_tokens,
                    &mut model_bundle,
                    &device,
                )? {
                    all_gaps.push(gap);
                }
            }

            if !all_gaps.is_empty() {
                let avg = all_gaps.iter().sum::<f64>() / all_gaps.len() as f64;
                println!("\n{}", "=".repeat(55));
                println!("Average gap from optimal : {:.2}%", avg);
                println!("Paper reports            : ~8.92%  (Phi-3, s=10, 10×10)");
            }
        }

        Commands::Solve { jobs: jobs_str, samples, temperature, top_p, max_new_tokens } => {
            let jobs = jssp::parse_jobs_str(&jobs_str)?;
            run_single(
                "custom",
                &jobs,
                None,
                samples, temperature, top_p, max_new_tokens,
                &mut model_bundle,
                &device,
            )?;
        }
    }

    Ok(())
}

// ── Shared run helper ─────────────────────────────────────────────────────────

fn run_single(
    _label: &str,
    jobs: &[Vec<(u32, u32)>],
    optimal: Option<u32>,
    n_samples: usize,
    temperature: f64,
    top_p: f64,
    max_new_tokens: usize,
    bundle: &mut llm::ModelBundle,
    device: &candle_core::Device,
) -> Result<Option<f64>> {
    use std::time::Instant;

    let t0 = Instant::now();
    let (best, valid_count) = llm::solve_with_llm(
        jobs, n_samples, temperature, top_p, max_new_tokens, bundle, device,
    )?;
    let elapsed = t0.elapsed().as_secs_f64();

    // Reference solution from the built-in greedy solver
    let ref_ms = solver::solve_greedy(jobs);

    if let Some(opt) = optimal {
        println!("  OR-Tools optimal : {}", opt);
    } else {
        println!("  Greedy reference : {}", ref_ms);
    }

    if let Some(ref res) = best {
        let baseline = optimal.unwrap_or(ref_ms);
        let gap = (res.makespan as f64 - baseline as f64) / baseline as f64 * 100.0;
        println!("  LLM best         : {}  (gap {:.1}%)", res.makespan, gap);
        println!("  Valid candidates : {} / {}", valid_count, n_samples);
        println!("  Time             : {:.1}s", elapsed);
        println!("\n  Schedule (start times per operation):");
        for (j, times) in res.start_times.iter().enumerate() {
            let times_str: Vec<String> = times.iter().map(|t| t.to_string()).collect();
            println!("    Job {:2}: {}", j, times_str.join("  "));
        }
        return Ok(Some(gap));
    } else {
        println!("  No valid solution in {} samples  ({:.1}s)", n_samples, elapsed);
        println!("  Tip: increase --samples or adjust --temperature");
    }

    Ok(None)
}