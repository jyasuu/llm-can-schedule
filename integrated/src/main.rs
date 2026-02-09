// Complete JSSP Training + Scheduling in One Rust Binary
// Integrated pipeline: Load Data → Train → Schedule → Evaluate

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rand::Rng;
use std::fs;

mod dataset;
mod inference;
mod metrics;
mod model;
mod scheduler;
mod trainer;

use dataset::{JSPInstance, StarjobDataset};
use inference::ScheduleInference;
use model::{LLMConfig, TransformerLLM};
use scheduler::{GreedyScheduler, SchedulingProblem};
use trainer::ModelTrainer;

// ============================================================================
// CLI Configuration
// ============================================================================

#[derive(Parser)]
#[command(name = "JSSP Scheduler")]
#[command(about = "Train LLM for Job Shop Scheduling and Schedule Problems", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train model on JSSP dataset
    Train {
        /// Path to dataset JSON file
        #[arg(short, long)]
        dataset: String,

        /// Number of training samples to use
        #[arg(short, long, default_value = "1000")]
        samples: usize,

        /// Output model path
        #[arg(short, long, default_value = "./trained_model")]
        output: String,

        /// Learning rate
        #[arg(long, default_value = "1e-4")]
        lr: f32,

        /// Batch size
        #[arg(short, long, default_value = "8")]
        batch_size: usize,

        /// Number of epochs
        #[arg(short, long, default_value = "3")]
        epochs: usize,
    },

    /// Schedule a problem using trained model
    Schedule {
        /// Path to trained model
        #[arg(short, long)]
        model: String,

        /// Problem JSON file (or create demo)
        #[arg(short, long)]
        problem: Option<String>,

        /// Number of jobs
        #[arg(long, default_value = "5")]
        jobs: usize,

        /// Number of machines
        #[arg(long, default_value = "5")]
        machines: usize,
    },

    /// Full pipeline: train then schedule
    Pipeline {
        /// Path to training dataset
        #[arg(short, long)]
        dataset: String,

        /// Training samples
        #[arg(short, long, default_value = "1000")]
        train_samples: usize,

        /// Test problems count
        #[arg(long, default_value = "10")]
        test_count: usize,

        /// Output directory
        #[arg(short, long, default_value = "./jssp_output")]
        output: String,
    },

    /// Generate demo dataset
    GenData {
        /// Number of instances to generate
        #[arg(short, long, default_value = "100")]
        count: usize,

        /// Output file
        #[arg(short, long, default_value = "./demo_data.json")]
        output: String,
    },

    /// Benchmark model performance
    Benchmark {
        /// Path to trained model
        #[arg(short, long)]
        model: String,

        /// Number of test problems
        #[arg(short, long, default_value = "50")]
        tests: usize,
    },
}

// ============================================================================
// Main Application
// ============================================================================

fn main() -> Result<()> {
    env_logger::init();

    println!("\n{}", "=".repeat(70));
    println!("JSSP LLM Scheduler - Integrated Training & Scheduling");
    println!("Based on: 'LLMs can Schedule' (arXiv:2408.06993)");
    println!("{}\n", "=".repeat(70));

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            dataset,
            samples,
            output,
            lr,
            batch_size,
            epochs,
        } => {
            train_command(&dataset, samples, &output, lr, batch_size, epochs)?;
        }

        Commands::Schedule {
            model,
            problem,
            jobs,
            machines,
        } => {
            schedule_command(&model, problem.as_deref(), jobs, machines)?;
        }

        Commands::Pipeline {
            dataset,
            train_samples,
            test_count,
            output,
        } => {
            pipeline_command(&dataset, train_samples, test_count, &output)?;
        }

        Commands::GenData { count, output } => {
            generate_data_command(count, &output)?;
        }

        Commands::Benchmark { model, tests } => {
            benchmark_command(&model, tests)?;
        }
    }

    println!("\n{}\nDone!\n{}", "=".repeat(70), "=".repeat(70));
    Ok(())
}

// ============================================================================
// Command Implementations
// ============================================================================

/// Train model on dataset
fn train_command(
    dataset_path: &str,
    num_samples: usize,
    output_path: &str,
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
) -> Result<()> {
    println!("{}\nTraining Model\n{}", "=".repeat(70), "=".repeat(70));

    // Load dataset
    println!("Loading dataset from: {}", dataset_path);
    let dataset =
        StarjobDataset::load(dataset_path, num_samples).context("Failed to load dataset")?;

    println!("Loaded {} training examples", dataset.instances.len());

    // Create model
    println!("\nInitializing model...");
    let config = LLMConfig {
        vocab_size: 10000,
        hidden_dim: 512,
        num_layers: 4,
        num_heads: 8,
        max_seq_len: 512,
        dropout_rate: 0.1,
    };

    let model = TransformerLLM::new(config).context("Failed to create model")?;

    // Create trainer
    let mut trainer = ModelTrainer::new(model, learning_rate, batch_size);

    // Train
    println!("\nTraining configuration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", epochs);
    println!("  Training samples: {}", dataset.instances.len());

    trainer.train(&dataset, epochs).context("Training failed")?;

    // Save model
    println!("\nSaving model to: {}", output_path);
    trainer.save_checkpoint(output_path)?;

    println!("✓ Training complete!");
    Ok(())
}

/// Schedule a problem using trained model
fn schedule_command(
    model_path: &str,
    problem_path: Option<&str>,
    num_jobs: usize,
    num_machines: usize,
) -> Result<()> {
    println!("{}\nScheduling\n{}", "=".repeat(70), "=".repeat(70));

    // Load model
    println!("Loading model from: {}", model_path);
    let inference = ScheduleInference::load(model_path)?;

    // Get problem
    let problem = if let Some(path) = problem_path {
        println!("Loading problem from: {}", path);
        SchedulingProblem::from_json(path)?
    } else {
        println!("Generating demo problem: {}x{}", num_jobs, num_machines);
        SchedulingProblem::random(num_jobs, num_machines, 42)
    };

    println!("\nProblem Description:");
    println!("{}", problem.description());

    // Generate solution
    println!("\n--- Generating LLM Solution ---");
    let llm_solution = inference.schedule(&problem)?;
    println!("LLM Solution:\n{:?}", llm_solution);

    // Generate baseline
    println!("\n--- Generating Baseline Solution ---");
    let baseline_solution = GreedyScheduler::schedule(&problem);
    println!("Baseline Solution:\n{}", baseline_solution.description());

    // Compare
    println!("\n--- Comparison ---");
    let llm_makespan = llm_solution.makespan;
    let baseline_makespan = baseline_solution.makespan;

    println!("LLM Makespan: {}", llm_makespan);
    println!("Baseline Makespan: {}", baseline_makespan);

    if baseline_makespan > 0 {
        let improvement = 1.0 - (llm_makespan as f32 / baseline_makespan as f32);
        println!("Improvement: {:.2}%", improvement * 100.0);
    }

    Ok(())
}

/// Full pipeline: train, then schedule
fn pipeline_command(
    dataset_path: &str,
    train_samples: usize,
    test_count: usize,
    output_dir: &str,
) -> Result<()> {
    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Phase 1: Training
    println!("{}\nPhase 1: Training\n{}", "=".repeat(70), "=".repeat(70));

    let model_path = format!("{}/trained_model", output_dir);
    train_command(dataset_path, train_samples, &model_path, 1e-4, 8, 3)?;

    // Phase 2: Testing
    println!(
        "\n{}\nPhase 2: Testing on {} Problems\n{}",
        "=".repeat(70),
        test_count,
        "=".repeat(70)
    );

    let inference = ScheduleInference::load(&model_path)?;
    let mut results = Vec::new();

    for i in 0..test_count {
        println!("\nTest {}/{}:", i + 1, test_count);

        let problem = SchedulingProblem::random(5, 5, i as u64);
        println!("Problem: {}x{}", problem.num_jobs, problem.num_machines);

        // LLM solution
        let llm_solution = inference.schedule(&problem)?;
        let llm_makespan = llm_solution.makespan;

        // Baseline
        let baseline_solution = GreedyScheduler::schedule(&problem);
        let baseline_makespan = baseline_solution.makespan;

        // Calculate improvement
        let improvement = if baseline_makespan > 0 {
            1.0 - (llm_makespan as f32 / baseline_makespan as f32)
        } else {
            0.0
        };

        println!("  LLM Makespan: {}", llm_makespan);
        println!("  Baseline Makespan: {}", baseline_makespan);
        println!("  Improvement: {:.2}%", improvement * 100.0);

        results.push((llm_makespan, baseline_makespan, improvement));
    }

    // Summary statistics
    println!(
        "\n{}\nSummary Statistics\n{}",
        "=".repeat(70),
        "=".repeat(70)
    );

    let avg_improvement = results.iter().map(|(_, _, i)| i).sum::<f32>() / results.len() as f32;
    let max_improvement = results
        .iter()
        .map(|(_, _, i)| i)
        .fold(f32::NEG_INFINITY, |arg0: f32, other: &f32| {
            f32::max(arg0, *other)
        });
    let min_improvement = results
        .iter()
        .map(|(_, _, i)| i)
        .fold(f32::INFINITY, |arg0: f32, other: &f32| {
            f32::min(arg0, *other)
        });

    println!("Average Improvement: {:.2}%", avg_improvement * 100.0);
    println!("Max Improvement: {:.2}%", max_improvement * 100.0);
    println!("Min Improvement: {:.2}%", min_improvement * 100.0);

    // Save results
    let results_file = format!("{}/results.json", output_dir);
    let json = serde_json::json!({
        "test_count": test_count,
        "avg_improvement": avg_improvement,
        "max_improvement": max_improvement,
        "min_improvement": min_improvement,
        "model_path": model_path,
    });

    fs::write(&results_file, serde_json::to_string_pretty(&json)?)?;
    println!("\nResults saved to: {}", results_file);

    Ok(())
}

/// Generate demo dataset
fn generate_data_command(count: usize, output_path: &str) -> Result<()> {
    println!("Generating {} demo instances...", count);

    let mut instances = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..count {
        let num_jobs = rng.gen_range(2..8);
        let num_machines = rng.gen_range(2..8);

        let problem = SchedulingProblem::random(num_jobs, num_machines, i as u64);

        let instance = JSPInstance {
            num_jobs,
            num_machines,
            input: problem.description(),
            output: format!("Solution for {}x{} problem", num_jobs, num_machines),
        };

        instances.push(instance);
    }

    let dataset = dataset::RawDataset { data: instances };
    let json = serde_json::to_string_pretty(&dataset)?;
    fs::write(output_path, json)?;

    println!("✓ Generated {} instances to: {}", count, output_path);
    Ok(())
}

/// Benchmark model on test set
fn benchmark_command(model_path: &str, num_tests: usize) -> Result<()> {
    println!("{}\nBenchmarking Model\n{}", "=".repeat(70), "=".repeat(70));

    let inference = ScheduleInference::load(model_path)?;

    let mut makespans = Vec::new();
    let mut times = Vec::new();

    for i in 0..num_tests {
        let problem = SchedulingProblem::random(5, 5, i as u64);

        let start = std::time::Instant::now();
        let solution = inference.schedule(&problem)?;
        let elapsed = start.elapsed();

        let metrics = solution.metrics();
        makespans.push(metrics.makespan);
        times.push(elapsed.as_millis());

        if (i + 1) % 10 == 0 {
            println!("Completed {}/{} tests", i + 1, num_tests);
        }
    }

    // Statistics
    println!("\n{}\nResults\n{}", "=".repeat(70), "=".repeat(70));

    let avg_makespan = makespans.iter().sum::<usize>() as f32 / makespans.len() as f32;
    let avg_time = times.iter().sum::<u128>() as f32 / times.len() as f32;

    println!("Average Makespan: {:.2}", avg_makespan);
    println!("Average Time: {:.2}ms", avg_time);
    println!("Min Makespan: {}", makespans.iter().min().unwrap());
    println!("Max Makespan: {}", makespans.iter().max().unwrap());

    let throughput = 1000.0 / avg_time;
    println!("Throughput: {:.2} problems/second", throughput);

    Ok(())
}

// ============================================================================
// Module declarations
// ============================================================================

// These would be in separate files in a real project
