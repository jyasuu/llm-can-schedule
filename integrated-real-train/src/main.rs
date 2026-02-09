
// Complete JSSP Training + Scheduling with REAL Candle Training
// Actual model training with loss computation and backpropagation

use anyhow::{Result, Context};
use std::fs;
use clap::{Parser, Subcommand};
use rand::Rng;

mod dataset;
mod model;
mod trainer;
mod inference;
mod scheduler;
mod metrics;

use dataset::{JSPInstance, StarjobDataset};
use model::{TransformerLLM, LLMConfig};
use trainer::CandelTrainer;
use inference::ScheduleInference;
use scheduler::{SchedulingProblem, GreedyScheduler};

#[derive(Parser)]
#[command(name = "JSSP Scheduler")]
#[command(about = "Real LLM Training + Job Shop Scheduling with Candle")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train model with REAL loss computation
    Train {
        #[arg(short, long)]
        dataset: String,

        #[arg(short, long, default_value = "1000")]
        samples: usize,

        #[arg(short, long, default_value = "./trained_model")]
        output: String,

        #[arg(long, default_value = "1e-4")]
        lr: f64,

        #[arg(short, long, default_value = "4")]
        batch_size: usize,

        #[arg(short, long, default_value = "3")]
        epochs: usize,
    },

    /// Schedule a problem
    Schedule {
        #[arg(short, long)]
        model: String,

        #[arg(short, long)]
        problem: Option<String>,

        #[arg(long, default_value = "5")]
        jobs: usize,

        #[arg(long, default_value = "5")]
        machines: usize,
    },

    /// Full training + testing pipeline
    Pipeline {
        #[arg(short, long)]
        dataset: String,

        #[arg(short, long, default_value = "1000")]
        train_samples: usize,

        #[arg(long, default_value = "10")]
        test_count: usize,

        #[arg(short, long, default_value = "./jssp_output")]
        output: String,
    },

    /// Generate demo dataset
    GenData {
        #[arg(short, long, default_value = "100")]
        count: usize,

        #[arg(short, long, default_value = "./demo_data.json")]
        output: String,
    },

    /// Benchmark model
    Benchmark {
        #[arg(short, long)]
        model: String,

        #[arg(short, long, default_value = "50")]
        tests: usize,
    },
}

fn main() -> Result<()> {
    println!("\n{}", "=".repeat(80));
    println!("JSSP LLM Scheduler - REAL Candle Training");
    println!("With actual loss computation and backpropagation!");
    println!("{}\n", "=".repeat(80));

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

    println!("\n{}\nDone!\n{}", "=".repeat(80), "=".repeat(80));
    Ok(())
}

fn train_command(
    dataset_path: &str,
    num_samples: usize,
    output_path: &str,
    learning_rate: f64,
    batch_size: usize,
    epochs: usize,
) -> Result<()> {
    println!("{}\nTraining with REAL Candle Model\n{}", "=".repeat(80), "=".repeat(80));

    // Load dataset
    println!("Loading dataset from: {}", dataset_path);
    let dataset = StarjobDataset::load(dataset_path, num_samples)
        .context("Failed to load dataset")?;

    println!("Loaded {} training examples", dataset.instances.len());

    // Create model with Candle
    println!("\nInitializing Candle model...");
    let config = LLMConfig {
        vocab_size: 10000,
        hidden_dim: 512,
        num_layers: 4,
        num_heads: 8,
        max_seq_len: 512,
        dropout_rate: 0.1,
    };

    let device = candle::Device::cuda_if_available(0).unwrap_or(candle::Device::Cpu);
    println!("Using device: {:?}", device);

    let model = TransformerLLM::new(config, &device)
        .context("Failed to create model")?;

    // Create trainer with REAL loss computation
    println!("\nInitializing trainer with Adam optimizer...");
    let mut trainer = CandelTrainer::new(model, learning_rate, batch_size, &device)?;

    println!("\nTraining configuration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", epochs);
    println!("  Training samples: {}", dataset.instances.len());
    println!("  Device: {:?}", device);

    // REAL training with loss computation
    trainer.train(&dataset, epochs).context("Training failed")?;

    // Save model
    println!("\nSaving model to: {}", output_path);
    trainer.save_checkpoint(output_path)?;

    println!("✓ Training complete!");
    Ok(())
}

fn schedule_command(
    model_path: &str,
    problem_path: Option<&str>,
    num_jobs: usize,
    num_machines: usize,
) -> Result<()> {
    println!("{}\nScheduling\n{}", "=".repeat(80), "=".repeat(80));

    println!("Loading model from: {}", model_path);
    let inference = ScheduleInference::load(model_path)?;

    let problem = if let Some(path) = problem_path {
        println!("Loading problem from: {}", path);
        SchedulingProblem::from_json(path)?
    } else {
        println!("Generating demo problem: {}x{}", num_jobs, num_machines);
        SchedulingProblem::random(num_jobs, num_machines, 42)
    };

    println!("\nProblem:");
    println!("{}", problem.description());

    // Generate solution
    println!("\n--- Generating Solution ---");
    let solution = inference.schedule(&problem)?;
    println!("Solution:\n{}", solution.description);

    // Baseline
    println!("\n--- Baseline (Greedy) ---");
    let baseline = GreedyScheduler::schedule(&problem);
    println!("Baseline:\n{}", baseline.description());

    println!("\n--- Comparison ---");
    let improvement = if baseline.makespan > 0 {
        1.0 - (solution.makespan as f32 / baseline.makespan as f32)
    } else {
        0.0
    };
    
    println!("LLM Makespan: {}", solution.makespan);
    println!("Baseline Makespan: {}", baseline.makespan);
    println!("Improvement: {:.2}%", improvement * 100.0);

    Ok(())
}

fn pipeline_command(
    dataset_path: &str,
    train_samples: usize,
    test_count: usize,
    output_dir: &str,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;

    println!("{}\nPhase 1: Training\n{}", "=".repeat(80), "=".repeat(80));
    
    let model_path = format!("{}/trained_model", output_dir);
    train_command(dataset_path, train_samples, &model_path, 1e-4, 4, 3)?;

    println!("\n{}\nPhase 2: Testing ({} problems)\n{}", "=".repeat(80), test_count, "=".repeat(80));

    let inference = ScheduleInference::load(&model_path)?;
    let mut improvements = Vec::new();

    for i in 0..test_count {
        println!("\nTest {}/{}:", i + 1, test_count);
        
        let problem = SchedulingProblem::random(5, 5, i as u64);
        let solution = inference.schedule(&problem)?;
        let baseline = GreedyScheduler::schedule(&problem);
        
        let improvement = if baseline.makespan > 0 {
            1.0 - (solution.makespan as f32 / baseline.makespan as f32)
        } else {
            0.0
        };
        
        println!("  LLM: {} | Baseline: {} | Improvement: {:.2}%", 
                 solution.makespan, baseline.makespan, improvement * 100.0);
        improvements.push(improvement);
    }

    println!("\n{}\nStatistics\n{}", "=".repeat(80), "=".repeat(80));
    
    let avg = improvements.iter().sum::<f32>() / improvements.len() as f32;
    let max = improvements.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = improvements.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    
    println!("Average Improvement: {:.2}%", avg * 100.0);
    println!("Max Improvement: {:.2}%", max * 100.0);
    println!("Min Improvement: {:.2}%", min * 100.0);

    let results = serde_json::json!({
        "test_count": test_count,
        "avg_improvement": avg,
        "max_improvement": max,
        "min_improvement": min,
    });

    fs::write(
        format!("{}/results.json", output_dir),
        serde_json::to_string_pretty(&results)?
    )?;

    Ok(())
}

fn generate_data_command(count: usize, output_path: &str) -> Result<()> {
    println!("Generating {} demo instances...", count);

    let mut instances = Vec::new();
    let mut _rng = rand::thread_rng();

    for i in 0..count {
        let num_jobs = 2 + (i % 5);
        let num_machines = 2 + (i % 5);

        let problem = SchedulingProblem::random(num_jobs, num_machines, i as u64);

        instances.push(JSPInstance {
            num_jobs,
            num_machines,
            input: problem.description(),
            output: format!("Solution for {}x{} problem", num_jobs, num_machines),
        });
    }

    let dataset = dataset::RawDataset { data: instances };
    fs::write(output_path, serde_json::to_string_pretty(&dataset)?)?;

    println!("✓ Generated {} instances to: {}", count, output_path);
    Ok(())
}

fn benchmark_command(model_path: &str, num_tests: usize) -> Result<()> {
    println!("{}\nBenchmarking\n{}", "=".repeat(80), "=".repeat(80));

    let inference = ScheduleInference::load(model_path)?;

    let mut times = Vec::new();
    let mut makespans = Vec::new();

    for i in 0..num_tests {
        let problem = SchedulingProblem::random(5, 5, i as u64);
        
        let start = std::time::Instant::now();
        let solution = inference.schedule(&problem)?;
        let elapsed = start.elapsed();

        times.push(elapsed.as_millis());
        makespans.push(solution.makespan);

        if (i + 1) % 10 == 0 {
            println!("Completed {}/{}", i + 1, num_tests);
        }
    }

    println!("\n{}\nResults\n{}", "=".repeat(80), "=".repeat(80));
    
    let avg_time = times.iter().sum::<u128>() as f32 / times.len() as f32;
    let avg_makespan = makespans.iter().sum::<usize>() as f32 / makespans.len() as f32;
    let throughput = 1000.0 / avg_time;

    println!("Average Time: {:.2}ms", avg_time);
    println!("Average Makespan: {:.2}", avg_makespan);
    println!("Throughput: {:.2} problems/second", throughput);
    println!("Min Makespan: {}", makespans.iter().min().unwrap());
    println!("Max Makespan: {}", makespans.iter().max().unwrap());

    Ok(())
}