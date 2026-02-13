use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

mod data;
mod jssp;
mod lora;
mod starjob;

use data::load_sample_data;
use jssp::{JSSPInstance, JSSPSolver, Schedule};
use lora::{LoRA, LoRAConfig};
use starjob::{load_and_analyze_starjob, load_starjob_by_size, load_starjob_dataset};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub dropout: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            epochs: 3,
            batch_size: 8,
            lora_rank: 8,
            lora_alpha: 16.0,
            dropout: 0.1,
        }
    }
}

struct JSSPLoRASolver {
    model: LoRA,
    device: Device,
    config: TrainingConfig,
}

impl JSSPLoRASolver {
    fn new(device: Device, config: TrainingConfig) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let lora_config = LoRAConfig {
            rank: config.lora_rank,
            alpha: config.lora_alpha,
            dropout: config.dropout,
        };

        let model = LoRA::new(&vb, lora_config)?;

        Ok(Self {
            model,
            device,
            config,
        })
    }

    fn train_on_instance(
        &mut self,
        instance: &JSSPInstance,
        optimal_schedule: &Schedule,
    ) -> Result<f32> {
        // Encode the JSSP instance
        let input = instance.encode()?;
        let input_tensor = Tensor::new(&input[..], &self.device)?;

        // Forward pass
        let logits = self.model.forward(&input_tensor)?;

        // Compute loss (MSE between predicted and optimal schedule)
        let target = optimal_schedule.to_tensor(&self.device)?;
        let loss = (logits - &target)?;
        let loss = (&loss * &loss)?;
        let loss = loss.mean_all()?;

        Ok(loss.to_scalar::<f32>()?)
    }

    fn solve(&self, instance: &JSSPInstance) -> Result<Schedule> {
        let input = instance.encode()?;
        let input_tensor = Tensor::new(&input[..], &self.device)?;

        let logits = self.model.forward(&input_tensor)?;
        let schedule = Schedule::from_tensor(&logits)?;

        Ok(schedule)
    }
}

/// Run example with sample data (built-in benchmarks)
fn example_with_sample_data() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║      JSSP LoRA Solver - Sample Data Example          ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    let config = TrainingConfig::default();
    println!("Configuration: {:?}\n", config);

    // Load sample JSSP instances
    let dataset = load_sample_data()?;
    println!("Loaded {} sample JSSP instances\n", dataset.instances.len());

    // Initialize the LoRA solver
    let mut solver = JSSPLoRASolver::new(device.clone(), config.clone())?;
    println!(
        "Initialized LoRA solver with rank={}, alpha={}\n",
        config.lora_rank, config.lora_alpha
    );

    // Training loop
    println!("Starting training...\n");
    for epoch in 0..config.epochs {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (idx, instance) in dataset.instances.iter().enumerate() {
            let optimal_schedule = &dataset.optimal_schedules[idx];

            match solver.train_on_instance(&instance, optimal_schedule) {
                Ok(loss) => {
                    total_loss += loss;
                    count += 1;
                }
                Err(e) => {
                    eprintln!("Error training on instance {}: {}", idx, e);
                }
            }

            if (idx + 1) % 5 == 0 {
                println!("  Processed {} instances", idx + 1);
            }
        }

        let avg_loss = if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        };
        println!("Epoch {}: Average Loss = {:.6}\n", epoch + 1, avg_loss);
    }

    // Test the trained model
    println!("\n=== Testing Trained Model ===\n");

    if let Some(test_instance) = dataset.instances.first() {
        println!("Test Instance:");
        println!("  Jobs: {}", test_instance.num_jobs);
        println!("  Machines: {}", test_instance.num_machines);

        match solver.solve(&test_instance) {
            Ok(schedule) => {
                println!("Generated Schedule:");
                println!("  Makespan: {}", schedule.makespan);
                println!("  Job sequence: {:?}\n", schedule.job_sequence);

                if let Some(optimal) = dataset.optimal_schedules.first() {
                    println!("Optimal Schedule (reference):");
                    println!("  Makespan: {}", optimal.makespan);
                    let efficiency = (optimal.makespan as f32) / (schedule.makespan as f32);
                    println!("Solution Quality: {:.2}% of optimal\n", efficiency * 100.0);
                }
            }
            Err(e) => {
                eprintln!("Error solving instance: {}", e);
            }
        }
    }

    Ok(())
}

/// Run example with Starjob dataset
fn example_with_starjob_data(json_path: &str, limit: Option<usize>) -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║      JSSP LoRA Solver - Starjob Dataset              ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // Analyze dataset first
    println!("\n=== Dataset Analysis ===");
    match load_and_analyze_starjob(json_path, limit) {
        Ok(stats) => {
            stats.print_summary();
        }
        Err(e) => {
            eprintln!("Error analyzing dataset: {}", e);
        }
    }

    // Load Starjob dataset
    println!("\n=== Loading Starjob Dataset ===");
    let starjob = load_starjob_dataset(json_path, limit)?;
    println!(
        "Loaded {} instances from Starjob\n",
        starjob.instances.len()
    );

    // Initialize solver
    let config = TrainingConfig {
        learning_rate: 1e-4,
        epochs: 2, // Fewer epochs for large dataset
        batch_size: 16,
        lora_rank: 16, // Larger rank for larger problems
        lora_alpha: 32.0,
        dropout: 0.1,
    };

    let mut solver = JSSPLoRASolver::new(device.clone(), config.clone())?;
    println!("Initialized LoRA solver with rank={}\n", config.lora_rank);

    // Training on Starjob data
    println!("Starting training on Starjob instances...\n");

    for epoch in 0..config.epochs {
        let mut total_loss = 0.0;
        let mut count = 0;
        let mut instance_count = 0;

        for (idx, instance) in starjob.instances.iter().take(100).enumerate() {
            // Use greedy solver as reference for loss computation
            match JSSPSolver::solve_nearest_neighbor(&instance) {
                Ok(reference_schedule) => {
                    match solver.train_on_instance(&instance, &reference_schedule) {
                        Ok(loss) => {
                            total_loss += loss;
                            count += 1;
                        }
                        Err(e) => {
                            eprintln!("Error training on instance {}: {}", idx, e);
                        }
                    }
                    instance_count += 1;
                }
                Err(e) => {
                    eprintln!("Error solving reference schedule: {}", e);
                }
            }

            if (idx + 1) % 20 == 0 {
                println!("  Processed {} instances", idx + 1);
            }
        }

        let avg_loss = if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        };
        println!(
            "Epoch {}: Average Loss = {:.6} ({} instances)\n",
            epoch + 1,
            avg_loss,
            instance_count
        );
    }

    // Test on some Starjob instances
    println!("\n=== Testing on Starjob Instances ===\n");

    for (idx, instance) in starjob.instances.iter().take(3).enumerate() {
        println!(
            "Instance {} ({} jobs, {} machines):",
            idx + 1,
            instance.num_jobs,
            instance.num_machines
        );

        match solver.solve(&instance) {
            Ok(schedule) => {
                println!("  Generated Makespan: {}", schedule.makespan);

                // Compare with NN solver
                match JSSPSolver::solve_nearest_neighbor(&instance) {
                    Ok(nn_schedule) => {
                        let gap = ((schedule.makespan as f32 - nn_schedule.makespan as f32)
                            / nn_schedule.makespan as f32)
                            * 100.0;
                        println!("  NN Solver Makespan: {}", nn_schedule.makespan);
                        println!("  Gap to NN: {:.2}%\n", gap);
                    }
                    Err(_) => {}
                }
            }
            Err(e) => {
                eprintln!("  Error: {}\n", e);
            }
        }
    }

    Ok(())
}

/// Run example with filtered Starjob data (specific sizes)
fn example_with_filtered_starjob(json_path: &str) -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   JSSP LoRA - Filtered Starjob Dataset (5×3-10×5)    ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Load only instances with 5-10 jobs and 3-5 machines
    let filtered = load_starjob_by_size(json_path, 5, 10, 3, 5, Some(50))?;
    println!("Filtered to {} instances\n", filtered.instances.len());

    // Show some examples
    println!("Sample instances:");
    for (idx, instance) in filtered.instances.iter().take(5).enumerate() {
        println!(
            "  {}: {} jobs × {} machines",
            idx + 1,
            instance.num_jobs,
            instance.num_machines
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║      JSSP LoRA Solver - Multiple Examples            ║");
    println!("╚════════════════════════════════════════════════════════╝");

    // Example 1: Run with built-in sample data
    println!("\n\n>>> EXAMPLE 1: Sample Data <<<");
    example_with_sample_data()?;

    // Example 2: Run with Starjob dataset (if available)
    // Uncomment and set the correct path to starjob130k.json
    println!("\n\n>>> EXAMPLE 2: Starjob Dataset <<<");
    match example_with_starjob_data("data/starjob130k.json", Some(200)) {
        Ok(_) => {
            println!("✓ Starjob example completed successfully");
        }
        Err(e) => {
            eprintln!("⚠ Starjob example failed (file not found?)");
            eprintln!("  To use Starjob data:");
            eprintln!("  1. Download from: https://huggingface.co/datasets/henri24/Starjob");
            eprintln!("  2. Save as: data/starjob130k.json");
            eprintln!("  3. Uncomment in main.rs");
            eprintln!("  Error: {}", e);
        }
    }

    // Example 3: Filtered Starjob (if available)
    println!("\n\n>>> EXAMPLE 3: Filtered Starjob Dataset <<<");
    match example_with_filtered_starjob("data/starjob130k.json") {
        Ok(_) => {
            println!("✓ Filtered Starjob example completed");
        }
        Err(_) => {
            eprintln!("⚠ Filtered Starjob example skipped (file not found)");
        }
    }

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║      All Examples Completed                          ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    Ok(())
}
