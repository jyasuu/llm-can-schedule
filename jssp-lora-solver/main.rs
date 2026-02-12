use anyhow::{Ok as AnyOk, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod lora;
mod jssp;
mod data;

use lora::{LoRA, LoRAConfig};
use jssp::{JSSPInstance, JSSPSolver, Schedule};
use data::{JSSPDataset, load_sample_data};

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

    fn train_on_instance(&mut self, instance: &JSSPInstance, optimal_schedule: &Schedule) -> Result<f32> {
        // Encode the JSSP instance
        let input = instance.encode()?;
        let input_tensor = Tensor::new(&input, &self.device)?;

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
        let input_tensor = Tensor::new(&input, &self.device)?;

        let logits = self.model.forward(&input_tensor)?;
        let schedule = Schedule::from_tensor(&logits)?;

        Ok(schedule)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== JSSP LoRA Solver with Candle ===\n");

    // Use CPU for demonstration (change to CUDA if available)
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    let config = TrainingConfig::default();
    println!("Configuration: {:?}\n", config);

    // Load sample JSSP instances
    let dataset = load_sample_data()?;
    println!("Loaded {} JSSP instances\n", dataset.instances.len());

    // Initialize the LoRA solver
    let mut solver = JSSPLoRASolver::new(device.clone(), config.clone())?;
    println!("Initialized LoRA solver with rank={}, alpha={}\n", 
             config.lora_rank, config.lora_alpha);

    // Training loop
    println!("Starting training...\n");
    for epoch in 0..config.epochs {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (idx, instance) in dataset.instances.iter().enumerate() {
            // Get optimal schedule (from dataset)
            let optimal_schedule = &dataset.optimal_schedules[idx];

            // Train on this instance
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

        let avg_loss = if count > 0 { total_loss / count as f32 } else { 0.0 };
        println!("Epoch {}: Average Loss = {:.6}\n", epoch + 1, avg_loss);
    }

    // Test the trained model
    println!("\n=== Testing Trained Model ===\n");
    
    if let Some(test_instance) = dataset.instances.first() {
        println!("Test Instance:");
        println!("  Jobs: {}", test_instance.num_jobs);
        println!("  Machines: {}", test_instance.num_machines);
        println!("  Processing times: {:?}\n", 
                 &test_instance.processing_times[0..std::cmp::min(5, test_instance.processing_times.len())]);

        match solver.solve(&test_instance) {
            Ok(schedule) => {
                println!("Generated Schedule:");
                println!("  Makespan: {}", schedule.makespan);
                println!("  Job sequence: {:?}\n", schedule.job_sequence);

                // Compare with optimal
                if let Some(optimal) = dataset.optimal_schedules.first() {
                    println!("Optimal Schedule (reference):");
                    println!("  Makespan: {}", optimal.makespan);
                    println!("  Job sequence: {:?}\n", optimal.job_sequence);

                    let efficiency = (optimal.makespan as f32) / (schedule.makespan as f32);
                    println!("Solution Quality: {:.2}% of optimal", efficiency * 100.0);
                }
            }
            Err(e) => {
                eprintln!("Error solving instance: {}", e);
            }
        }
    }

    println!("\n=== Solving Multiple Examples ===\n");
    
    // Solve multiple instances
    for (idx, instance) in dataset.instances.iter().take(3).enumerate() {
        println!("Example {} ({} jobs, {} machines):", 
                 idx + 1, instance.num_jobs, instance.num_machines);
        
        match solver.solve(&instance) {
            Ok(schedule) => {
                println!("  Generated Makespan: {}", schedule.makespan);
                if let Some(optimal) = dataset.optimal_schedules.get(idx) {
                    println!("  Optimal Makespan: {}", optimal.makespan);
                    let gap = ((schedule.makespan - optimal.makespan) as f32 / optimal.makespan as f32) * 100.0;
                    println!("  Gap to Optimal: {:.2}%\n", gap);
                }
            }
            Err(e) => {
                eprintln!("  Error: {}\n", e);
            }
        }
    }

    println!("=== Training Complete ===");
    Ok(())
}
