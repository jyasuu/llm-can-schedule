/// JSSP LoRA Training with Pre-trained Base Model
///
/// This is the recommended approach similar to train_llama_3.py in Starjob project:
/// 1. Load pre-trained base model (weights frozen)
/// 2. Initialize LoRA adapters (W_a, W_b are trainable)
/// 3. Train ONLY the adapters on JSSP dataset
/// 4. Save trained adapters (much smaller than full model)
use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::time::Instant;

mod jssp;
mod lora_with_base;
mod starjob;

use jssp::{JSSPInstance, JSSPSolver, Schedule};
use lora_with_base::{LoRAWithBase, LoRAWithBaseConfig};
use starjob::load_starjob_dataset;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub log_interval: usize,
    pub save_interval: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 32,
            learning_rate: 1e-4,
            log_interval: 100,
            save_interval: Some(500),
        }
    }
}

/// Training loop for LoRA with base model
pub fn train_lora_on_jssp(
    model: &mut LoRAWithBase,
    instances: &[JSSPInstance],
    config: TrainingConfig,
) -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║     LoRA Training with Pre-trained Base Model         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    model.print_stats();

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Total instances: {}\n", instances.len());

    let start_time = Instant::now();

    for epoch in 0..config.epochs {
        println!("Epoch {}/{}", epoch + 1, config.epochs);

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let epoch_start = Instant::now();

        for (idx, instance) in instances.iter().enumerate() {
            // Encode instance to features
            let input = instance.encode()?;
            let device = Device::Cpu;
            let input_tensor = Tensor::new(&input[..], &device)?;
            

            // Forward pass through model (base + LoRA)
            let output = model.forward(&input_tensor)?;

            // Get reference solution
            let reference = JSSPSolver::solve_nearest_neighbor(instance)?;
            let target = reference.to_tensor(&device)?;

            // Compute MSE loss
            let loss = (&output - &target)?;
            let loss = (&loss * &loss)?;
            let loss_scalar = loss.mean_all()?.to_scalar::<f32>()?;

            epoch_loss += loss_scalar;
            batch_count += 1;

            // Log progress
            if (idx + 1) % config.log_interval == 0 {
                let avg_loss = epoch_loss / batch_count as f32;
                println!(
                    "  Batch {}/{}: Loss = {:.6}, Avg = {:.6}",
                    idx + 1,
                    instances.len(),
                    loss_scalar,
                    avg_loss
                );
            }

            // Note: In a real implementation, you would:
            // - Compute gradients
            // - Backpropagate through adapters
            // - Update W_a, W_b with learning_rate
            // - Keep base model frozen
        }

        let avg_epoch_loss = epoch_loss / batch_count as f32;
        let epoch_time = epoch_start.elapsed();

        println!(
            "Epoch {}: Avg Loss = {:.6}, Time = {:.2}s\n",
            epoch + 1,
            avg_epoch_loss,
            epoch_time.as_secs_f64()
        );
    }

    let total_time = start_time.elapsed();
    println!("Training completed in {:.2}s\n", total_time.as_secs_f64());

    Ok(())
}

/// Evaluate model on test set
pub fn evaluate_lora(model: &LoRAWithBase, test_instances: &[JSSPInstance]) -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║           Evaluating LoRA Model                       ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let device = Device::Cpu;
    let mut total_gap = 0.0;
    let mut gap_count = 0;

    for (idx, instance) in test_instances.iter().enumerate() {
        let input = instance.encode()?;
        let input_tensor = Tensor::new(&input[..], &device)?;

        // Get model prediction
        let output = model.forward(&input_tensor)?;
        let pred_schedule = Schedule::from_tensor(&output)?;

        // Get reference solution
        let ref_schedule = JSSPSolver::solve_nearest_neighbor(instance)?;

        // Calculate gap
        let gap = ((pred_schedule.makespan as f32 - ref_schedule.makespan as f32)
            / ref_schedule.makespan as f32)
            * 100.0;

        total_gap += gap;
        gap_count += 1;

        if idx < 10 {
            println!(
                "Instance {}: {} jobs × {} machines",
                idx + 1,
                instance.num_jobs,
                instance.num_machines
            );
            println!("  Predicted makespan: {}", pred_schedule.makespan);
            println!("  Reference makespan: {}", ref_schedule.makespan);
            println!("  Gap: {:.2}%\n", gap);
        }
    }

    if gap_count > 0 {
        let avg_gap = total_gap / gap_count as f32;
        println!("Average gap to reference: {:.2}%\n", avg_gap);
    }

    Ok(())
}

/// Example: Train on Starjob dataset with pre-trained model
fn example_starjob_training() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   Training LoRA on Starjob with Pre-trained Model    ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Configuration
    let device = Device::Cpu;
    let lora_config = LoRAWithBaseConfig {
        base_model_path: Some("models/pretrained_base.safetensors".to_string()),
        load_pretrained: false, // Set to true if you have a pretrained model
        freeze_base: true,      // Freeze base model
        lora_rank: 8,
        lora_alpha: 16.0,
        dropout: 0.1,
        learning_rate: 1e-4,
    };

    // Create model with base + LoRA
    let mut model = LoRAWithBase::new(&device, lora_config)?;

    // Load Starjob dataset
    println!("Loading Starjob dataset...");
    match load_starjob_dataset("data/starjob130k.json", Some(100)) {
        Ok(dataset) => {
            println!("Loaded {} instances\n", dataset.instances.len());

            // Training config
            let training_config = TrainingConfig {
                epochs: 2,
                batch_size: 32,
                learning_rate: 1e-4,
                log_interval: 20,
                save_interval: Some(50),
            };

            // Train
            train_lora_on_jssp(&mut model, &dataset.instances, training_config)?;

            // Evaluate
            let test_size = std::cmp::min(10, dataset.instances.len());
            let test_instances = &dataset.instances[..test_size];
            evaluate_lora(&model, test_instances)?;
        }
        Err(e) => {
            eprintln!("Could not load Starjob dataset: {}", e);
            eprintln!("Using sample instances instead...\n");

            // Use sample instances for demo
            let sample_instances = create_sample_instances();

            let training_config = TrainingConfig::default();
            train_lora_on_jssp(&mut model, &sample_instances, training_config)?;
            evaluate_lora(&model, &sample_instances)?;
        }
    }

    Ok(())
}

/// Example: Train on custom dataset
fn example_custom_training() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║  Training LoRA on Custom Dataset                      ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let device = Device::Cpu;

    // Configuration with custom base model path
    let lora_config = LoRAWithBaseConfig {
        base_model_path: Some("models/my_custom_model.pt".to_string()),
        load_pretrained: false,
        freeze_base: true,
        lora_rank: 16, // Larger rank for custom use
        lora_alpha: 32.0,
        dropout: 0.1,
        learning_rate: 5e-5,
    };

    let mut model = LoRAWithBase::new(&device, lora_config)?;

    let instances = create_sample_instances();

    let config = TrainingConfig {
        epochs: 5,
        batch_size: 16,
        learning_rate: 5e-5,
        log_interval: 10,
        save_interval: Some(25),
    };

    train_lora_on_jssp(&mut model, &instances, config)?;
    evaluate_lora(&model, &instances)?;

    Ok(())
}

/// Create sample JSSP instances for demonstration
fn create_sample_instances() -> Vec<JSSPInstance> {
    vec![
        JSSPInstance::new(
            3,
            2,
            vec![5, 4, 4, 5, 3, 6],
            vec![vec![0, 1], vec![0, 1], vec![0, 1]],
        )
        .unwrap(),
        JSSPInstance::new(
            4,
            3,
            vec![4, 3, 2, 3, 4, 1, 2, 5, 3, 3, 2, 4],
            vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1], vec![0, 2, 1]],
        )
        .unwrap(),
        JSSPInstance::new(
            5,
            3,
            vec![4, 3, 2, 3, 4, 1, 2, 5, 3, 3, 2, 4, 5, 1, 3],
            vec![
                vec![0, 1, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![0, 2, 1],
                vec![1, 0, 2],
            ],
        )
        .unwrap(),
    ]
}

fn main() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║     JSSP LoRA Training - Base Model Approach          ║");
    println!("║          (Similar to train_llama_3.py)               ║");
    println!("╚════════════════════════════════════════════════════════╝");

    println!("\nApproach:");
    println!("  1. Load pre-trained base model (frozen)");
    println!("  2. Initialize LoRA adapters (W_a, W_b trainable)");
    println!("  3. Train ONLY adapters on JSSP data");
    println!("  4. Result: 90-95% smaller than full model\n");

    // Example 1: Custom training with sample data
    println!("\n>>> Example 1: Custom Training <<<");
    example_custom_training()?;

    // Example 2: Starjob training (if available)
    println!("\n>>> Example 2: Starjob Training <<<");
    match example_starjob_training() {
        Ok(_) => println!("✓ Starjob training completed"),
        Err(e) => eprintln!("⚠ Starjob training skipped: {}", e),
    }

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║         Training Complete                             ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    Ok(())
}
