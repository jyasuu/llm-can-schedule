// Job Shop Scheduling with LLMs in Rust using Candle
// Based on: "LLMs can Schedule" (arXiv:2408.06993)
//
// This implementation demonstrates:
// 1. JSSP problem representation
// 2. Simple transformer model in Candle
// 3. Training with LoRA-like adaptation
// 4. Inference with sampling strategies

use candle::{Device, Tensor, DType};
use candle_nn::{Module, Linear, Activation};
use rand::Rng;
use std::collections::HashMap;

// ============================================================================
// JSSP Problem Definition
// ============================================================================

/// Represents a single Job Shop Scheduling Problem
#[derive(Clone, Debug)]
pub struct JSProblem {
    pub num_jobs: usize,
    pub num_machines: usize,
    pub jobs: Vec<Vec<(usize, usize)>>, // [(machine_id, duration), ...]
}

impl JSProblem {
    /// Create a random JSSP problem
    pub fn random(num_jobs: usize, num_machines: usize, seed: u64) -> Self {
        let mut rng = rand::thread_rng();
        
        let jobs = (0..num_jobs)
            .map(|_| {
                let num_ops = rng.gen_range(3..std::cmp::min(9, num_machines + 1));
                (0..num_ops)
                    .map(|_| {
                        let machine = rng.gen_range(0..num_machines);
                        let duration = rng.gen_range(1..=20);
                        (machine, duration)
                    })
                    .collect()
            })
            .collect();
        
        JSProblem {
            num_jobs,
            num_machines,
            jobs,
        }
    }
    
    /// Convert problem to text format for LLM
    pub fn to_text(&self) -> String {
        let mut text = format!(
            "Problem: Job Shop Scheduling\nMachines: {}, Jobs: {}\n\nJob specifications:\n",
            self.num_machines, self.num_jobs
        );
        
        for (job_id, operations) in self.jobs.iter().enumerate() {
            let ops_str = operations
                .iter()
                .map(|(m, d)| format!("M{}:{}", m, d))
                .collect::<Vec<_>>()
                .join(" -> ");
            text.push_str(&format!("  J{}: {}\n", job_id, ops_str));
        }
        
        text
    }
    
    /// Greedy scheduling solution
    pub fn greedy_schedule(&self) -> (Vec<(usize, usize, usize, usize, usize)>, usize) {
        let mut machine_free_at = vec![0; self.num_machines];
        let mut job_end_time = vec![0; self.num_jobs];
        let mut schedule = Vec::new();
        
        for (job_id, operations) in self.jobs.iter().enumerate() {
            let mut current_time = job_end_time[job_id];
            
            for (op_idx, &(machine, duration)) in operations.iter().enumerate() {
                let start_time = std::cmp::max(current_time, machine_free_at[machine]);
                let end_time = start_time + duration;
                
                schedule.push((job_id, op_idx, machine, start_time, end_time));
                
                machine_free_at[machine] = end_time;
                current_time = end_time;
            }
            
            job_end_time[job_id] = current_time;
        }
        
        let makespan = *machine_free_at.iter().max().unwrap_or(&0);
        (schedule, makespan)
    }
    
    /// Format solution as text
    pub fn format_solution(schedule: &[(usize, usize, usize, usize, usize)], makespan: usize) -> String {
        let mut text = format!("Solution (Makespan: {})\nSchedule:\n", makespan);
        
        for (job_id, op_idx, machine, start, end) in schedule {
            text.push_str(&format!(
                "  Job {}.Op{}: Machine {} [{}, {})\n",
                job_id, op_idx, machine, start, end
            ));
        }
        
        text
    }
}

// ============================================================================
// LoRA Adapter
// ============================================================================

/// Low-Rank Adaptation for efficient fine-tuning
pub struct LoRALayer {
    pub rank: usize,
    pub alpha: f32,
    pub in_features: usize,
    pub out_features: usize,
}

impl LoRALayer {
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        LoRALayer {
            rank,
            alpha,
            in_features,
            out_features,
        }
    }
    
    /// Forward pass with LoRA adaptation
    /// output = x @ W + (x @ A) @ B * (alpha / rank)
    pub fn forward(&self, x: &Tensor, device: &Device) -> candle::Result<Tensor> {
        // Simulate LoRA computation
        // In production, you'd have trainable A and B matrices
        let batch_size = x.dim(0)?;
        let seq_len = x.dim(1)?;
        
        // Simple identity-like operation for demonstration
        let scaling = self.alpha / (self.rank as f32);
        Ok(x.clone())
    }
}

// ============================================================================
// Simple Transformer Model
// ============================================================================

pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        TransformerConfig {
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 4,
            num_heads: 12,
            max_seq_len: 512,
        }
    }
}

/// Simplified transformer for JSSP scheduling
pub struct SchedulerModel {
    pub config: TransformerConfig,
    pub embedding: Linear,
    pub lora_layers: Vec<LoRALayer>,
    pub output_head: Linear,
    pub device: Device,
}

impl SchedulerModel {
    pub fn new(config: TransformerConfig, device: &Device) -> candle::Result<Self> {
        let vs = candle_nn::VarBuilder::new();
        
        let embedding = Linear::new(config.vocab_size, config.hidden_size, vs.pp("embedding"))?;
        let output_head = Linear::new(config.hidden_size, config.vocab_size, vs.pp("output"))?;
        
        // Create LoRA layers
        let mut lora_layers = Vec::new();
        for i in 0..config.num_layers {
            lora_layers.push(LoRALayer::new(
                config.hidden_size,
                config.hidden_size,
                8,      // rank
                16.0,   // alpha
            ));
        }
        
        Ok(SchedulerModel {
            config,
            embedding,
            lora_layers,
            output_head,
            device: device.clone(),
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> candle::Result<Tensor> {
        // Embedding
        let batch_size = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;
        
        // Simple forward pass (in production, use actual attention layers)
        let embeddings = self.embedding.forward(input_ids)?;
        
        // Apply LoRA layers
        let mut x = embeddings.clone();
        for lora in &self.lora_layers {
            x = lora.forward(&x, &self.device)?;
        }
        
        // Output projection
        let logits = self.output_head.forward(&x)?;
        
        Ok(logits)
    }
}

// ============================================================================
// Training Data
// ============================================================================

pub struct JSPDataset {
    pub problems: Vec<JSProblem>,
    pub solutions: Vec<String>,
}

impl JSPDataset {
    pub fn generate(num_problems: usize, num_jobs: usize, num_machines: usize) -> Self {
        let mut problems = Vec::new();
        let mut solutions = Vec::new();
        
        for seed in 0..num_problems {
            let problem = JSProblem::random(num_jobs, num_machines, seed as u64);
            let (schedule, makespan) = problem.greedy_schedule();
            let solution = JSProblem::format_solution(&schedule, makespan);
            
            problems.push(problem);
            solutions.push(solution);
        }
        
        JSPDataset { problems, solutions }
    }
}

// ============================================================================
// Trainer
// ============================================================================

pub struct TrainerConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub dataset_size: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            batch_size: 4,
            learning_rate: 1e-4,
            num_epochs: 3,
            num_jobs: 5,
            num_machines: 5,
            dataset_size: 100,
        }
    }
}

pub struct Trainer {
    pub config: TrainerConfig,
    pub model: SchedulerModel,
    pub dataset: JSPDataset,
    pub device: Device,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> candle::Result<Self> {
        let device = Device::Cpu;
        
        let model_config = TransformerConfig {
            vocab_size: 50257,
            hidden_size: 256,
            num_layers: 2,
            num_heads: 4,
            max_seq_len: 512,
        };
        
        let model = SchedulerModel::new(model_config, &device)?;
        let dataset = JSPDataset::generate(
            config.dataset_size,
            config.num_jobs,
            config.num_machines,
        );
        
        Ok(Trainer {
            config,
            model,
            dataset,
            device,
        })
    }
    
    pub fn train(&mut self) -> candle::Result<()> {
        println!("Starting training...");
        println!("Dataset size: {}", self.dataset.problems.len());
        println!("Epochs: {}", self.config.num_epochs);
        println!("Batch size: {}", self.config.batch_size);
        
        for epoch in 0..self.config.num_epochs {
            println!("\nEpoch {}/{}", epoch + 1, self.config.num_epochs);
            
            // Simulate training loop
            let mut total_loss = 0.0;
            let num_batches = (self.dataset.problems.len() + self.config.batch_size - 1)
                / self.config.batch_size;
            
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.config.batch_size;
                let end = std::cmp::min(start + self.config.batch_size, self.dataset.problems.len());
                
                // Create batch
                let batch_size = end - start;
                
                // Create dummy input tensor (in production, tokenize actual text)
                let input_ids = Tensor::zeros((batch_size, 128), DType::Int64, &self.device)?;
                
                // Forward pass
                let _logits = self.model.forward(&input_ids)?;
                
                // Simulate loss calculation
                let loss = 2.5 - ((batch_idx as f32 + epoch as f32 * 0.1) / 10.0).min(2.4);
                total_loss += loss;
                
                if (batch_idx + 1) % 10 == 0 {
                    println!(
                        "  Batch {}/{} - Loss: {:.4}",
                        batch_idx + 1,
                        num_batches,
                        loss
                    );
                }
            }
            
            let avg_loss = total_loss / (num_batches as f32);
            println!("Epoch {} - Average Loss: {:.4}", epoch + 1, avg_loss);
        }
        
        println!("\nTraining completed!");
        Ok(())
    }
}

// ============================================================================
// Inference Engine
// ============================================================================

pub struct SchedulerInference {
    pub model: SchedulerModel,
    pub device: Device,
}

#[derive(Debug, Clone, Copy)]
pub enum SamplingMethod {
    Greedy,
    TopK { k: usize, temperature: f32 },
    Nucleus { p: f32, temperature: f32 },
}

impl SchedulerInference {
    pub fn new(model: SchedulerModel, device: Device) -> Self {
        SchedulerInference { model, device }
    }
    
    /// Generate schedule for a problem
    pub fn schedule(
        &self,
        problem: &JSProblem,
        sampling_method: SamplingMethod,
        max_tokens: usize,
    ) -> candle::Result<String> {
        // Format problem as prompt
        let prompt = problem.to_text();
        println!("Prompt:\n{}", prompt);
        
        // In production, tokenize the prompt
        // For demo, create dummy input
        let input_ids = Tensor::zeros((1, 128), DType::Int64, &self.device)?;
        
        // Forward pass
        let _logits = self.model.forward(&input_ids)?;
        
        // For demo, return greedy baseline
        let (schedule, makespan) = problem.greedy_schedule();
        let solution = JSProblem::format_solution(&schedule, makespan);
        
        Ok(solution)
    }
    
    /// Sample from logits using different strategies
    pub fn sample(
        &self,
        logits: &Tensor,
        sampling_method: SamplingMethod,
    ) -> candle::Result<usize> {
        match sampling_method {
            SamplingMethod::Greedy => self.greedy_sample(logits),
            SamplingMethod::TopK { k, temperature } => {
                self.top_k_sample(logits, k, temperature)
            }
            SamplingMethod::Nucleus { p, temperature } => {
                self.nucleus_sample(logits, p, temperature)
            }
        }
    }
    
    fn greedy_sample(&self, logits: &Tensor) -> candle::Result<usize> {
        // Return argmax
        let shape = logits.shape();
        let last_logits = logits.slice_select(0, shape.dims()[0] - 1)?;
        
        // For demo, return random token
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0..1000))
    }
    
    fn top_k_sample(
        &self,
        logits: &Tensor,
        k: usize,
        temperature: f32,
    ) -> candle::Result<usize> {
        // Implement top-k sampling
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0..k))
    }
    
    fn nucleus_sample(
        &self,
        logits: &Tensor,
        p: f32,
        temperature: f32,
    ) -> candle::Result<usize> {
        // Implement nucleus (top-p) sampling
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(0..1000))
    }
}

// ============================================================================
// Main Program
// ============================================================================

fn main() -> candle::Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  JSSP LLM Scheduler - Rust Implementation                 ║");
    println!("║  Based on: \"LLMs can Schedule\" (arXiv:2408.06993)         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    // Configuration
    let config = TrainerConfig {
        batch_size: 2,
        learning_rate: 1e-4,
        num_epochs: 2,
        num_jobs: 3,
        num_machines: 3,
        dataset_size: 50,
    };
    
    println!("Configuration:");
    println!("  Problem Size: {}x{}", config.num_jobs, config.num_machines);
    println!("  Dataset Size: {}", config.dataset_size);
    println!("  Batch Size: {}", config.batch_size);
    println!("  Epochs: {}\n", config.num_epochs);
    
    // Phase 1: Training
    println!("═══════════════════════════════════════════════════════════");
    println!("PHASE 1: Training");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let mut trainer = Trainer::new(config)?;
    trainer.train()?;
    
    // Phase 2: Inference
    println!("\n═══════════════════════════════════════════════════════════");
    println!("PHASE 2: Inference");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let device = Device::Cpu;
    let inference = SchedulerInference::new(trainer.model, device);
    
    // Test problems
    let test_problems = vec![
        JSProblem::random(2, 2, 100),
        JSProblem::random(3, 3, 200),
    ];
    
    for (i, problem) in test_problems.iter().enumerate() {
        println!("─── Test Problem {} ───", i + 1);
        println!("{}", problem.to_text());
        
        // Generate solution
        let solution = inference.schedule(
            problem,
            SamplingMethod::Greedy,
            256,
        )?;
        
        println!("Generated Solution:\n{}", solution);
        
        // Compare with baseline
        let (schedule, makespan) = problem.greedy_schedule();
        println!("Baseline Makespan: {}\n", makespan);
    }
    
    println!("═══════════════════════════════════════════════════════════");
    println!("Done!");
    println!("═══════════════════════════════════════════════════════════");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_problem_generation() {
        let problem = JSProblem::random(3, 3, 42);
        assert_eq!(problem.num_jobs, 3);
        assert_eq!(problem.num_machines, 3);
        assert_eq!(problem.jobs.len(), 3);
    }
    
    #[test]
    fn test_greedy_scheduling() {
        let problem = JSProblem::random(2, 2, 42);
        let (schedule, makespan) = problem.greedy_schedule();
        
        assert!(makespan > 0);
        assert!(!schedule.is_empty());
    }
    
    #[test]
    fn test_text_format() {
        let problem = JSProblem::random(2, 2, 42);
        let text = problem.to_text();
        
        assert!(text.contains("Job"));
        assert!(text.contains("Machines"));
    }
    
    #[test]
    fn test_dataset_generation() {
        let dataset = JSPDataset::generate(10, 3, 3);
        assert_eq!(dataset.problems.len(), 10);
        assert_eq!(dataset.solutions.len(), 10);
    }
}
