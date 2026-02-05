# JSSP LLM Scheduler in Rust

Complete guide for implementing Job Shop Scheduling with LLMs using Rust and the Candle framework.

## 📋 Overview

This Rust implementation provides:
- **High Performance**: Compiled code, no GC overhead
- **Memory Safe**: Leverages Rust's ownership system
- **Deployable**: Single binary, easy to containerize
- **Concurrent**: Built for parallel processing
- **Production Ready**: Error handling, logging, testing

## 🚀 Getting Started

### Installation

#### Prerequisites
- Rust 1.70+ (install from https://rustup.rs/)
- Cargo (comes with Rust)

#### Setup
```bash
# Create new project
cargo new jssp_scheduler --bin
cd jssp_scheduler

# Copy the provided files
# - jssp_scheduler.rs → src/main.rs
# - Cargo.toml → Cargo.toml
```

### Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release
```

### Running

```bash
# Debug build
cargo run

# Release build (faster)
cargo run --release

# With logging
RUST_LOG=debug cargo run

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## 📂 Project Structure

```
jssp_scheduler/
├── Cargo.toml              # Project manifest
├── src/
│   ├── main.rs            # Main program
│   ├── problem.rs         # JSSP problem (optional module)
│   ├── model.rs           # Scheduler model (optional module)
│   ├── trainer.rs         # Training logic (optional module)
│   └── inference.rs       # Inference engine (optional module)
├── benches/
│   └── scheduling_benchmark.rs
├── tests/
│   └── integration_tests.rs
└── README.md
```

## 🏗️ Architecture

### Main Components

#### 1. **JSProblem**
Represents a single JSSP instance
```rust
pub struct JSProblem {
    pub num_jobs: usize,
    pub num_machines: usize,
    pub jobs: Vec<Vec<(usize, usize)>>, // [(machine, duration), ...]
}

// Methods:
// - random(num_jobs, num_machines, seed) → JSProblem
// - to_text() → String
// - greedy_schedule() → (schedule, makespan)
```

#### 2. **LoRALayer**
Efficient fine-tuning adapter
```rust
pub struct LoRALayer {
    pub rank: usize,
    pub alpha: f32,
    pub in_features: usize,
    pub out_features: usize,
}

// Low-rank adaptation: output = x @ W + (x @ A) @ B * (alpha/rank)
```

#### 3. **SchedulerModel**
Transformer-based scheduler
```rust
pub struct SchedulerModel {
    pub embedding: Linear,      // Token embedding
    pub lora_layers: Vec<LoRALayer>, // Adaptable layers
    pub output_head: Linear,    // Output projection
}

// Methods:
// - new(config, device) → Result<SchedulerModel>
// - forward(input_ids) → Result<Tensor>
```

#### 4. **Trainer**
Training loop
```rust
pub struct Trainer {
    pub model: SchedulerModel,
    pub dataset: JSPDataset,
    pub config: TrainerConfig,
}

// Methods:
// - new(config) → Result<Trainer>
// - train() → Result<()>
```

#### 5. **SchedulerInference**
Inference engine with multiple sampling strategies
```rust
pub struct SchedulerInference {
    pub model: SchedulerModel,
    pub device: Device,
}

pub enum SamplingMethod {
    Greedy,
    TopK { k: usize, temperature: f32 },
    Nucleus { p: f32, temperature: f32 },
}

// Methods:
// - schedule(problem, method, max_tokens) → Result<String>
// - sample(logits, method) → Result<usize>
```

## ⚙️ Configuration

### TrainerConfig
```rust
pub struct TrainerConfig {
    pub batch_size: usize,          // Training batch size
    pub learning_rate: f32,         // LoRA learning rate
    pub num_epochs: usize,          // Training epochs
    pub num_jobs: usize,            // JSSP problem size
    pub num_machines: usize,        // JSSP problem size
    pub dataset_size: usize,        // Number of training samples
}

// Default:
let config = TrainerConfig::default();
// batch_size: 4, learning_rate: 1e-4, num_epochs: 3, etc.
```

### TransformerConfig
```rust
pub struct TransformerConfig {
    pub vocab_size: usize,          // Token vocabulary
    pub hidden_size: usize,         // Model dimension
    pub num_layers: usize,          // Number of transformer layers
    pub num_heads: usize,           // Number of attention heads
    pub max_seq_len: usize,         // Maximum sequence length
}

// Default: GPT-2 like configuration
```

## 🎯 Usage Examples

### Basic Training and Inference

```rust
use jssp_scheduler::{Trainer, TrainerConfig, SchedulerInference, JSProblem};

fn main() -> Result<()> {
    // Configure
    let config = TrainerConfig {
        batch_size: 4,
        learning_rate: 1e-4,
        num_epochs: 3,
        num_jobs: 5,
        num_machines: 5,
        dataset_size: 500,
    };
    
    // Train
    let mut trainer = Trainer::new(config)?;
    trainer.train()?;
    
    // Inference
    let device = Device::Cpu;
    let inference = SchedulerInference::new(trainer.model, device);
    
    // Schedule a problem
    let problem = JSProblem::random(3, 3, 42);
    let solution = inference.schedule(
        &problem,
        SamplingMethod::Greedy,
        256,
    )?;
    
    println!("{}", solution);
    Ok(())
}
```

### Custom JSSP Problem

```rust
let mut problem = JSProblem {
    num_jobs: 2,
    num_machines: 2,
    jobs: vec![
        vec![(0, 5), (1, 3)],      // Job 0: 5h on M0, 3h on M1
        vec![(1, 4), (0, 2)],      // Job 1: 4h on M1, 2h on M0
    ]
};

println!("{}", problem.to_text());
let (schedule, makespan) = problem.greedy_schedule();
println!("Makespan: {}", makespan);
```

### Multiple Sampling Strategies

```rust
let methods = vec![
    SamplingMethod::Greedy,
    SamplingMethod::TopK { k: 50, temperature: 0.8 },
    SamplingMethod::Nucleus { p: 0.95, temperature: 0.8 },
];

for method in methods {
    let solution = inference.schedule(&problem, method, 256)?;
    println!("{:?}: {}", method, evaluate_solution(&solution));
}
```

### Batch Processing

```rust
let problems: Vec<JSProblem> = (0..100)
    .map(|i| JSProblem::random(5, 5, i))
    .collect();

let solutions: Vec<String> = problems
    .par_iter()  // Parallel iteration with Rayon
    .map(|p| inference.schedule(p, SamplingMethod::Greedy, 256).unwrap())
    .collect();
```

## 📊 Dataset Handling

### Generate Synthetic Data

```rust
use jssp_scheduler::JSPDataset;

let dataset = JSPDataset::generate(
    1000,    // num_problems
    5,       // num_jobs
    5,       // num_machines
);

println!("Generated {} problems", dataset.problems.len());
for (problem, solution) in dataset.problems.iter().zip(&dataset.solutions) {
    println!("{}", problem.to_text());
    println!("{}", solution);
}
```

### Load from CSV

```rust
use std::fs::File;
use csv::Reader;

fn load_jssp_from_csv(path: &str) -> Result<Vec<JSProblem>> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);
    
    let mut problems = Vec::new();
    for result in reader.deserialize() {
        let record: JSProblemRecord = result?;
        problems.push(record.to_jssp());
    }
    
    Ok(problems)
}
```

### Save Results to JSON

```rust
use serde_json::json;
use std::fs;

fn save_results(
    problems: &[JSProblem],
    solutions: &[String],
    metrics: &[SolutionMetrics],
) -> Result<()> {
    let results = json!({
        "problems": problems.len(),
        "avg_makespan": calculate_avg_makespan(metrics),
        "solutions": solutions,
    });
    
    fs::write("results.json", results.to_string())?;
    Ok(())
}
```

## 🔧 Advanced Usage

### Custom Model Architecture

```rust
pub struct CustomSchedulerModel {
    pub embedding: Linear,
    pub attention_layers: Vec<SelfAttention>,
    pub ffn_layers: Vec<FeedForwardNetwork>,
    pub lora_adapters: Vec<LoRALayer>,
    pub output_head: Linear,
}

impl CustomSchedulerModel {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Custom implementation
        Ok(Tensor::zeros((1, 1), DType::Float32, &Device::Cpu)?)
    }
}
```

### Multi-GPU Training

```rust
// With Candle, use Device::Cuda for GPU
let device = Device::new_cuda(0)?;  // GPU 0
let model = SchedulerModel::new(config, &device)?;

// Or use multiple GPUs with data parallelism
let devices = vec![
    Device::new_cuda(0)?,
    Device::new_cuda(1)?,
];
```

### Model Serialization

```rust
use std::fs;

// Save model weights
trainer.model.save_weights("model.safetensors")?;

// Load model weights
let model = SchedulerModel::new(config, &device)?;
model.load_weights("model.safetensors")?;
```

## 🧪 Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_problem_generation() {
        let problem = JSProblem::random(3, 3, 42);
        assert_eq!(problem.num_jobs, 3);
        assert_eq!(problem.num_machines, 3);
    }
    
    #[test]
    fn test_greedy_scheduling() {
        let problem = JSProblem::random(5, 5, 42);
        let (schedule, makespan) = problem.greedy_schedule();
        
        assert!(!schedule.is_empty());
        assert!(makespan > 0);
    }
    
    #[test]
    fn test_model_forward() -> Result<()> {
        let device = Device::Cpu;
        let config = TransformerConfig::default();
        let model = SchedulerModel::new(config, &device)?;
        
        let input = Tensor::zeros((1, 128), DType::Int64, &device)?;
        let _output = model.forward(&input)?;
        
        Ok(())
    }
}
```

Run tests:
```bash
cargo test

# Run specific test
cargo test test_problem_generation

# Run with output
cargo test -- --nocapture

# Run ignored tests
cargo test -- --ignored
```

### Benchmarking

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn scheduling_benchmark(c: &mut Criterion) {
    c.bench_function("greedy_schedule_5x5", |b| {
        b.iter_with_setup(
            || JSProblem::random(5, 5, 42),
            |problem| problem.greedy_schedule(),
        )
    });
    
    c.bench_function("model_forward_pass", |b| {
        let device = Device::Cpu;
        let model = SchedulerModel::new(TransformerConfig::default(), &device)
            .expect("Failed to create model");
        let input = Tensor::zeros((1, 128), DType::Int64, &device)
            .expect("Failed to create tensor");
        
        b.iter(|| {
            let _ = model.forward(black_box(&input));
        })
    });
}

criterion_group!(benches, scheduling_benchmark);
criterion_main!(benches);
```

Run benchmarks:
```bash
cargo bench

# Run specific benchmark
cargo bench -- scheduling_benchmark
```

## 📈 Performance Optimization

### Memory Efficiency

```rust
// Use device memory wisely
let device = Device::Cpu;  // Or Device::Cuda for GPU

// Batch operations
let batch_size = 32;
let input = Tensor::zeros((batch_size, seq_len), DType::Float32, &device)?;

// Use in-place operations where possible
let output = input.clone();
```

### Parallelization

```rust
use rayon::prelude::*;

// Parallel problem generation
let problems: Vec<_> = (0..1000)
    .into_par_iter()
    .map(|i| JSProblem::random(5, 5, i as u64))
    .collect();

// Parallel solution generation
let solutions: Vec<_> = problems
    .par_iter()
    .map(|p| generate_solution(p))
    .collect();
```

### Async Processing

```rust
use tokio::task;

#[tokio::main]
async fn main() -> Result<()> {
    let problems = vec![
        JSProblem::random(5, 5, 1),
        JSProblem::random(5, 5, 2),
        // ...
    ];
    
    let tasks: Vec<_> = problems
        .into_iter()
        .map(|p| {
            task::spawn_blocking(move || {
                schedule_problem(&p)
            })
        })
        .collect();
    
    let results = futures::future::join_all(tasks).await;
    Ok(())
}
```

## 📊 Evaluation

### Solution Quality Metrics

```rust
pub struct SolutionMetrics {
    pub makespan: usize,
    pub feasible: bool,
    pub lateness: usize,
}

pub fn evaluate_solution(
    solution: &str,
    problem: &JSProblem,
) -> SolutionMetrics {
    // Extract makespan from solution text
    let makespan = extract_makespan(solution).unwrap_or(0);
    
    // Check feasibility
    let feasible = validate_schedule(solution, problem);
    
    SolutionMetrics {
        makespan,
        feasible,
        lateness: 0,
    }
}

pub fn compare_with_baseline(
    llm_solution: &str,
    baseline_solution: &str,
) -> f32 {
    let llm_makespan = extract_makespan(llm_solution).unwrap_or(0);
    let baseline_makespan = extract_makespan(baseline_solution).unwrap_or(0);
    
    if baseline_makespan == 0 {
        return 0.0;
    }
    
    (1.0 - llm_makespan as f32 / baseline_makespan as f32) * 100.0
}
```

## 🚢 Deployment

### Building for Production

```bash
# Optimize build
cargo build --release

# Check binary size
ls -lh target/release/jssp_scheduler

# Strip debugging symbols
strip target/release/jssp_scheduler
```

### Docker Deployment

```dockerfile
FROM rust:1.70 as builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim

COPY --from=builder /app/target/release/jssp_scheduler /usr/local/bin/

ENTRYPOINT ["jssp_scheduler"]
```

Build and run:
```bash
docker build -t jssp_scheduler .
docker run jssp_scheduler
```

### Command-Line Interface

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "JSSP Scheduler")]
#[command(about = "LLM-based Job Shop Scheduling", long_about = None)]
struct Args {
    /// Number of jobs
    #[arg(short, long, default_value_t = 5)]
    jobs: usize,
    
    /// Number of machines
    #[arg(short, long, default_value_t = 5)]
    machines: usize,
    
    /// Dataset size
    #[arg(short, long, default_value_t = 500)]
    dataset_size: usize,
    
    /// Number of epochs
    #[arg(short, long, default_value_t = 3)]
    epochs: usize,
}

fn main() {
    let args = Args::parse();
    println!("Training with {}x{} problems", args.jobs, args.machines);
}
```

Usage:
```bash
# Run with defaults
cargo run --release

# Custom parameters
cargo run --release -- --jobs 10 --machines 10 --epochs 5
```

## 📚 Useful Resources

### Candle Documentation
- **Candle Book**: https://huggingface.co/docs/candle/
- **Repository**: https://github.com/huggingface/candle
- **Examples**: https://github.com/huggingface/candle/tree/main/examples

### Rust Resources
- **Rust Book**: https://doc.rust-lang.org/book/
- **Rust by Example**: https://doc.rust-lang.org/rust-by-example/
- **Tokio Tutorial**: https://tokio.rs/tokio/tutorial

### Optimization
- **Criterion.rs**: https://bheisler.github.io/criterion.rs/book/
- **Rayon**: https://docs.rs/rayon/

## 🐛 Troubleshooting

### Compilation Issues

**Problem**: `error: linker not found`
**Solution**:
```bash
# Install build tools
# Ubuntu/Debian
sudo apt install build-essential

# macOS
xcode-select --install
```

**Problem**: CUDA not found
**Solution**:
```bash
# Use CPU version
cargo build

# For GPU, install CUDA toolkit and set environment
export CUDA_HOME=/usr/local/cuda
cargo build --features cuda
```

### Runtime Issues

**Problem**: Out of memory
**Solution**:
```rust
// Use smaller batch size
let config = TrainerConfig {
    batch_size: 1,  // Reduce from default
    ..Default::default()
};
```

**Problem**: Slow performance
**Solution**:
```bash
# Use release build
cargo build --release
cargo run --release

# Enable optimizations
# (see profile.release in Cargo.toml)
```

## ✅ Checklist

- [ ] Install Rust
- [ ] Create Cargo project
- [ ] Add Candle dependencies
- [ ] Implement core structures
- [ ] Add training loop
- [ ] Implement inference
- [ ] Write tests
- [ ] Add benchmarks
- [ ] Create CLI
- [ ] Build release binary
- [ ] Deploy

## 📝 Citation

If you use this Rust implementation, cite the original paper:

```bibtex
@article{abgaryan2024llms,
  title={LLMs can Schedule},
  author={Abgaryan, Henrik and Harutyunyan, Ararat and Cazenave, Tristan},
  journal={arXiv preprint arXiv:2408.06993},
  year={2024}
}
```

---

**Ready to build! 🚀**

For more information, check the source code comments and the original paper.
