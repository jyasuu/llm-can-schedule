# JSSP LoRA Solver - Complete Implementation Guide

## Quick Start

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Create and Setup Project
```bash
cargo new jssp-lora-solver
cd jssp-lora-solver
```

### 3. Update Cargo.toml
Replace the entire `Cargo.toml` with:

```toml
[package]
name = "jssp-lora-solver"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = { version = "0.7", features = ["cuda"] }
candle-nn = "0.7"
candle-transformers = "0.7"
tokenizers = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
rand = "0.8"
num-traits = "0.2"

[profile.release]
opt-level = 3
lto = true
```

### 4. Create Module Files

Create `src/lora.rs`, `src/jssp.rs`, `src/data.rs` with the contents provided earlier.

Update `src/main.rs` with the main program.

### 5. Build and Run
```bash
# For CPU (recommended for first run)
cargo build --release
cargo run --release

# For GPU (requires CUDA toolkit)
cargo build --release --features cuda
cargo run --release
```

## File Organization

```
jssp-lora-solver/
├── Cargo.toml                    # Project configuration
├── src/
│   ├── main.rs                   # Main program
│   ├── lora.rs                   # LoRA implementation
│   ├── jssp.rs                   # JSSP problem definition
│   └── data.rs                   # Data loading and generation
├── examples/
│   └── jssp_examples.rs          # Practical examples
└── README.md                     # Documentation
```

## Code Components Explained

### 1. LoRA Module (`src/lora.rs`)

The Low-Rank Adaptation technique reduces trainable parameters:

**Traditional fine-tuning:**
```
Weight matrix: (d_in × d_out) = millions of parameters
```

**With LoRA:**
```
W_a: (d_in × rank)        = thousands
W_b: (rank × d_out)       = thousands
Total: 2 × d_in × rank << d_in × d_out
```

**Forward pass:**
```
output = LoRA_scale × W_b @ W_a @ input
where LoRA_scale = alpha / rank
```

Key classes:
- `LoRALinear`: Single LoRA layer
- `LoRA`: Multi-layer LoRA network
- Configurable rank, alpha, dropout

### 2. JSSP Module (`src/jssp.rs`)

Job Shop Scheduling Problem components:

**JSSPInstance:** Defines a scheduling problem
- `num_jobs`: Number of jobs
- `num_machines`: Number of machines  
- `processing_times`: Time required for each job on each machine
- `machine_sequences`: Order machines must be visited for each job

**Schedule:** Represents a solution
- `job_sequence`: Order to process jobs
- `makespan`: Total completion time
- `completion_times`: When each job finishes

**JSSPSolver:** Multiple solution strategies
- `solve_greedy()`: Process jobs in order 0,1,2,...
- `solve_nearest_neighbor()`: Greedy ordering by distance
- `solve_random()`: Random job permutation

Encoding for neural network:
```rust
// Convert JSSP instance to 256-dimensional feature vector
instance.encode() -> Vec<f32>
```

### 3. Data Module (`src/data.rs`)

Provides benchmark instances and dataset management:

**Sample Instances:**
- 3×2: 3 jobs, 2 machines (simple)
- 4×3: 4 jobs, 3 machines (medium)
- 5×3: 5 jobs, 3 machines (larger)

**Optimal Schedules:**
Generated using best heuristic from:
- Greedy
- Nearest Neighbor
- Both approaches

**Usage:**
```rust
let dataset = load_sample_data()?;
for instance in &dataset.instances {
    // Process each instance
}
```

## Training Loop Explained

The main training procedure in `src/main.rs`:

```
1. Initialize LoRA model
   └─ VarMap for parameter storage
   └─ LoRA layers with rank=8, alpha=16

2. Load training data (5 JSSP instances)
   └─ Each with optimal reference schedule

3. For each epoch (3 epochs):
   └─ For each training instance:
      ├─ Encode instance to features
      ├─ Forward pass through LoRA
      ├─ Compute loss vs optimal schedule
      └─ Accumulate loss
   └─ Report average loss

4. Test trained model
   └─ Solve new instances
   └─ Compare with optimal
   └─ Calculate gap percentage
```

## Example JSSP Problem

### Problem Definition
```
Job 0: 5 on M0, 4 on M1  (sequence: M0 → M1)
Job 1: 4 on M0, 5 on M1  (sequence: M0 → M1)
Job 2: 3 on M0, 6 on M1  (sequence: M0 → M1)
```

### Gantt Chart for Schedule [0,1,2]
```
Machine 0:  [Job0:5] [Job1:4] [Job2:3]  Total: 12
Machine 1:                [Job0:4] [Job1:5] [Job2:6]  Total: 15

Makespan (completion time) = 15
```

### Different Schedules
```
[0,1,2]: Makespan = 15
[0,2,1]: Makespan = 14
[2,1,0]: Makespan = 13  ← Likely optimal or near-optimal
```

## Configuration Tuning

### Hyperparameters

```rust
TrainingConfig {
    learning_rate: 1e-4,      // Gradient step size
    epochs: 3,                // Number of training passes
    batch_size: 8,            // Samples per batch
    lora_rank: 8,             // Decomposition rank
    lora_alpha: 16.0,         // Scaling factor (usually 2×rank)
    dropout: 0.1,             // Regularization
}
```

### Tuning Guide

**For small problems (3×2 to 4×3):**
```rust
lora_rank: 4,
lora_alpha: 8.0,
epochs: 5,
learning_rate: 1e-3,
```

**For medium problems (5×5 to 10×10):**
```rust
lora_rank: 16,
lora_alpha: 32.0,
epochs: 10,
learning_rate: 1e-4,
```

**For large problems (15×15+):**
```rust
lora_rank: 32,
lora_alpha: 64.0,
epochs: 20,
learning_rate: 1e-5,
```

## Adding Custom JSSP Instances

### Method 1: Hardcoded Instance
```rust
// In src/data.rs
fn create_my_problem() -> Result<JSSPInstance> {
    let processing_times = vec![
        // Job 0: [time_on_M0, time_on_M1, time_on_M2]
        5, 3, 4,
        // Job 1
        2, 6, 3,
        // Job 2
        4, 2, 5,
    ];

    let machine_sequences = vec![
        vec![0, 1, 2],  // Job 0: M0 → M1 → M2
        vec![2, 0, 1],  // Job 1: M2 → M0 → M1
        vec![1, 2, 0],  // Job 2: M1 → M2 → M0
    ];

    JSSPInstance::new(3, 3, processing_times, machine_sequences)
}
```

### Method 2: From JSON (Starjob Format)
```rust
use serde_json::json;

fn load_from_starjob(json_str: &str) -> Result<JSSPInstance> {
    let value: serde_json::Value = serde_json::from_str(json_str)?;
    
    let num_jobs = value["num_jobs"].as_u64().unwrap() as usize;
    let num_machines = value["num_machines"].as_u64().unwrap() as usize;
    
    let processing_times: Vec<u32> = value["matrix"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    
    // Generate sequential machine ordering for simplicity
    let machine_sequences = (0..num_jobs)
        .map(|_| (0..num_machines).collect())
        .collect();
    
    JSSPInstance::new(num_jobs, num_machines, processing_times, machine_sequences)
}
```

### Method 3: Random Generator
```rust
fn generate_random_instance(jobs: usize, machines: usize) -> Result<JSSPInstance> {
    use rand::Rng;
    
    let mut rng = rand::thread_rng();
    let processing_times = (0..jobs*machines)
        .map(|_| rng.gen_range(1..10))
        .collect();
    
    let machine_sequences = (0..jobs)
        .map(|_| {
            let mut seq: Vec<usize> = (0..machines).collect();
            use rand::seq::SliceRandom;
            seq.shuffle(&mut rng);
            seq
        })
        .collect();
    
    JSSPInstance::new(jobs, machines, processing_times, machine_sequences)
}
```

## Performance Optimization

### Memory Optimization
```rust
// Use smaller rank for limited memory
LoRA {
    rank: 4,        // Instead of 8
    alpha: 8.0,     // Instead of 16
    dropout: 0.0,   // Skip dropout
}
```

### Speed Optimization
```rust
// Use Release build
cargo build --release

// Enable LTO in Cargo.toml
[profile.release]
opt-level = 3
lto = true  # Link-time optimization
```

### GPU Acceleration
```bash
# Install CUDA toolkit first
# Then build with CUDA support
cargo build --release --features cuda

# Verify GPU usage in code
let device = if candle_core::utils::cuda_is_available() {
    Device::cuda(0)?  // GPU 0
} else {
    Device::Cpu
};
```

## Testing

### Unit Tests
```bash
# Run all tests
cargo test --release

# Run specific module tests
cargo test jssp::tests
cargo test lora::tests
cargo test data::tests

# With output
cargo test --release -- --nocapture
```

### Integration Tests
```rust
#[test]
fn test_full_pipeline() -> Result<()> {
    let device = Device::Cpu;
    let config = TrainingConfig::default();
    let mut solver = JSSPLoRASolver::new(device, config)?;
    
    let dataset = load_sample_data()?;
    let schedule = solver.solve(&dataset.instances[0])?;
    
    assert!(schedule.makespan > 0);
    Ok(())
}
```

## Troubleshooting

### Common Issues

**1. Compilation fails with "candle-core not found"**
```
Solution: Update dependencies
cargo update
cargo clean
cargo build --release
```

**2. Runtime: "CUDA not found" but --features cuda used**
```
Solution: Remove CUDA feature for CPU-only
cargo build --release  # Without --features cuda
```

**3. Out of memory during training**
```
Solution: Reduce problem size or batch size
// In main.rs
config.batch_size = 4;  // Was 8
config.lora_rank = 4;   // Was 8
```

**4. Slow training on CPU**
```
Solution: This is normal, but you can:
- Use --release build
- Reduce problem size
- Use GPU if available
- Reduce epochs
```

## Integration with Starjob Dataset

To use the full Starjob dataset:

```rust
// Download from huggingface.co/datasets/henri24/Starjob
// Save as data/starjob130k.json

fn load_starjob_dataset() -> Result<JSSPDataset> {
    let file = std::fs::File::open("data/starjob130k.json")?;
    let reader = std::io::BufReader::new(file);
    
    let entries: Vec<StarjobEntry> = serde_json::from_reader(reader)?;
    
    let instances = entries.iter()
        .map(|e| load_from_starjob_entry(e))
        .collect::<Result<Vec<_>>>()?;
    
    // ... compute optimal schedules
    
    Ok(JSSPDataset {
        instances,
        optimal_schedules,
    })
}
```

## Next Steps

1. **Run the basic example**: `cargo run --release`
2. **Explore examples**: `cargo run --example jssp_examples`
3. **Add custom problems**: Modify `src/data.rs`
4. **Integrate Starjob dataset**: Implement dataset loader
5. **Train on larger instances**: Increase problem size
6. **Optimize with GPU**: Add CUDA support
7. **Benchmark performance**: Compare with classical solvers
8. **Extend with new solvers**: Add your own algorithms

## References

- **LoRA**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Candle**: [GitHub Candle](https://github.com/huggingface/candle)
- **JSSP**: [Wikipedia JSSP](https://en.wikipedia.org/wiki/Job_shop_scheduling)
- **Starjob**: [GitHub Starjob](https://github.com/starjob42/Starjob)
