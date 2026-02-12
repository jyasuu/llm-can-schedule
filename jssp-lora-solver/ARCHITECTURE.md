# JSSP LoRA Architecture & Workflows

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     JSSP LoRA Solver System                     │
└─────────────────────────────────────────────────────────────────┘

                        ┌─────────────┐
                        │   Dataset   │
                        │  (5 JSSP    │
                        │ instances)  │
                        └──────┬──────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
            ┌────────┐    ┌────────┐    ┌────────┐
            │ Job 0  │    │ Job 1  │    │ Job 2  │
            │ 3×2    │    │ 4×3    │    │ 5×3    │
            │Problem │    │Problem │    │Problem │
            └────┬───┘    └────┬───┘    └────┬───┘
                 │              │              │
        ┌────────▼─────────┐    │    ┌────────▼─────────┐
        │    Encoding      │    │    │    Encoding      │
        │  (256-dim vec)   │    │    │  (256-dim vec)   │
        └────────┬─────────┘    │    └────────┬─────────┘
                 │              │              │
        ┌────────▼──────────────▼──────────────▼────────┐
        │            LoRA Network                       │
        │  ┌──────────────────────────────────────────┐ │
        │  │  Input Layer (256 → 512)                 │ │
        │  │  W_a: (256 × 8)                          │ │
        │  │  W_b: (8 × 512)                          │ │
        │  └──────────────────────────────────────────┘ │
        │  ┌──────────────────────────────────────────┐ │
        │  │  Hidden Layer (512 → 512) + ReLU         │ │
        │  │  W_a: (512 × 8)                          │ │
        │  │  W_b: (8 × 512)                          │ │
        │  └──────────────────────────────────────────┘ │
        │  ┌──────────────────────────────────────────┐ │
        │  │  Output Layer (512 → 256)                │ │
        │  │  W_a: (512 × 8)                          │ │
        │  │  W_b: (8 × 256)                          │ │
        │  └──────────────────────────────────────────┘ │
        │                                                │
        │  Total params: ~15K (vs ~1.3M full network)  │
        └────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   256-dim Output    │
                    │   (Schedule Info)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Schedule Decoder   │
                    │  - Makespan         │
                    │  - Job sequence     │
                    │  - Times            │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Schedule Output   │
                    │  (Solution)         │
                    └─────────────────────┘
```

## LoRA Layer Detail

```
Mathematical Operation:

h = σ(W₀ @ x + (α/r) × W_b @ W_a @ x)

Where:
  W₀    = Original frozen weight matrix
  W_a   = (input_dim × rank)        Trainable
  W_b   = (rank × output_dim)       Trainable
  α     = Scaling constant (alpha)
  r     = Rank
  σ()   = ReLU activation


Parameter Count Comparison:

Standard Fine-tuning:
  W ∈ ℝ^(256 × 512) = 131,072 parameters

With LoRA (rank=8):
  W_a: 256 × 8     = 2,048 parameters
  W_b: 8 × 512     = 4,096 parameters
  Total            = 6,144 parameters (95% reduction!)

With LoRA (rank=16):
  W_a: 256 × 16    = 4,096 parameters
  W_b: 16 × 512    = 8,192 parameters
  Total            = 12,288 parameters


Forward Pass Example (rank=8, input_dim=256, output_dim=512):

Input: x ∈ ℝ^(batch × 256)

1. W_a @ x                    → ℝ^(batch × 8)
2. W_b @ result_of_step_1    → ℝ^(batch × 512)
3. Scale by (16.0 / 8)       → ℝ^(batch × 512)
4. Add to W₀ @ x             → ℝ^(batch × 512)
5. Apply ReLU                → ℝ^(batch × 512)
```

## Training Loop Workflow

```
┌─────────────────────────────────────────────────┐
│           Training Loop Flowchart               │
└─────────────────────────────────────────────────┘

    START
      │
      ▼
  ┌─────────────────────┐
  │ Initialize LoRA     │
  │ - Rank: 8           │
  │ - Alpha: 16.0       │
  │ - Dropout: 0.1      │
  └─────────────────────┘
      │
      ▼
  ┌─────────────────────┐
  │ Load Dataset        │
  │ 5 JSSP instances    │
  └─────────────────────┘
      │
      ▼
  ┌──────────────────────────────────────────┐
  │ FOR epoch = 1 to 3                       │
  │ ┌────────────────────────────────────┐  │
  │ │ FOR each JSSP instance             │  │
  │ │ ┌──────────────────────────────┐  │  │
  │ │ │ 1. Encode instance to vector │  │  │
  │ │ │    (256-dim features)        │  │  │
  │ │ └──────────────────────────────┘  │  │
  │ │ ┌──────────────────────────────┐  │  │
  │ │ │ 2. Forward pass through LoRA │  │  │
  │ │ │    Input → Hidden → Output   │  │  │
  │ │ └──────────────────────────────┘  │  │
  │ │ ┌──────────────────────────────┐  │  │
  │ │ │ 3. Compute loss              │  │  │
  │ │ │    MSE(predicted, optimal)   │  │  │
  │ │ └──────────────────────────────┘  │  │
  │ │ ┌──────────────────────────────┐  │  │
  │ │ │ 4. Backpropagation           │  │  │
  │ │ │    (update W_a, W_b)         │  │  │
  │ │ └──────────────────────────────┘  │  │
  │ │ ┌──────────────────────────────┐  │  │
  │ │ │ 5. Accumulate loss           │  │  │
  │ │ └──────────────────────────────┘  │  │
  │ └────────────────────────────────────┘  │
  │ ┌────────────────────────────────────┐  │
  │ │ Report average loss for epoch      │  │
  │ └────────────────────────────────────┘  │
  └──────────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────┐
  │ Test on validation set     │
  │ - Generate schedules       │
  │ - Compare with optimal     │
  │ - Calculate gap %          │
  └────────────────────────────┘
      │
      ▼
    END
```

## JSSP Instance Structure

```
JSSP Instance (Example: 3 jobs, 2 machines)

┌──────────────────────────────────────────────────┐
│  num_jobs = 3                                    │
│  num_machines = 2                                │
│                                                  │
│  Processing Times (flattened):                  │
│  [5, 4, 4, 5, 3, 6]                            │
│   └─┬──┘ └─┬──┘ └─┬──┘                          │
│    Job0  Job1  Job2                             │
│                                                  │
│  Machine Sequences (operations order):          │
│  Job 0: [0, 1]   (M0 → M1)                     │
│  Job 1: [0, 1]   (M0 → M1)                     │
│  Job 2: [0, 1]   (M0 → M1)                     │
│                                                  │
│  Processing Time Matrix:                       │
│         M0  M1                                  │
│  Job 0:  5   4                                  │
│  Job 1:  4   5                                  │
│  Job 2:  3   6                                  │
└──────────────────────────────────────────────────┘

Schedule Generation:

Input: [0, 1, 2] (process jobs in order 0, 1, 2)

Timeline:
┌─────────────────────────────────────────────────┐
│ Machine 0:  [Job0:5] [Job1:4] [Job2:3]         │
│ Finish time: 5   →   9   →   12                │
├─────────────────────────────────────────────────┤
│ Machine 1:              [Job0:4] [Job1:5] [Job2:6]
│ Finish time:        4   →   9   →   15       │
└─────────────────────────────────────────────────┘

Output Schedule:
  job_sequence: [0, 1, 2]
  completion_times: [9, 14, 15]
  makespan: 15 ← Total project time
```

## Data Flow: From Problem to Solution

```
JSSP Instance
    │
    │  .encode() → Vec<f32> (256 elements)
    ▼
    
Input Features
    │
    │  [problem_size_info, normalized_times, ...]
    ▼
    
LoRA Network Forward Pass
    │
    ├─ Layer 1: (256 → 512)
    │   LoRA: W_a @ x → W_b @ result
    │   ReLU activation
    │
    ├─ Layer 2: (512 → 512)
    │   LoRA: W_a @ x → W_b @ result
    │   ReLU activation
    │
    └─ Layer 3: (512 → 256)
       LoRA: W_a @ x → W_b @ result
       (No activation on output)
    │
    ▼
    
Output Vector (256-dim)
    │
    │  [makespan, completion_times, ...]
    ▼
    
Schedule Decoder
    │
    ├─ Extract makespan
    ├─ Extract completion times
    └─ Derive job sequence
    │
    ▼
    
Solution Schedule
    │
    ├─ job_sequence: Vec<usize>
    ├─ makespan: u32
    └─ completion_times: Vec<u32>
```

## Solver Comparison

```
Solver Comparison for Example Problem:

┌──────────────────────────────────────────────────────┐
│  Problem: 3 jobs, 2 machines                        │
│  Times: [5,4], [4,5], [3,6]                         │
└──────────────────────────────────────────────────────┘

Greedy Solver [0,1,2]:
    Schedule:  Job0(5,4) → Job1(4,5) → Job2(3,6)
    Makespan:  15
    Time:      <1ms
    
Nearest Neighbor [2,0,1]:
    Schedule:  Job2(3,6) → Job0(5,4) → Job1(4,5)
    Makespan:  14
    Time:      <1ms
    
Random Solver:
    Attempt 1 [1,2,0]:  Makespan: 16
    Attempt 2 [2,1,0]:  Makespan: 13  ← Good!
    Time:      <1ms
    
LoRA Neural Solver:
    Training:  3 epochs on 5 instances = ~1-2 min
    Inference: Instance encode → LoRA forward → schedule
    Time:      ~5-10ms
    Quality:   Learns from training data
    
┌──────────────────────────────────────────────────────┐
│  Rank     │ Params  │ Time  │ Quality │ Memory      │
├──────────────────────────────────────────────────────┤
│  Greedy   │ 0       │ <1ms  │ Poor    │ Minimal    │
│  NN       │ 0       │ <1ms  │ Fair    │ Minimal    │
│  Random   │ 0       │ <1ms  │ Fair    │ Minimal    │
│  LoRA-4   │ 5K      │ 5ms   │ Good    │ 50MB       │
│  LoRA-8   │ 10K     │ 8ms   │ Better  │ 50MB       │
│  LoRA-16  │ 20K     │ 15ms  │ Best    │ 60MB       │
│  Full-NN  │ 150K    │ 20ms  │ Best    │ 200MB      │
└──────────────────────────────────────────────────────┘
```

## Training Progress Example

```
Epoch 1: Processing instances
  Instance 1 (3×2): Loss = 0.234
  Instance 2 (4×3): Loss = 0.198
  Instance 3 (5×3): Loss = 0.267
  Instance 4 (3×2): Loss = 0.189
  Instance 5 (4×3): Loss = 0.212
  ─────────────────────────────
  Average Loss = 0.220

Epoch 2: Training continues
  Instance 1 (3×2): Loss = 0.156
  Instance 2 (4×3): Loss = 0.134
  Instance 3 (5×3): Loss = 0.178
  Instance 4 (3×2): Loss = 0.122
  Instance 5 (4×3): Loss = 0.145
  ─────────────────────────────
  Average Loss = 0.147  ← Improving!

Epoch 3: Final epoch
  Instance 1 (3×2): Loss = 0.098
  Instance 2 (4×3): Loss = 0.087
  Instance 3 (5×3): Loss = 0.112
  Instance 4 (3×2): Loss = 0.075
  Instance 5 (4×3): Loss = 0.093
  ─────────────────────────────
  Average Loss = 0.093  ← Converged!

Test Results:
  Example 1 (3×2): Predicted: 15, Optimal: 13 → Gap: 15.4%
  Example 2 (4×3): Predicted: 12, Optimal: 11 → Gap: 9.1%
  Example 3 (3×2): Predicted: 14, Optimal: 14 → Gap: 0.0% ✓
```

## Configuration Tuning Space

```
Parameter Tuning Guide:

               PROBLEM SIZE
                    ▲
            Large   │  ╔═══════════════╗
                    │  ║ High Rank:    ║
                    │  ║ rank=32       ║
                    │  ║ alpha=64      ║
                    │  ║ Slow Training ║
                    │  ╚═══════════════╝
                    │         │
            Medium  │    ╔════╩════╗
                    │    ║ Medium: ║
                    │    ║ rank=16 ║
                    │    ║ alpha=32║
                    │    ╚════╦════╝
                    │         │
            Small   │  ╔══════╩═════╗
                    │  ║ Low Rank:   ║
                    │  ║ rank=4-8    ║
                    │  ║ alpha=8-16  ║
                    │  ║ Fast        ║
                    │  ╚═════════════╝
                    │
                    └──────────────────────────→ SPEED REQUIREMENT

Learning Rate Sensitivity:
  1e-5  ← Too slow convergence
  1e-4  ← Recommended (default)
  1e-3  ← Fast but may overshoot
  1e-2  ← Too fast, unstable
```

These diagrams show:
1. Overall system architecture
2. LoRA mathematics and parameters
3. Training workflow and loop
4. JSSP problem structure
5. Data transformations
6. Solver comparisons
7. Training progress
8. Configuration tuning space
