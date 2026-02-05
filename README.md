# Job Shop Scheduling with LLMs

Based on the paper **"LLMs can Schedule"** (arXiv:2408.06993), this repository provides implementations for training and using Large Language Models to solve the Job Shop Scheduling Problem (JSSP).

## 📋 Overview

The Job Shop Scheduling Problem (JSSP) is a combinatorial optimization challenge where:
- **Goal**: Schedule N jobs on M machines
- **Constraints**: Each job has ordered operations that must be completed sequentially
- **Objective**: Minimize makespan (total completion time)

The paper demonstrates that LLMs with LoRA fine-tuning can achieve competitive performance for JSSP compared to traditional neural approaches.

## 📂 Files

### 1. `jssp_llm_scheduler.py` - Simplified Implementation
A standalone implementation with:
- Custom transformer model from scratch
- LoRA adapter implementation
- JSSP dataset generation
- Training loop with validation
- Inference with multiple sampling strategies

**Best for**: Learning, understanding the architecture, quick prototyping

**Dependencies**:
```bash
pip install torch numpy tqdm
```

### 2. `jssp_scheduler_hf.py` - Production Implementation
Production-ready code using:
- Hugging Face Transformers
- PEFT library for LoRA
- HuggingFace Trainer API
- Proper tokenization and generation

**Best for**: Training on actual models (GPT-2, Llama-2, etc.), production use

**Dependencies**:
```bash
pip install torch transformers peft datasets accelerate
```

## 🚀 Quick Start

### Option 1: Simple Standalone Version

```python
from jssp_llm_scheduler import JSSproblem, JSSpSchedulerTrainer, JSSpScheduler, Config

# Configuration
config = Config(
    num_jobs=10,
    num_machines=10,
    batch_size=4,
    num_epochs=3,
    dataset_size=1000,
)

# Train
trainer = JSSpSchedulerTrainer(config)
trainer.train()
trainer.save_model("jssp_model.pt")

# Inference
scheduler = JSSpScheduler(trainer.model, trainer.device, sampling_method='greedy')
problem = JSSproblem(num_jobs=5, num_machines=5)
solution = scheduler.schedule(problem)
print(solution)
```

### Option 2: Production with Hugging Face

```python
from jssp_scheduler_hf import JSSproblem, JSSpTrainer, JSSpScheduler, Config

# Configuration
config = Config(
    model_name="gpt2",  # or "meta-llama/Llama-2-7b"
    num_jobs=5,
    num_machines=5,
    train_size=500,
    batch_size=4,
    num_epochs=3,
)

# Train with LoRA
trainer = JSSpTrainer(config)
trainer.train()

# Inference with different sampling methods
scheduler = JSSpScheduler("./jssp_lora_model")

problem = JSSproblem(num_jobs=3, num_machines=3)
print(problem.to_text())

# Try different sampling strategies
solutions = {
    'greedy': scheduler.schedule(problem, sampling_method='greedy'),
    'nucleus': scheduler.schedule(problem, sampling_method='nucleus', temperature=0.8),
    'top_k': scheduler.schedule(problem, sampling_method='top_k', top_k=50),
    'beam': scheduler.schedule(problem, sampling_method='beam'),
}

for method, solution in solutions.items():
    print(f"\n{method.upper()} Solution:\n{solution}")
```

## 🔧 Configuration Guide

### LoRA Parameters
```python
Config(
    lora_r=8,           # Rank of LoRA matrices (lower = fewer params)
    lora_alpha=16,      # Scaling factor
    lora_dropout=0.05,  # Dropout in LoRA layers
    lora_target_modules=["c_attn", "c_proj"],  # Which layers to apply LoRA to
)
```

**Choosing LoRA parameters**:
- Smaller `r` (4-8): Less memory, faster training, may lose expressiveness
- Larger `r` (16-32): More memory, slower training, better expressiveness
- `alpha = 2 * r` is a good default

### Training Parameters
```python
Config(
    batch_size=4,           # Increase for faster training (needs more VRAM)
    learning_rate=1e-4,     # LoRA LR typically 1e-4 to 5e-4
    num_epochs=3,           # Increase for harder problems
    max_grad_norm=1.0,      # Gradient clipping
    warmup_steps=500,       # Linear warmup
)
```

### JSSP Parameters
```python
Config(
    num_jobs=10,                    # Number of jobs
    num_machines=10,                # Number of machines
    max_operation_duration=20,      # Max operation duration
)
```

## 📊 Dataset

The implementations generate synthetic JSSP instances on-the-fly. Each instance includes:
- Random job generation
- Random operation-to-machine assignment
- Greedy baseline solutions

### Custom Dataset

To use your own JSSP dataset:

```python
class CustomJSSpDataset(Dataset):
    def __init__(self, problems, solutions, tokenizer, config):
        self.samples = []
        for problem, solution in zip(problems, solutions):
            text = problem.to_text() + "\nSolve:\n" + solution
            self.samples.append(text)
    
    def __getitem__(self, idx):
        return self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
```

## 🎯 Inference Strategies

### Sampling Methods

1. **Greedy Decoding**
   - Fastest, deterministic
   - Best when well-trained
   ```python
   scheduler.schedule(problem, sampling_method='greedy')
   ```

2. **Top-k Sampling**
   - Samples from k most likely tokens
   - Reduces repetition
   ```python
   scheduler.schedule(problem, sampling_method='top_k', top_k=50, temperature=0.8)
   ```

3. **Nucleus (Top-p) Sampling**
   - Dynamically adjusts k based on cumulative probability
   - Better quality samples
   ```python
   scheduler.schedule(problem, sampling_method='nucleus', top_p=0.95, temperature=0.8)
   ```

4. **Beam Search**
   - Explores multiple hypotheses
   - Slower but often better quality
   ```python
   scheduler.schedule(problem, sampling_method='beam')
   ```

## 📈 Evaluation Metrics

To evaluate solution quality:

```python
from jssp_scheduler_hf import evaluate_solution

solution = scheduler.schedule(problem)
metrics = evaluate_solution(solution)

print(f"Makespan: {metrics['makespan']}")
print(f"Valid: {metrics['valid']}")
print(f"Length: {metrics['length']}")
```

Compare against baselines:
```python
baseline = JSSproblem.solve_greedy(problem)
baseline_metrics = evaluate_solution(baseline)

print(f"Baseline makespan: {baseline_metrics['makespan']}")
print(f"LLM makespan: {metrics['makespan']}")
print(f"Improvement: {1 - metrics['makespan']/baseline_metrics['makespan']:.2%}")
```

## 🔬 Advanced Usage

### Using Different Models

```python
from jssp_scheduler_hf import Config, JSSpTrainer

# GPT-2 (small, fast)
config = Config(model_name="gpt2", ...)

# Larger models
config = Config(model_name="gpt2-medium", ...)
config = Config(model_name="gpt2-large", ...)

# Llama-2 (better quality, larger)
config = Config(model_name="meta-llama/Llama-2-7b-hf", ...)

# Other options
config = Config(model_name="EleutherAI/gpt-neo-125M", ...)
```

### Multi-GPU Training

```python
from jssp_scheduler_hf import JSSpTrainer, Config

config = Config(...)
trainer = JSSpTrainer(config)

# The TrainingArguments will automatically use all available GPUs
trainer.training_args.ddp_find_unused_parameters = False
trainer.train()
```

### Adapter Merging (Export LoRA)

```python
from peft import PeftModel

# Load model with LoRA
model = AutoModelForCausalLM.from_pretrained("./jssp_lora_model")

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./jssp_merged_model")
```

### Fine-tune on Specific Problem Size

```python
# Generate dataset for 15x15 problems
config = Config(
    num_jobs=15,
    num_machines=15,
    train_size=1000,  # More samples needed for harder problems
)

trainer = JSSpTrainer(config)
trainer.train()
```

## 📚 Paper Reference

**"LLMs can Schedule"** by Henrik Abgaryan, Ararat Harutyunyan, Tristan Cazenave
- arXiv: 2408.06993
- Key findings:
  - LoRA fine-tuning enables LLMs to solve JSSP
  - Performance comparable to graph neural networks
  - Can be enhanced with sampling methods
  - 120k supervised dataset for JSSP

## 🛠️ Troubleshooting

### Out of Memory

1. Reduce batch size:
   ```python
   config.batch_size = 1
   ```

2. Use gradient accumulation:
   ```python
   trainer.training_args.gradient_accumulation_steps = 4
   ```

3. Use smaller model:
   ```python
   config.model_name = "gpt2"  # instead of larger model
   ```

### Poor Solution Quality

1. Increase training data:
   ```python
   config.train_size = 5000  # from 500
   ```

2. Train longer:
   ```python
   config.num_epochs = 10
   ```

3. Use better sampling:
   ```python
   scheduler.schedule(problem, sampling_method='beam', num_beams=10)
   ```

### Slow Generation

1. Use greedy decoding:
   ```python
   scheduler.schedule(problem, sampling_method='greedy')
   ```

2. Reduce max_length:
   ```python
   scheduler.schedule(problem, max_length=256)
   ```

3. Use quantization:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(
       "gpt2",
       torch_dtype=torch.float16,  # Use FP16
   )
   ```

## 📝 Citation

If you use this code, please cite the original paper:

```bibtex
@article{abgaryan2024llms,
  title={LLMs can Schedule},
  author={Abgaryan, Henrik and Harutyunyan, Ararat and Cazenave, Tristan},
  journal={arXiv preprint arXiv:2408.06993},
  year={2024}
}
```

## 📄 License

This implementation is provided for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. Real JSSP solver integration (instead of greedy baseline)
2. Evaluation metrics and benchmarking
3. Multi-job batch scheduling
4. Integration with constraint solvers
5. Hybrid approaches (LLM + local search)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the paper for methodology details
3. Refer to HuggingFace Transformers documentation
4. Check PEFT library documentation for LoRA details

---

**Happy Scheduling! 🚀**
