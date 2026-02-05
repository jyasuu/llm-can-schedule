# JSSP LLM Scheduler - Quick Reference Card

## 📥 Installation

```bash
# Option 1: Standalone
pip install torch numpy tqdm

# Option 2: Production (Recommended)
pip install torch transformers peft datasets accelerate
```

## 🚀 Basic Usage (30 seconds)

```python
from jssp_scheduler_hf import JSSproblem, JSSpTrainer, JSSpScheduler, Config

# Configure
config = Config(num_jobs=5, num_machines=5, batch_size=2, num_epochs=1)

# Train
trainer = JSSpTrainer(config)
trainer.train()

# Inference
scheduler = JSSpScheduler(config.output_dir)
problem = JSSproblem(num_jobs=3, num_machines=3)
solution = scheduler.schedule(problem)
print(solution)
```

## ⚙️ Configuration Quick Guide

```python
from jssp_scheduler_hf import Config

config = Config(
    # Model
    model_name="gpt2",                    # or "meta-llama/Llama-2-7b-hf"
    
    # LoRA
    lora_r=8,                             # Rank (4-16 typical)
    lora_alpha=16,                        # Alpha (usually 2*r)
    lora_dropout=0.05,                    # Dropout
    
    # Training
    batch_size=4,                         # Reduce if OOM
    num_epochs=3,                         # More = better
    learning_rate=1e-4,                   # LoRA LR
    
    # JSSP
    num_jobs=5,                           # Problem size
    num_machines=5,
    max_operation_duration=20,
    
    # Dataset
    train_size=500,                       # More = better
    val_size=100,
    
    # Output
    output_dir="./jssp_model",
)
```

## 📊 Sampling Strategies

```python
scheduler = JSSpScheduler("./model")
problem = JSSproblem(num_jobs=3, num_machines=3)

# Greedy (fastest, deterministic)
solution = scheduler.schedule(problem, sampling_method='greedy')

# Top-k (explore alternatives)
solution = scheduler.schedule(
    problem, 
    sampling_method='top_k', 
    top_k=50, 
    temperature=0.8
)

# Nucleus/Top-p (balanced)
solution = scheduler.schedule(
    problem,
    sampling_method='nucleus',
    top_p=0.95,
    temperature=0.8
)

# Beam Search (most thorough)
solution = scheduler.schedule(
    problem,
    sampling_method='beam',
    num_beams=5
)
```

## 🎯 Problem Representation

```python
# Create problem
problem = JSSproblem(num_jobs=3, num_machines=3, seed=42)

# View as text
print(problem.to_text())
# Output:
# Problem: Job Shop Scheduling
# Machines: 3, Jobs: 3
# 
# Job specifications:
#   J0: M2:4h -> M0:3h -> M1:2h
#   J1: M1:5h -> M2:2h -> M0:3h
#   J2: M0:2h -> M1:4h -> M2:3h

# Greedy solution
solution = JSSproblem.solve_greedy(problem)
print(solution)
```

## 📈 Training Different Problem Sizes

```python
# Small problems (quick training)
config = Config(num_jobs=3, num_machines=3, train_size=100)

# Medium problems (balanced)
config = Config(num_jobs=5, num_machines=5, train_size=500)

# Large problems (needs more data & time)
config = Config(num_jobs=10, num_machines=10, train_size=2000, num_epochs=10)
```

## 🔍 Evaluation

```python
from jssp_scheduler_hf import evaluate_solution

# Generate solution
solution = scheduler.schedule(problem)

# Evaluate
metrics = evaluate_solution(solution)
print(metrics)
# {'makespan': 15, 'valid': True, 'length': 342}

# Compare with baseline
baseline = JSSproblem.solve_greedy(problem)
baseline_metrics = evaluate_solution(baseline)

improvement = (1 - metrics['makespan'] / baseline_metrics['makespan']) * 100
print(f"Improvement: {improvement:+.1f}%")
```

## 🔄 Batch Processing

```python
# Multiple problems
problems = [
    JSSproblem(num_jobs=5, num_machines=5, seed=i)
    for i in range(10)
]

# Batch schedule
solutions = scheduler.batch_schedule(problems, sampling_method='greedy')

# Evaluate
for problem, solution in zip(problems, solutions):
    metrics = evaluate_solution(solution)
    print(f"Makespan: {metrics['makespan']}")
```

## 💾 Model Management

```python
from transformers import AutoModelForCausalLM

# Save trained model
trainer.trainer.save_model("./my_model")

# Load for inference
scheduler = JSSpScheduler("./my_model")

# Export merged model (remove LoRA)
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("./my_model")
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")

# Load merged model
scheduler = JSSpScheduler("./merged_model")
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce `batch_size` to 1 |
| Slow Training | Use smaller model (gpt2) |
| Poor Solutions | Increase `train_size` and `num_epochs` |
| Slow Generation | Use `sampling_method='greedy'` |
| Model not improving | Check `learning_rate` (try 1e-4) |

## ⚡ Performance Tips

```python
# Fast training
config = Config(
    model_name="gpt2",          # Smaller model
    batch_size=4,               # Larger batches
    num_epochs=1,               # Fewer epochs
    train_size=100,             # Less data
    lora_r=4,                   # Smaller rank
)

# High quality (slower)
config = Config(
    model_name="meta-llama/Llama-2-7b-hf",  # Larger model
    batch_size=8,
    num_epochs=10,
    train_size=5000,
    lora_r=16,
)
```

## 🎓 Examples to Try

```bash
# Run all examples
python examples.py

# Or individual examples
python -c "from examples import example_1_basic_training; example_1_basic_training()"
python -c "from examples import example_2_sampling_methods; example_2_sampling_methods()"
python -c "from examples import example_3_scaling; example_3_scaling()"
```

## 📝 Custom JSSP Format

```python
# Define custom problem
problem = JSSproblem(num_jobs=2, num_machines=2)

# Override with custom jobs
# Format: [(machine_id, duration), ...]
problem.jobs = [
    [(0, 5), (1, 3), (0, 2)],    # Job 0: 5h on M0, 3h on M1, 2h on M0
    [(1, 4), (0, 2), (1, 3)],    # Job 1: 4h on M1, 2h on M0, 3h on M1
]

print(problem.to_text())
```

## 🌐 Using Different Models

```python
from jssp_scheduler_hf import Config, JSSpTrainer

# GPT-2 family
Config(model_name="gpt2")
Config(model_name="gpt2-medium")
Config(model_name="gpt2-large")

# Open source alternatives
Config(model_name="EleutherAI/gpt-neo-125M")
Config(model_name="EleutherAI/gpt-j-6B")

# Llama
Config(model_name="meta-llama/Llama-2-7b-hf")
Config(model_name="meta-llama/Llama-2-13b-hf")

# Mistral
Config(model_name="mistralai/Mistral-7B-v0.1")
```

## 🔗 Key Files

| File | Purpose | When to Use |
|------|---------|------------|
| `jssp_llm_scheduler.py` | Standalone implementation | Learning, quick prototyping |
| `jssp_scheduler_hf.py` | Production with HuggingFace | Real use, large models |
| `examples.py` | 6 runnable examples | Learn by doing |
| `README.md` | Full documentation | Reference |
| `SUMMARY.md` | Complete overview | Big picture |

## 📊 Typical Training Time

| Model | Dataset | Batch | Epochs | Time (GPU) | Time (CPU) |
|-------|---------|-------|--------|-----------|-----------|
| GPT-2 | 100 | 2 | 1 | 30s | 2m |
| GPT-2 | 500 | 4 | 3 | 2m | 15m |
| Llama-2-7B | 500 | 4 | 3 | 10m | ~1h |

## 💡 Pro Tips

1. **Start small**: Use GPT-2 with small problems first
2. **Validate early**: Check solution quality after each epoch
3. **Experiment**: Try different sampling methods for same problem
4. **Scale gradually**: Increase problem size incrementally
5. **Use baselines**: Always compare against greedy heuristic
6. **Monitor memory**: Watch VRAM usage during training
7. **Save checkpoints**: Use `save_steps` in config
8. **Batch inference**: Process multiple problems together

## 🔗 Important Links

- **Paper**: https://arxiv.org/abs/2408.06993
- **HuggingFace**: https://huggingface.co
- **PEFT**: https://github.com/huggingface/peft
- **PyTorch**: https://pytorch.org

---

**Need help?** Check README.md for detailed documentation!
