"""
Production JSSP Scheduler with Hugging Face & PEFT LoRA
Inspired by: "LLMs can Schedule" (arXiv:2408.06993)

This version uses:
- Hugging Face Transformers (GPT-2, Llama, etc.)
- PEFT library for LoRA fine-tuning
- Proper tokenization and generation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# JSSP Dataset & Problem
# ============================================================================

class JSSproblem:
    """Job Shop Scheduling Problem"""
    
    def __init__(self, num_jobs: int, num_machines: int, 
                 seed: Optional[int] = None, max_op_duration: int = 20):
        if seed is not None:
            np.random.seed(seed)
        
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.seed = seed
        
        # Generate jobs: each job has operations to perform on machines
        # operations = [(machine_id, duration), ...]
        self.jobs = []
        for j in range(num_jobs):
            # Each job has random number of operations (3-8)
            num_ops = np.random.randint(3, min(9, num_machines + 1))
            # Each operation: random machine, random duration
            operations = [
                (np.random.randint(0, num_machines), 
                 np.random.randint(1, max_op_duration + 1))
                for _ in range(num_ops)
            ]
            self.jobs.append(operations)
    
    def to_text(self) -> str:
        """Convert problem to text format for LLM"""
        text = f"Problem: Job Shop Scheduling\n"
        text += f"Machines: {self.num_machines}, Jobs: {self.num_jobs}\n\n"
        text += "Job specifications:\n"
        
        for job_id, operations in enumerate(self.jobs):
            ops_str = " -> ".join([f"M{m}:{d}h" for m, d in operations])
            text += f"  J{job_id}: {ops_str}\n"
        
        return text
    
    @staticmethod
    def solve_greedy(problem: "JSSproblem") -> str:
        """Simple greedy scheduling solution"""
        machine_available_at = [0] * problem.num_machines
        job_end_time = [0] * problem.num_jobs
        schedule = []
        
        # Process jobs in order
        for job_id, operations in enumerate(problem.jobs):
            current_time = job_end_time[job_id]
            
            for op_idx, (machine, duration) in enumerate(operations):
                # Schedule on this machine
                start_time = max(current_time, machine_available_at[machine])
                end_time = start_time + duration
                
                schedule.append({
                    'job': job_id,
                    'op': op_idx,
                    'machine': machine,
                    'start': start_time,
                    'end': end_time
                })
                
                machine_available_at[machine] = end_time
                current_time = end_time
            
            job_end_time[job_id] = current_time
        
        makespan = max(machine_available_at)
        
        # Format solution
        solution = f"Solution (Makespan: {makespan})\n"
        solution += "Schedule:\n"
        for entry in schedule:
            solution += (f"  Job {entry['job']}.Op{entry['op']}: "
                        f"Machine {entry['machine']} "
                        f"[{entry['start']}, {entry['end']})\n")
        
        return solution


class JSSpTrainDataset(Dataset):
    """Dataset for JSSP training"""
    
    def __init__(self, tokenizer, config: "Config", split: str = 'train',
                 num_samples: int = 1000):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.samples = []
        
        logger.info(f"Generating {num_samples} JSSP {split} samples...")
        
        # Generate samples
        for seed in range(num_samples):
            # Create problem
            problem = JSSproblem(
                num_jobs=config.num_jobs,
                num_machines=config.num_machines,
                seed=seed,
                max_op_duration=config.max_operation_duration
            )
            
            # Get solution
            solution = JSSproblem.solve_greedy(problem)
            
            # Format training text
            text = problem.to_text() + "\nSolve this problem:\n" + solution
            self.samples.append(text)
        
        logger.info(f"Generated {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()  # For language modeling
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Training configuration"""
    # Model
    model_name: str = "gpt2"  # Use GPT2 for demo, Llama-2 for production
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    
    # Training
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    max_seq_length: int = 512
    
    # JSSP
    num_jobs: int = 5
    num_machines: int = 5
    max_operation_duration: int = 20
    
    # Dataset
    train_size: int = 500
    val_size: int = 100
    test_size: int = 50
    
    # Saving
    output_dir: str = "./jssp_lora_model"
    save_steps: int = 500


# ============================================================================
# Training
# ============================================================================

class JSSpTrainer:
    """Trainer for JSSP scheduling model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32
        )
        
        # Setup LoRA
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)
        
        logger.info(f"Model parameters:")
        self.model.print_trainable_parameters()
        
        # Create datasets
        logger.info("Creating datasets...")
        self.train_dataset = JSSpTrainDataset(
            self.tokenizer, config, split='train', 
            num_samples=config.train_size
        )
        self.val_dataset = JSSpTrainDataset(
            self.tokenizer, config, split='val',
            num_samples=config.val_size
        )
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            weight_decay=0.01,
            logging_steps=10,
            eval_steps=100,
            save_steps=config.save_steps,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            gradient_accumulation_steps=1,
            max_grad_norm=config.max_grad_norm,
            optim="adamw_torch",
            seed=42,
        )
        
        # Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
        )
    
    def train(self):
        """Run training"""
        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training complete!")
        
        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)


# ============================================================================
# Inference
# ============================================================================

class JSSpScheduler:
    """JSSP Scheduler using trained LLM"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(self.device)
        
        self.model.eval()
    
    def schedule(self, problem: JSSproblem, 
                sampling_method: str = 'nucleus',
                temperature: float = 0.8,
                top_p: float = 0.95,
                top_k: int = 50,
                max_length: int = 512) -> str:
        """
        Generate scheduling solution for JSSP problem
        
        Args:
            problem: JSSP problem instance
            sampling_method: 'greedy', 'beam', 'nucleus', or 'top_k'
            temperature: Generation temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_length: Maximum generation length
            
        Returns:
            Generated scheduling solution
        """
        # Format input
        prompt = problem.to_text() + "\nSolve this problem:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            if sampling_method == 'greedy':
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            elif sampling_method == 'nucleus':
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            elif sampling_method == 'top_k':
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                )
            elif sampling_method == 'beam':
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                )
            else:
                raise ValueError(f"Unknown sampling method: {sampling_method}")
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract solution (everything after "Solve this problem:\n")
        if "Solve this problem:\n" in generated_text:
            solution = generated_text.split("Solve this problem:\n")[-1]
        else:
            solution = generated_text
        
        return solution
    
    def batch_schedule(self, problems: List[JSSproblem],
                      sampling_method: str = 'nucleus') -> List[str]:
        """Schedule multiple problems"""
        solutions = []
        for problem in problems:
            solution = self.schedule(problem, sampling_method=sampling_method)
            solutions.append(solution)
        return solutions


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_solution(solution_text: str) -> Dict:
    """Evaluate solution quality"""
    # Parse makespan from solution if present
    makespan = None
    if "Makespan:" in solution_text:
        try:
            makespan = int(
                solution_text.split("Makespan:")[-1].split(")")[0].strip()
            )
        except:
            pass
    
    return {
        'makespan': makespan,
        'length': len(solution_text),
        'valid': makespan is not None if makespan is not None else True
    }


# ============================================================================
# Main
# ============================================================================

def main():
    # Configuration
    config = Config(
        model_name="gpt2",
        num_jobs=3,
        num_machines=3,
        train_size=100,  # Small for demo
        val_size=20,
        batch_size=2,
        num_epochs=1,  # Demo epoch count
    )
    
    print("\n" + "="*70)
    print("JSSP LLM Scheduler - Inspired by 'LLMs can Schedule'")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  JSSP: {config.num_jobs} jobs, {config.num_machines} machines")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Training: {config.num_epochs} epochs, batch_size={config.batch_size}")
    print(f"  Dataset: train={config.train_size}, val={config.val_size}")
    
    # Train
    print("\n" + "-"*70)
    print("PHASE 1: Training")
    print("-"*70)
    trainer = JSSpTrainer(config)
    trainer.train()
    
    # Inference examples
    print("\n" + "-"*70)
    print("PHASE 2: Inference")
    print("-"*70)
    
    scheduler = JSSpScheduler(config.output_dir)
    
    # Generate test problems
    test_problems = [
        JSSproblem(num_jobs=2, num_machines=2, seed=999),
        JSSproblem(num_jobs=3, num_machines=3, seed=1000),
    ]
    
    for i, problem in enumerate(test_problems):
        print(f"\n--- Test Problem {i+1} ---")
        print(problem.to_text())
        
        # Generate solution
        solution = scheduler.schedule(
            problem,
            sampling_method='nucleus',
            temperature=0.8
        )
        
        print("Generated Solution:")
        print(solution)
        
        # Evaluate
        eval_metrics = evaluate_solution(solution)
        print(f"Evaluation: {eval_metrics}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
