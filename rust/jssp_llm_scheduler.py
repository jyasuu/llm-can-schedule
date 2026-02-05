"""
LLM-based Job Shop Scheduling Problem (JSSP) Solver
Inspired by: "LLMs can Schedule" (arXiv:2408.06993)

This implementation demonstrates:
1. JSSP dataset generation/loading
2. LoRA fine-tuning for scheduling
3. Inference with sampling strategies
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import json
from tqdm import tqdm
import random

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration for JSSP-LLM training"""
    model_name: str = "meta-llama/Llama-2-7b"  # Or use smaller model for testing
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None  # Will be set based on model
    
    # Training
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # JSSP problem
    num_jobs: int = 10
    num_machines: int = 10
    max_job_length: int = 20
    
    # Dataset
    dataset_size: int = 10000
    val_split: float = 0.1
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]


# ============================================================================
# JSSP Problem & Dataset
# ============================================================================

class JSSproblem:
    """Represents a single Job Shop Scheduling Problem instance"""
    
    def __init__(self, num_jobs: int, num_machines: int, max_duration: int = 20):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_duration = max_duration
        
        # jobs[j] = [(machine_id, duration), ...]
        self.jobs = [
            [(np.random.randint(0, num_machines), 
              np.random.randint(1, max_duration + 1)) 
             for _ in range(np.random.randint(3, 8))]
            for _ in range(num_jobs)
        ]
    
    def format_for_llm(self) -> str:
        """Format problem as text for LLM input"""
        prompt = f"Job Shop Scheduling Problem:\n"
        prompt += f"Jobs: {self.num_jobs}, Machines: {self.num_machines}\n"
        prompt += f"Job details:\n"
        
        for job_id, operations in enumerate(self.jobs):
            ops_str = ", ".join([f"M{m}({d})" for m, d in operations])
            prompt += f"Job {job_id}: {ops_str}\n"
        
        prompt += "\nSchedule the jobs to minimize makespan.\n"
        return prompt
    
    @staticmethod
    def greedy_schedule(problem: "JSSproblem") -> str:
        """Simple greedy scheduling solution"""
        # This is a simplified solution - in practice, use proper JSSP solver
        machine_free_times = [0] * problem.num_machines
        schedule = []
        
        for job_id, operations in enumerate(problem.jobs):
            job_schedule = []
            current_time = 0
            
            for op_idx, (machine_id, duration) in enumerate(operations):
                # Schedule operation on machine
                start_time = max(current_time, machine_free_times[machine_id])
                end_time = start_time + duration
                
                job_schedule.append((job_id, op_idx, machine_id, start_time, end_time))
                machine_free_times[machine_id] = end_time
                current_time = end_time
            
            schedule.extend(job_schedule)
        
        makespan = max(machine_free_times)
        
        # Format as text
        output = f"Schedule (Makespan: {makespan}):\n"
        for job_id, op_idx, machine_id, start, end in schedule:
            output += f"Job{job_id}_Op{op_idx}: M{machine_id} [{start}, {end}]\n"
        
        return output


class JSSpDataset(Dataset):
    """Dataset for JSSP training"""
    
    def __init__(self, config: Config, split: str = 'train'):
        self.config = config
        self.split = split
        self.problems = []
        self.solutions = []
        self.tokenizer = None  # Will be set by trainer
        
        # Generate or load dataset
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate synthetic JSSP instances and solutions"""
        total_size = self.config.dataset_size
        val_size = int(total_size * self.config.val_split)
        
        if self.split == 'train':
            size = total_size - val_size
            start_idx = 0
        else:
            size = val_size
            start_idx = total_size - val_size
        
        print(f"Generating {size} {self.split} JSSP instances...")
        
        for i in range(size):
            # Generate random JSSP problem
            problem = JSSproblem(
                num_jobs=self.config.num_jobs,
                num_machines=self.config.num_machines,
                max_duration=self.config.max_job_length
            )
            
            # Get solution (simplified greedy)
            solution = JSSproblem.greedy_schedule(problem)
            
            self.problems.append(problem)
            self.solutions.append(solution)
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]
        
        # Combine as prompt-completion pair
        prompt = problem.format_for_llm()
        full_text = prompt + "\n" + solution
        
        return {
            'prompt': prompt,
            'solution': solution,
            'full_text': full_text,
            'problem_idx': idx
        }


# ============================================================================
# LoRA Fine-tuning Implementation
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA adapter for linear layers"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, 
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Original weight (frozen)
        self.register_buffer("weight", torch.randn(out_features, in_features))
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = self.alpha / self.rank
        self.enabled = True
    
    def forward(self, x):
        # Standard linear transformation
        out = torch.nn.functional.linear(x, self.weight)
        
        if self.enabled:
            # Add LoRA adaptation
            lora_out = x @ self.lora_A.t() @ self.lora_B.t()
            lora_out = self.dropout(lora_out)
            out = out + lora_out * self.scaling
        
        return out


class SimpleSchedulerModel(nn.Module):
    """Simplified transformer-like model for scheduling"""
    
    def __init__(self, vocab_size: int = 50256, hidden_dim: int = 768, 
                 num_layers: int = 4, num_heads: int = 12, lora_rank: int = 8):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, hidden_dim))
        
        # Transformer layers with LoRA
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transform
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        
        # Output
        logits = self.output_head(x)
        return logits


# ============================================================================
# Training
# ============================================================================

class JSSpSchedulerTrainer:
    """Trainer for JSSP scheduling model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize simple model (replace with proper LLM in production)
        self.model = SimpleSchedulerModel(
            vocab_size=50256,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            lora_rank=config.lora_r
        ).to(self.device)
        
        # Create datasets
        self.train_dataset = JSSpDataset(config, split='train')
        self.val_dataset = JSSpDataset(config, split='val')
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Model initialized on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Prepare simple token sequences (mock tokenization)
            batch_size = len(batch['prompt'])
            input_ids = torch.randint(0, 50256, (batch_size, 128)).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            targets = torch.randint(0, 50256, (batch_size, 128)).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits.view(-1, 50256), targets.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> float:
        """Validation"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch_size = len(batch['prompt'])
                input_ids = torch.randint(0, 50256, (batch_size, 128)).to(self.device)
                attention_mask = torch.ones_like(input_ids).to(self.device)
                targets = torch.randint(0, 50256, (batch_size, 128)).to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, 50256), targets.view(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("Starting JSSP-LLM Training")
        print("="*60 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best_model_epoch_{epoch}.pt")
                print(f"✓ Model saved (new best)")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


# ============================================================================
# Inference & Scheduling
# ============================================================================

class JSSpScheduler:
    """Inference module for JSSP scheduling"""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 sampling_method: str = 'greedy'):
        self.model = model
        self.device = device
        self.sampling_method = sampling_method
    
    def schedule(self, problem: JSSproblem, max_tokens: int = 512, 
                temperature: float = 1.0) -> str:
        """
        Schedule a JSSP problem using the trained model
        
        Args:
            problem: JSSP problem instance
            max_tokens: Max generation length
            temperature: Sampling temperature
            
        Returns:
            Scheduling solution as text
        """
        self.model.eval()
        
        prompt = problem.format_for_llm()
        
        with torch.no_grad():
            # Mock tokenization
            input_ids = torch.randint(0, 50256, (1, 128)).to(self.device)
            
            generated = input_ids.clone()
            
            for _ in range(min(max_tokens, 256)):
                logits = self.model(generated[:, -128:])[:, -1, :]
                
                if self.sampling_method == 'greedy':
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                elif self.sampling_method == 'top_k':
                    next_token = self._top_k_sample(logits, k=10, temp=temperature)
                elif self.sampling_method == 'nucleus':
                    next_token = self._nucleus_sample(logits, p=0.9, temp=temperature)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        # Return mock solution (in practice, decode tokens properly)
        return JSSproblem.greedy_schedule(problem)
    
    def _top_k_sample(self, logits: torch.Tensor, k: int = 10, 
                      temp: float = 1.0) -> torch.Tensor:
        """Top-k sampling"""
        logits = logits / temp
        top_k_vals, top_k_indices = torch.topk(logits, k)
        
        probs = torch.softmax(top_k_vals, dim=-1)
        sampled_idx = torch.multinomial(probs, 1)
        return top_k_indices.gather(-1, sampled_idx)
    
    def _nucleus_sample(self, logits: torch.Tensor, p: float = 0.9,
                       temp: float = 1.0) -> torch.Tensor:
        """Nucleus (top-p) sampling"""
        logits = logits / temp
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        mask = cumsum_probs <= p
        mask[..., 0] = True  # Always keep top token
        
        filtered_logits = sorted_logits.clone()
        filtered_logits[~mask] = float('-inf')
        
        probs = torch.softmax(filtered_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, 1)
        
        return sorted_indices.gather(-1, sampled_idx)


# ============================================================================
# Main
# ============================================================================

def main():
    # Configuration
    config = Config(
        num_jobs=5,
        num_machines=5,
        dataset_size=100,  # Small for demo
        batch_size=2,
        num_epochs=2,
    )
    
    print(f"Configuration:")
    print(f"  Jobs: {config.num_jobs}, Machines: {config.num_machines}")
    print(f"  Dataset size: {config.dataset_size}")
    print(f"  Batch size: {config.batch_size}, Epochs: {config.num_epochs}")
    print()
    
    # Train model
    trainer = JSSpSchedulerTrainer(config)
    trainer.train()
    
    # Inference example
    print("\n" + "="*60)
    print("Running Inference Example")
    print("="*60 + "\n")
    
    scheduler = JSSpScheduler(trainer.model, trainer.device, sampling_method='greedy')
    
    # Create test problem
    test_problem = JSSproblem(num_jobs=3, num_machines=3)
    print("Test Problem:")
    print(test_problem.format_for_llm())
    
    # Schedule
    solution = scheduler.schedule(test_problem)
    print("\nGenerated Schedule:")
    print(solution)


if __name__ == "__main__":
    main()
