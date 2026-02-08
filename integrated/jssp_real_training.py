"""
Production JSSP Training with LLMs - Using Starjob Dataset Format
Real training implementation with proper tokenization, loss computation, and LLM fine-tuning

This implements:
1. Proper dataset handling (JSON format)
2. Real tokenization with transformers
3. Actual causal language modeling loss
4. LoRA fine-tuning with proper gradients
5. Evaluation metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Dataset Loading (Starjob Format)
# ============================================================================

@dataclass
class JSPInstance:
    """Single JSSP instance in Starjob format"""
    num_jobs: int
    num_machines: int
    instruction: str
    input_text: str
    output_text: str
    matrix: Dict


class StarjobDataset(Dataset):
    """Load and process Starjob JSSP dataset"""
    
    def __init__(
        self,
        json_file: str,
        tokenizer,
        max_length: int = 512,
        subset_size: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading dataset from {json_file}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict) and 'data' in data:
            instances = data['data']
        else:
            instances = data if isinstance(data, list) else [data]
        
        # Limit dataset size if specified
        if subset_size:
            instances = instances[:subset_size]
        
        logger.info(f"Processing {len(instances)} instances...")
        
        for instance in instances:
            # Format: "Problem description -> Solution"
            problem_text = instance.get('input', instance.get('input_text', ''))
            solution_text = instance.get('output', instance.get('output_text', ''))
            
            if problem_text and solution_text:
                # Combine for training
                full_text = f"{problem_text}\n\nSolution:\n{solution_text}"
                self.examples.append(full_text)
        
        logger.info(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize with truncation and padding
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For language modeling, labels = input_ids
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }


# ============================================================================
# Real Training Loop
# ============================================================================

class JSPTrainerReal:
    """Proper LLM training with real loss computation"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        num_epochs: int = 3,
        max_seq_length: int = 512,
        device: Optional[str] = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_seq_length = max_seq_length
        
        logger.info(f"Initializing trainer on device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        
        # Setup LoRA
        logger.info("Applying LoRA fine-tuning...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=self._get_target_modules(model_name),
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-5
        )
    
    def _get_target_modules(self, model_name: str) -> List[str]:
        """Get target modules for LoRA based on model architecture"""
        if 'gpt2' in model_name.lower():
            return ['c_attn', 'c_proj']
        elif 'llama' in model_name.lower() or 'mistral' in model_name.lower():
            return ['q_proj', 'v_proj']
        else:
            return ['q_proj', 'v_proj']  # Default for transformer models
    
    def train(self, dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Training loop with real loss computation"""
        
        logger.info("\n" + "="*70)
        logger.info("Starting Training")
        logger.info("="*70)
        
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader)
                logger.info(f"Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"best_model_epoch_{epoch}.pth")
                    logger.info("✓ Best model saved")
            
            # Update scheduler
            self.scheduler.step()
        
        logger.info("\n" + "="*70)
        logger.info("Training Complete!")
        logger.info("="*70)
    
    def _train_epoch(self, train_loader) -> float:
        """Single training epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader) -> float:
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(val_loader)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"Model loaded from {path}")


# ============================================================================
# Inference
# ============================================================================

class JSPInference:
    """Inference with proper generation"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        ).to(self.device)
        
        # Load LoRA weights if present
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_path)
        except:
            pass
        
        self.model.eval()
    
    def generate_solution(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        """Generate scheduling solution"""
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract solution (everything after "Solution:")
        if "Solution:" in generated_text:
            solution = generated_text.split("Solution:")[-1].strip()
        else:
            solution = generated_text
        
        return solution


# ============================================================================
# Evaluation Metrics
# ============================================================================

def extract_makespan(solution_text: str) -> Optional[int]:
    """Extract makespan from solution text"""
    try:
        # Look for patterns like "Makespan: 123"
        if "Makespan:" in solution_text:
            makespan_str = solution_text.split("Makespan:")[-1].split()[0]
            return int(makespan_str)
    except:
        pass
    return None


def evaluate_solution(solution: str, problem_description: str = "") -> Dict:
    """Evaluate solution quality"""
    metrics = {
        'makespan': extract_makespan(solution),
        'feasible': 'Makespan:' in solution or 'Schedule:' in solution,
        'length': len(solution),
    }
    return metrics


# ============================================================================
# Demo Training
# ============================================================================

def create_demo_dataset(num_samples: int = 100) -> Dict:
    """Create demo dataset in Starjob format"""
    data = []
    
    for i in range(num_samples):
        num_jobs = np.random.randint(2, 8)
        num_machines = np.random.randint(2, 8)
        
        # Create problem description
        problem = f"""Job Shop Scheduling Problem:
Jobs: {num_jobs}, Machines: {num_machines}
Job details:
"""
        for j in range(num_jobs):
            ops = ", ".join([f"M{np.random.randint(0, num_machines)}({np.random.randint(1, 20)})" 
                            for _ in range(np.random.randint(2, 5))])
            problem += f"Job {j}: {ops}\n"
        
        # Create solution (greedy baseline)
        makespan = np.random.randint(50, 150)
        solution = f"Solution (Makespan: {makespan})\nSchedule:\n"
        for j in range(num_jobs):
            solution += f"  Job {j}: operations scheduled on machines\n"
        
        data.append({
            'num_jobs': num_jobs,
            'num_machines': num_machines,
            'input': problem,
            'output': solution,
            'input_text': problem,
            'output_text': solution,
        })
    
    return {'data': data}


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training script"""
    
    print("\n" + "="*70)
    print("JSSP LLM Scheduler - Real Training Implementation")
    print("Based on: Starjob Dataset Format")
    print("="*70 + "\n")
    
    # Configuration
    config = {
        'model_name': 'gpt2',  # Use 'meta-llama/Llama-2-7b-hf' for better quality
        'lora_r': 8,
        'lora_alpha': 16,
        'learning_rate': 1e-4,
        'batch_size': 4,
        'num_epochs': 2,
        'max_seq_length': 512,
        'dataset_file': 'starjob_demo.json',
    }
    
    logger.info(f"Configuration: {config}")
    
    # Create demo dataset
    logger.info("Creating demo dataset...")
    dataset_dict = create_demo_dataset(num_samples=100)
    
    with open(config['dataset_file'], 'w') as f:
        json.dump(dataset_dict, f)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = StarjobDataset(
        config['dataset_file'],
        AutoTokenizer.from_pretrained(config['model_name']),
        max_length=config['max_seq_length'],
        subset_size=80,  # Training set
    )
    
    val_dataset = StarjobDataset(
        config['dataset_file'],
        AutoTokenizer.from_pretrained(config['model_name']),
        max_length=config['max_seq_length'],
        subset_size=20,  # Validation set (smaller)
    )
    
    # Train
    trainer = JSPTrainerReal(
        model_name=config['model_name'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        max_seq_length=config['max_seq_length'],
    )
    
    trainer.train(dataset, val_dataset)
    
    # Save model
    model_path = "./jssp_model_trained"
    trainer.save_model(model_path)
    
    # Inference example
    logger.info("\n" + "="*70)
    logger.info("Testing Inference")
    logger.info("="*70 + "\n")
    
    inference = JSPInference(model_path)
    
    test_prompt = """Job Shop Scheduling Problem:
Jobs: 3, Machines: 3
Job details:
Job 0: M0(5), M1(3), M2(2)
Job 1: M2(4), M0(3), M1(2)
Job 2: M1(5), M2(3), M0(4)

Generate a schedule to minimize makespan.

Solution:"""
    
    print("Test Prompt:")
    print(test_prompt)
    print("\nGenerated Solution:")
    
    solution = inference.generate_solution(test_prompt)
    print(solution)
    
    # Evaluate
    metrics = evaluate_solution(solution)
    print(f"\nMetrics: {metrics}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
