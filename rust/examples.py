"""
Example Scripts for JSSP LLM Scheduler

These examples demonstrate:
1. Basic training and inference
2. Comparing different sampling methods
3. Evaluating solution quality
4. Scaling to larger problems
5. Batch processing multiple problems
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict

# Import from implementations (adjust path as needed)
from jssp_scheduler_hf import (
    JSSproblem, JSSpTrainer, JSSpScheduler, Config,
    evaluate_solution
)


# ============================================================================
# Example 1: Basic Training and Single Inference
# ============================================================================

def example_1_basic_training():
    """Simplest training and inference example"""
    print("\n" + "="*70)
    print("Example 1: Basic Training and Single Inference")
    print("="*70)
    
    # Configuration for quick demo
    config = Config(
        model_name="gpt2",
        num_jobs=3,
        num_machines=3,
        train_size=50,  # Small for demo
        val_size=10,
        batch_size=2,
        num_epochs=1,
        output_dir="./example_1_model",
    )
    
    print(f"\nTraining configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  JSSP: {config.num_jobs}x{config.num_machines}")
    print(f"  Training: {config.num_epochs} epoch(s)")
    
    # Train
    print("\nTraining model...")
    trainer = JSSpTrainer(config)
    trainer.train()
    
    # Inference
    print("\nRunning inference...")
    scheduler = JSSpScheduler(config.output_dir)
    
    problem = JSSproblem(num_jobs=2, num_machines=2, seed=123)
    print(f"\nProblem:\n{problem.to_text()}")
    
    solution = scheduler.schedule(problem, sampling_method='greedy')
    print(f"Solution:\n{solution}")
    
    return config.output_dir


# ============================================================================
# Example 2: Comparing Sampling Methods
# ============================================================================

def example_2_sampling_methods():
    """Compare different decoding strategies"""
    print("\n" + "="*70)
    print("Example 2: Comparing Sampling Methods")
    print("="*70)
    
    config = Config(
        model_name="gpt2",
        num_jobs=4,
        num_machines=4,
        train_size=100,
        val_size=20,
        batch_size=2,
        num_epochs=1,
        output_dir="./example_2_model",
    )
    
    # Train
    print("\nTraining model...")
    trainer = JSSpTrainer(config)
    trainer.train()
    
    # Create scheduler
    scheduler = JSSpScheduler(config.output_dir)
    
    # Test problem
    problem = JSSproblem(num_jobs=3, num_machines=3, seed=456)
    print(f"\nTest Problem:\n{problem.to_text()}")
    
    # Compare sampling methods
    sampling_configs = {
        'greedy': {
            'sampling_method': 'greedy',
        },
        'top_k': {
            'sampling_method': 'top_k',
            'top_k': 50,
            'temperature': 0.8,
        },
        'nucleus (p=0.95)': {
            'sampling_method': 'nucleus',
            'top_p': 0.95,
            'temperature': 0.8,
        },
        'nucleus (p=0.9)': {
            'sampling_method': 'nucleus',
            'top_p': 0.9,
            'temperature': 0.5,
        },
        'beam (5 beams)': {
            'sampling_method': 'beam',
            'max_length': 512,
        },
    }
    
    results = {}
    
    print("\nGenerating solutions with different methods:\n")
    for method_name, kwargs in sampling_configs.items():
        print(f"  {method_name}...", end=" ", flush=True)
        
        solution = scheduler.schedule(problem, **kwargs)
        metrics = evaluate_solution(solution)
        results[method_name] = {
            'solution': solution,
            'metrics': metrics,
        }
        
        print(f"Makespan={metrics.get('makespan', 'N/A')}")
    
    # Compare results
    print("\n" + "-"*70)
    print("Comparison Summary:")
    print("-"*70)
    
    makespans = {
        name: result['metrics'].get('makespan', float('inf'))
        for name, result in results.items()
        if result['metrics'].get('makespan') is not None
    }
    
    if makespans:
        best_method = min(makespans, key=makespans.get)
        print(f"\nBest method: {best_method} (Makespan: {makespans[best_method]})")
        
        for method, makespan in sorted(makespans.items(), key=lambda x: x[1]):
            print(f"  {method}: {makespan}")


# ============================================================================
# Example 3: Scaling to Larger Problems
# ============================================================================

def example_3_scaling():
    """Train and test on different problem sizes"""
    print("\n" + "="*70)
    print("Example 3: Scaling to Larger Problems")
    print("="*70)
    
    problem_sizes = [
        (3, 3),
        (5, 5),
        (10, 10),
    ]
    
    results_by_size = {}
    
    for num_jobs, num_machines in problem_sizes:
        print(f"\n{'='*70}")
        print(f"Training for {num_jobs}x{num_machines} problems")
        print(f"{'='*70}")
        
        config = Config(
            model_name="gpt2",
            num_jobs=num_jobs,
            num_machines=num_machines,
            train_size=100,
            val_size=20,
            batch_size=2,
            num_epochs=1,
            output_dir=f"./example_3_model_{num_jobs}x{num_machines}",
        )
        
        # Train
        print(f"\nTraining model for {num_jobs}x{num_machines} problems...")
        trainer = JSSpTrainer(config)
        trainer.train()
        
        # Test
        print(f"\nTesting on {num_jobs}x{num_machines} problems...")
        scheduler = JSSpScheduler(config.output_dir)
        
        test_makespans = []
        for seed in range(3):  # Test on 3 problems
            problem = JSSproblem(num_jobs=num_jobs, num_machines=num_machines, seed=seed)
            solution = scheduler.schedule(problem, sampling_method='greedy')
            metrics = evaluate_solution(solution)
            
            if metrics.get('makespan'):
                test_makespans.append(metrics['makespan'])
        
        if test_makespans:
            avg_makespan = np.mean(test_makespans)
            std_makespan = np.std(test_makespans)
            
            results_by_size[f"{num_jobs}x{num_machines}"] = {
                'avg_makespan': float(avg_makespan),
                'std_makespan': float(std_makespan),
                'samples': test_makespans,
            }
            
            print(f"  Average Makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("Scaling Summary:")
    print("="*70)
    for size, results in results_by_size.items():
        print(f"{size}: {results['avg_makespan']:.2f} ± {results['std_makespan']:.2f}")
    
    return results_by_size


# ============================================================================
# Example 4: Batch Processing Multiple Problems
# ============================================================================

def example_4_batch_processing():
    """Process multiple problems efficiently"""
    print("\n" + "="*70)
    print("Example 4: Batch Processing Multiple Problems")
    print("="*70)
    
    config = Config(
        model_name="gpt2",
        num_jobs=5,
        num_machines=5,
        train_size=100,
        val_size=20,
        batch_size=2,
        num_epochs=1,
        output_dir="./example_4_model",
    )
    
    # Train
    print("\nTraining model...")
    trainer = JSSpTrainer(config)
    trainer.train()
    
    # Create scheduler
    scheduler = JSSpScheduler(config.output_dir)
    
    # Generate batch of problems
    print("\nGenerating batch of problems...")
    num_problems = 10
    problems = [
        JSSproblem(num_jobs=5, num_machines=5, seed=seed)
        for seed in range(num_problems)
    ]
    
    print(f"Generated {num_problems} problems")
    
    # Process batch
    print("\nProcessing batch...", flush=True)
    solutions = scheduler.batch_schedule(problems, sampling_method='greedy')
    
    # Evaluate batch
    print("\nEvaluating solutions...")
    batch_results = {
        'problems': num_problems,
        'solutions': [],
    }
    
    makespans = []
    for i, (problem, solution) in enumerate(zip(problems, solutions)):
        metrics = evaluate_solution(solution)
        batch_results['solutions'].append(metrics)
        
        if metrics.get('makespan'):
            makespans.append(metrics['makespan'])
        
        print(f"  Problem {i+1}: Makespan={metrics.get('makespan', 'N/A')}")
    
    # Summary statistics
    if makespans:
        print(f"\nBatch Statistics:")
        print(f"  Mean Makespan: {np.mean(makespans):.2f}")
        print(f"  Std Makespan: {np.std(makespans):.2f}")
        print(f"  Min Makespan: {np.min(makespans)}")
        print(f"  Max Makespan: {np.max(makespans)}")
    
    return batch_results


# ============================================================================
# Example 5: Custom Problem Data
# ============================================================================

def example_5_custom_problems():
    """Work with custom JSSP problem specifications"""
    print("\n" + "="*70)
    print("Example 5: Custom Problem Data")
    print("="*70)
    
    # Define custom problems
    custom_problems = [
        {
            'name': 'Simple 2x2',
            'jobs': [
                [(0, 5), (1, 3)],      # Job 0: 5h on M0, 3h on M1
                [(1, 4), (0, 2)],      # Job 1: 4h on M1, 2h on M0
            ]
        },
        {
            'name': 'Small 3x3',
            'jobs': [
                [(0, 3), (1, 2), (2, 1)],
                [(1, 2), (2, 3), (0, 2)],
                [(2, 4), (0, 1), (1, 2)],
            ]
        },
    ]
    
    config = Config(
        model_name="gpt2",
        train_size=100,
        val_size=20,
        batch_size=2,
        num_epochs=1,
        output_dir="./example_5_model",
    )
    
    # Train
    print("\nTraining model...")
    trainer = JSSpTrainer(config)
    trainer.train()
    
    # Test on custom problems
    print("\nTesting on custom problems:")
    print("-"*70)
    
    scheduler = JSSpScheduler(config.output_dir)
    
    for custom_prob in custom_problems:
        print(f"\n{custom_prob['name']}:")
        print(f"  Jobs: {custom_prob['jobs']}")
        
        # Create synthetic problem matching structure
        problem = JSSproblem(num_jobs=len(custom_prob['jobs']), 
                            num_machines=3, seed=999)
        
        # Solve using baseline (greedy)
        baseline = JSSproblem.solve_greedy(problem)
        baseline_metrics = evaluate_solution(baseline)
        
        # Solve using LLM
        llm_solution = scheduler.schedule(problem, sampling_method='greedy')
        llm_metrics = evaluate_solution(llm_solution)
        
        print(f"  Baseline makespan: {baseline_metrics.get('makespan', 'N/A')}")
        print(f"  LLM makespan: {llm_metrics.get('makespan', 'N/A')}")
        
        if (baseline_metrics.get('makespan') and 
            llm_metrics.get('makespan')):
            improvement = (1 - llm_metrics['makespan'] / 
                          baseline_metrics['makespan']) * 100
            print(f"  Improvement: {improvement:+.1f}%")


# ============================================================================
# Example 6: Hyperparameter Tuning
# ============================================================================

def example_6_hyperparameter_tuning():
    """Experiment with different hyperparameters"""
    print("\n" + "="*70)
    print("Example 6: Hyperparameter Tuning")
    print("="*70)
    
    hyperparameter_configs = {
        'low_lr': Config(
            model_name="gpt2",
            learning_rate=1e-5,
            num_epochs=1,
            train_size=100,
        ),
        'medium_lr': Config(
            model_name="gpt2",
            learning_rate=1e-4,
            num_epochs=1,
            train_size=100,
        ),
        'high_lr': Config(
            model_name="gpt2",
            learning_rate=5e-4,
            num_epochs=1,
            train_size=100,
        ),
    }
    
    results = {}
    
    for config_name, config in hyperparameter_configs.items():
        print(f"\n{'='*70}")
        print(f"Training with {config_name} (LR={config.learning_rate})")
        print(f"{'='*70}")
        
        config.output_dir = f"./example_6_model_{config_name}"
        
        # Train
        trainer = JSSpTrainer(config)
        trainer.train()
        
        # Evaluate
        print(f"\nEvaluating...")
        scheduler = JSSpScheduler(config.output_dir)
        
        test_makespans = []
        for seed in range(3):
            problem = JSSproblem(num_jobs=5, num_machines=5, seed=seed)
            solution = scheduler.schedule(problem, sampling_method='greedy')
            metrics = evaluate_solution(solution)
            if metrics.get('makespan'):
                test_makespans.append(metrics['makespan'])
        
        if test_makespans:
            results[config_name] = {
                'learning_rate': config.learning_rate,
                'avg_makespan': float(np.mean(test_makespans)),
                'std_makespan': float(np.std(test_makespans)),
            }
            
            print(f"  Average Makespan: {np.mean(test_makespans):.2f} "
                  f"± {np.std(test_makespans):.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("Hyperparameter Tuning Summary:")
    print("="*70)
    
    for config_name, result in sorted(
        results.items(), 
        key=lambda x: x[1]['avg_makespan']
    ):
        print(f"{config_name} (LR={result['learning_rate']}): "
              f"{result['avg_makespan']:.2f} ± {result['std_makespan']:.2f}")
    
    return results


# ============================================================================
# Main Runner
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("JSSP LLM Scheduler - Example Suite")
    print("="*70)
    
    examples = [
        ("1", "Basic Training and Inference", example_1_basic_training),
        ("2", "Comparing Sampling Methods", example_2_sampling_methods),
        ("3", "Scaling to Larger Problems", example_3_scaling),
        ("4", "Batch Processing", example_4_batch_processing),
        ("5", "Custom Problem Data", example_5_custom_problems),
        ("6", "Hyperparameter Tuning", example_6_hyperparameter_tuning),
    ]
    
    print("\nAvailable Examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    
    print("\nRunning all examples... (this will take a few minutes)")
    
    for num, name, func in examples:
        try:
            print(f"\n{'#'*70}")
            func()
            print(f"\n✓ Example {num} completed")
        except Exception as e:
            print(f"\n✗ Example {num} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
