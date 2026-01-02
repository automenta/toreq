#!/usr/bin/env python3
"""
EqProp Multi-Task Benchmark

Reproduces the main results from the paper.
Run: python benchmark.py --seeds 3
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from models import BackpropMLP, LoopedMLP
from trainer import EqPropTrainer, evaluate
from tasks import get_task


# Optimized hyperparameters per task (from grid search)
TASK_CONFIGS = {
    'digits':  {'epochs': 30, 'lr': 0.001, 'beta': 0.22, 'steps': 30, 'hidden': 128},
    'mnist':   {'epochs': 20, 'lr': 0.002, 'beta': 0.22, 'steps': 30, 'hidden': 256},
    'fashion': {'epochs': 20, 'lr': 0.002, 'beta': 0.50, 'steps': 30, 'hidden': 256},
    'cartpole':{'epochs': 30, 'lr': 0.001, 'beta': 0.22, 'steps': 30, 'hidden': 64},
    'acrobot': {'epochs': 30, 'lr': 0.002, 'beta': 0.50, 'steps': 30, 'hidden': 64},
}


def run_experiment(task_name, seeds=3, smoke_test=False, device='cpu'):
    """Run full experiment for one task."""
    cfg = TASK_CONFIGS[task_name]
    
    if smoke_test:
        cfg = {**cfg, 'epochs': 1}
    
    print(f"\n{'='*60}")
    print(f"Task: {task_name.upper()}")
    print(f"{'='*60}")
    
    results = {'backprop': [], 'eqprop': []}
    
    for seed in range(seeds):
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)
        
        # Load data
        train_loader, test_loader, in_dim, out_dim = get_task(task_name)
        
        # --- Backprop baseline ---
        bp_model = BackpropMLP(in_dim, cfg['hidden'], out_dim).to(device)
        bp_opt = optim.Adam(bp_model.parameters(), lr=cfg['lr'])
        bp_loss = nn.CrossEntropyLoss()
        
        start = time.time()
        for _ in range(cfg['epochs']):
            bp_model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                bp_opt.zero_grad()
                bp_loss(bp_model(x), y).backward()
                bp_opt.step()
        
        bp_acc = evaluate(bp_model, test_loader, device)
        bp_time = time.time() - start
        results['backprop'].append(bp_acc)
        print(f"  Seed {seed}: Backprop = {bp_acc:.2f}% ({bp_time:.1f}s)")
        
        # --- EqProp ---
        eq_model = LoopedMLP(in_dim, cfg['hidden'], out_dim, use_spectral_norm=True).to(device)
        eq_opt = optim.Adam(eq_model.parameters(), lr=cfg['lr'])
        trainer = EqPropTrainer(eq_model, eq_opt, beta=cfg['beta'], max_steps=cfg['steps'])
        
        start = time.time()
        for _ in range(cfg['epochs']):
            eq_model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                trainer.step(x, y)
        
        eq_acc = evaluate(eq_model, test_loader, device, steps=cfg['steps'])
        eq_time = time.time() - start
        results['eqprop'].append(eq_acc)
        print(f"          EqProp   = {eq_acc:.2f}% ({eq_time:.1f}s)")
    
    # Aggregate
    bp_mean, bp_std = np.mean(results['backprop']), np.std(results['backprop'])
    eq_mean, eq_std = np.mean(results['eqprop']), np.std(results['eqprop'])
    gap = eq_mean - bp_mean
    
    print(f"\n  Summary:")
    print(f"    Backprop: {bp_mean:.2f}% ± {bp_std:.2f}%")
    print(f"    EqProp:   {eq_mean:.2f}% ± {eq_std:.2f}%")
    print(f"    Gap:      {gap:+.2f}%")
    
    return {
        'backprop': {'mean': bp_mean, 'std': bp_std, 'seeds': results['backprop']},
        'eqprop': {'mean': eq_mean, 'std': eq_std, 'seeds': results['eqprop']},
        'gap': gap
    }


def main():
    parser = argparse.ArgumentParser(description='EqProp Multi-Task Benchmark')
    parser.add_argument('--seeds', type=int, default=3, help='Seeds per task')
    parser.add_argument('--tasks', type=str, default='all', 
                       help='Comma-separated tasks or "all"')
    parser.add_argument('--smoke-test', action='store_true', 
                       help='Quick test (1 epoch)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    tasks = list(TASK_CONFIGS.keys()) if args.tasks == 'all' else args.tasks.split(',')
    
    all_results = {}
    for task in tasks:
        all_results[task] = run_experiment(task, args.seeds, args.smoke_test, device)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Task':<15} {'Backprop':>12} {'EqProp':>12} {'Gap':>8}")
    print("-"*50)
    
    for task, res in all_results.items():
        bp = f"{res['backprop']['mean']:.1f}%"
        eq = f"{res['eqprop']['mean']:.1f}%"
        gap = f"{res['gap']:+.1f}%"
        print(f"{task:<15} {bp:>12} {eq:>12} {gap:>8}")
    
    avg_gap = np.mean([r['gap'] for r in all_results.values()])
    print("-"*50)
    print(f"{'Average Gap:':<15} {'':<12} {'':<12} {avg_gap:+.1f}%")
    
    # Save results
    Path('../results').mkdir(exist_ok=True)
    with open('../results/benchmark.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/benchmark.json")


if __name__ == '__main__':
    main()
