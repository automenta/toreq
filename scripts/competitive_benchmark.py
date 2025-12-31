#!/usr/bin/env python3
"""Run competitive benchmark: EqProp models vs Backprop baseline.

Uses insights from analysis:
- Spectral norm enabled for stability
- Optimal beta = 0.22
- max_steps = 25
- Train for sufficient epochs to show convergence
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.optim as optim
import time
import json

from src.models import LoopedMLP, ToroidalMLP, ModernEqProp, BackpropMLP
from src.training import EqPropTrainer
from src.tasks import get_task_loader


def train_eqprop(model, train_loader, test_loader, epochs=50, lr=0.001, beta=0.22, max_steps=25):
    """Train EqProp model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = EqPropTrainer(model, optimizer, beta=beta, max_steps=max_steps)
    
    history = {'train_loss': [], 'test_acc': [], 'convergence_rate': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_converged = 0
        total_batches = 0
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            metrics = trainer.step(x, y)
            total_loss += metrics['loss']
            total_converged += int(metrics.get('converged_free', False))
            total_batches += 1
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x, steps=max_steps)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100. * correct / total
        conv_rate = total_converged / total_batches
        
        history['train_loss'].append(total_loss / total_batches)
        history['test_acc'].append(acc)
        history['convergence_rate'].append(conv_rate)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={total_loss/total_batches:.4f}, "
                  f"Acc={acc:.2f}%, Conv={conv_rate:.1%}")
    
    elapsed = time.time() - start_time
    return history, elapsed


def train_backprop(model, train_loader, test_loader, epochs=50, lr=0.001):
    """Train Backprop baseline."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_acc': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100. * correct / total
        history['train_loss'].append(total_loss / total_batches)
        history['test_acc'].append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={total_loss/total_batches:.4f}, Acc={acc:.2f}%")
    
    elapsed = time.time() - start_time
    return history, elapsed


def main():
    print("=" * 70)
    print("COMPETITIVE BENCHMARK: EqProp vs Backprop")
    print("=" * 70)
    
    # Configuration
    epochs = 50
    hidden_dim = 256
    lr = 0.001
    beta = 0.22  # Optimal from analysis
    max_steps = 25  # Most models converge by step 25
    
    # Load data
    train_loader, test_loader, input_dim, output_dim = get_task_loader(
        'digits', batch_size=64, dataset_size=10000
    )
    
    results = {}
    
    # Test models
    configs = [
        ("BackpropMLP", lambda: BackpropMLP(input_dim, hidden_dim, output_dim, depth=2)),
        ("LoopedMLP (SN)", lambda: LoopedMLP(input_dim, hidden_dim, output_dim, 
                                              symmetric=True, use_spectral_norm=True)),
        ("ToroidalMLP (SN)", lambda: ToroidalMLP(input_dim, hidden_dim, output_dim, 
                                                  use_spectral_norm=True)),
        ("ModernEqProp (SN)", lambda: ModernEqProp(input_dim, hidden_dim, output_dim,
                                                    use_spectral_norm=True)),
    ]
    
    for name, model_fn in configs:
        print(f"\n## {name}")
        print("-" * 50)
        
        model = model_fn().cuda()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        if "Backprop" in name:
            history, elapsed = train_backprop(model, train_loader, test_loader, 
                                             epochs=epochs, lr=lr)
        else:
            history, elapsed = train_eqprop(model, train_loader, test_loader, 
                                           epochs=epochs, lr=lr, beta=beta, max_steps=max_steps)
        
        final_acc = history['test_acc'][-1]
        best_acc = max(history['test_acc'])
        
        results[name] = {
            'params': param_count,
            'final_acc': final_acc,
            'best_acc': best_acc,
            'time': elapsed,
            'history': history
        }
        
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Best Accuracy:  {best_acc:.2f}%")
        print(f"  Training Time:  {elapsed:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} | {'Params':>10} | {'Final Acc':>10} | {'Best Acc':>10} | {'Time':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<20} | {r['params']:>10,} | {r['final_acc']:>10.2f}% | "
              f"{r['best_acc']:>10.2f}% | {r['time']:>8.1f}s")
    
    # Save results
    with open('/tmp/competitive_benchmark.json', 'w') as f:
        # Convert history lists to serializable format
        save_results = {k: {**v, 'history': {hk: [float(x) for x in hv] 
                            for hk, hv in v['history'].items()}} 
                       for k, v in results.items()}
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to /tmp/competitive_benchmark.json")
    print("=" * 70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slow)")
    main()
