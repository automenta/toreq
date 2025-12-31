import argparse
import sys
import torch
from src.models import LoopedMLP, ToroidalMLP
from src.training import EqPropTrainer, get_mnist_loaders
from hyperopt import run_study

def smoke_test():
    print("Running Smoke Test...")
    model = LoopedMLP(784, 256, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = EqPropTrainer(model, optimizer)
    
    # Fake data
    x = torch.randn(4, 784)
    y = torch.tensor([0, 1, 2, 3])
    
    try:
        metrics = trainer.step(x, y)
        print(f"Smoke Test Passed! Metrics: {metrics}")
    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        sys.exit(1)

def campaign(time_budget, epochs, dataset_size):
    print(f"Starting Comparison Campaign ({epochs} epochs, Budget: {time_budget}s, Data: {dataset_size})...")
    
    results = {}
    
    # 1. Backprop Baseline
    print("\n--- optimizing BackpropMLP ---")
    bp_stats = run_study("bp_study", "BackpropMLP", n_trials=50, time_budget=time_budget, epochs=epochs, dataset_size=dataset_size)
    results["BackpropMLP"] = bp_stats
    
    # 2. LoopedMLP (EqProp Baseline)
    print("\n--- optimizing LoopedMLP ---")
    looped_stats = run_study("looped_study", "LoopedMLP", n_trials=50, time_budget=time_budget, epochs=epochs, dataset_size=dataset_size)
    results["LoopedMLP"] = looped_stats
    
    # 3. ToroidalMLP (TEP)
    print("\n--- optimizing ToroidalMLP ---")
    toroidal_stats = run_study("toroidal_study", "ToroidalMLP", n_trials=50, time_budget=time_budget, epochs=epochs, dataset_size=dataset_size)
    results["ToroidalMLP"] = toroidal_stats
    
    print("\n\n=== CAMPAIGN RESULTS ===")
    print(f"{'Model':<15} | {'Best Acc':<10} | {'Time/Trial':<10} | {'Params':<8}")
    print("-" * 55)
    for model, (score, time_avg, params) in results.items():
        print(f"{model:<15} | {score:.4f}     | {time_avg:.2f}s      | {params:<8}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorEqProp Research CLI")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick verification")
    parser.add_argument("--campaign", action="store_true", help="Run full comparison")
    parser.add_argument("--time-budget", type=int, default=60, help="Time in seconds per model for campaign")
    
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument("--dataset-size", type=int, default=1000, help="Training set size (max 60000)")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        smoke_test()
    elif args.campaign:
        campaign(args.time_budget, args.epochs, args.dataset_size)
    else:
        parser.print_help()
