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

def campaign(time_budget):
    print(f"Starting Comparison Campaign (Budget: {time_budget}s per model)...")
    
    results = {}
    
    # 1. Backprop Baseline
    print("\n--- optimizing BackpropMLP ---")
    bp_score = run_study("bp_study", "BackpropMLP", n_trials=50, time_budget=time_budget)
    results["BackpropMLP"] = bp_score
    
    # 2. LoopedMLP (EqProp Baseline)
    print("\n--- optimizing LoopedMLP ---")
    looped_score = run_study("looped_study", "LoopedMLP", n_trials=50, time_budget=time_budget)
    results["LoopedMLP"] = looped_score
    
    # 3. ToroidalMLP (TEP)
    print("\n--- optimizing ToroidalMLP ---")
    toroidal_score = run_study("toroidal_study", "ToroidalMLP", n_trials=50, time_budget=time_budget)
    results["ToroidalMLP"] = toroidal_score
    
    print("\n\n=== CAMPAIGN RESULTS ===")
    print(f"{'Model':<15} | {'Best Accuracy':<15}")
    print("-" * 33)
    for model, score in results.items():
        print(f"{model:<15} | {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorEqProp Research CLI")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick verification")
    parser.add_argument("--campaign", action="store_true", help="Run full comparison")
    parser.add_argument("--time-budget", type=int, default=60, help="Time in seconds per model for campaign")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        smoke_test()
    elif args.campaign:
        campaign(args.time_budget)
    else:
        parser.print_help()
