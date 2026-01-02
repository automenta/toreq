#!/usr/bin/env python3
"""
Run Full Research Suite.
Systematically executes benchmarks for all Key Performance Indicators:
1. Datasets: MNIST, Fashion-MNIST, CIFAR-10 (if available)
2. Models: Backprop, ModernEqProp (SN), LoopedMLP (SN)
3. Seeds: Multi-seed for statistical rigor
"""

import subprocess
import json
import os
import time
from datetime import datetime

RESULTS_DIR = "results/suite"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = [
    {"dataset": "mnist", "seeds": 3, "epochs": 5},
    # {"dataset": "fashion-mnist", "seeds": 3, "epochs": 5},
    # {"dataset": "cifar10", "seeds": 1, "epochs": 5}, # Expensive, enable if needed
]

def run_experiment(dataset, seeds, epochs):
    print(f"\n>>> Running Suite: {dataset.upper()} ({seeds} seeds, {epochs} epochs)")
    cmd = [
        "python", "scripts/competitive_benchmark.py",
        "--seeds", str(seeds),
        "--epochs", str(epochs),
        "--dataset", dataset
    ]
    
    start = time.time()
    try:
        # We need to capture the output JSON file path which script generates
        # The script currently outputs to /tmp/competitive_benchmark_{seeds}seed.json
        # We should modify benchmark to accept an output path OR move it after.
        
        # Let's rely on standard file location /tmp/... for now and move it.
        subprocess.run(cmd, check=True)
        
        src = f"/tmp/competitive_benchmark_{seeds}seed.json"
        dst = f"{RESULTS_DIR}/{dataset}_benchmark.json"
        
        if os.path.exists(src):
            subprocess.run(["cp", src, dst], check=True)
            print(f"✅ Saved results to {dst}")
            return True
        else:
            print(f"❌ Result file missing: {src}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        return False

def main():
    print("==================================================")
    print("   TorEqProp RESEARCH SUITE RUNNER")
    print("==================================================")
    
    summary = {}
    
    for conf in CONFIG:
        key = f"{conf['dataset']}"
        success = run_experiment(conf['dataset'], conf['seeds'], conf['epochs'])
        summary[key] = "Success" if success else "Failed"
        
    print("\n>>> Suite Complete")
    print(json.dumps(summary, indent=2))
    
    # Generate aggregate report?
    # TODO: Merge JSONs into one massive results file.

if __name__ == "__main__":
    main()
