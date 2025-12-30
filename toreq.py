#!/usr/bin/env python3
"""
TorEqProp - THE research system

One clean interface for rigorous hyperparameter optimization.
Demonstrates TorEqProp vs baselines with full transparency.

Usage:
    python toreq.py                    # Quick 3-min validation
    python toreq.py 10                 # Run 10 minutes
    python toreq.py --deep             # Deep exploration mode
"""

import argparse
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

import numpy as np

# Progress bars
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

# Rich for nice output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

from hyperopt_engine import HyperOptEngine, HyperOptTrial
from statistics import StatisticalAnalyzer

console = Console()


# =============================================================================
# TASK-APPROPRIATE CONFIGURATIONS
# =============================================================================

class TaskConfig:
    """Task-appropriate hyperparameter spaces."""
    
    TASKS = {
        # Micro tasks: minimal models
        "xor": {
            "d_model": [16, 32],
            "epochs": 10,
            "tier": "micro"
        },
        "xor3": {
            "d_model": [16, 32],
            "epochs": 10,
            "tier": "micro"
        },
        
        # Small tasks: moderate models
        "mnist": {
            "d_model": [64, 128],
            "epochs": 3,
            "tier": "small"
        },
        "fashion": {
            "d_model": [64, 128],
            "epochs": 3,
            "tier": "small"
        },
        
        # RL: depends on length
        "cartpole": {
            "d_model": [32, 64],
            "epochs": 30,
            "tier": "small"
        }
    }
    
    @staticmethod
    def get_eqprop_space(task: str, breadth: str = "normal"):
        """Get task-appropriate EqProp space."""
        config = TaskConfig.TASKS.get(task, TaskConfig.TASKS["mnist"])
        
        if breadth == "low":
            # Smoke test: minimal variation
            return {
                "beta": [0.2, 0.22],
                "damping": [0.9],
                "d_model": [config["d_model"][0]],  # Smallest size
                "lr": [1e-3],
                "attention_type": ["linear"],
                "max_iters": [20],
                "tol": [1e-4],
                "update_mode": ["mse_proxy"],
            }
        elif breadth == "high":
            # Deep exploration
            return {
                "beta": [0.15, 0.2, 0.22, 0.25],
                "damping": [0.8, 0.9],
                "d_model": config["d_model"],
                "lr": [1e-3, 2e-3],
                "attention_type": ["linear", "softmax"],
                "max_iters": [20, 40],
                "tol": [1e-4],
                "update_mode": ["mse_proxy"],
            }
        else:
            # Normal
            return {
                "beta": [0.2, 0.22],
                "damping": [0.9],
                "d_model": config["d_model"],
                "lr": [1e-3],
                "attention_type": ["linear"],
                "max_iters": [20],
                "tol": [1e-4],
                "update_mode": ["mse_proxy"],
            }
    
    @staticmethod
    def get_bp_space(task: str, breadth: str = "normal"):
        """Get task-appropriate BP space."""
        config = TaskConfig.TASKS.get(task, TaskConfig.TASKS["mnist"])
        
        if breadth == "low":
            return {
                "lr": [1e-3],
                "d_model": [config["d_model"][0]],
                "optimizer": ["adam"],
            }
        elif breadth == "high":
            return {
                "lr": [1e-3, 2e-3, 5e-3],
                "d_model": config["d_model"],
                "optimizer": ["adam", "sgd"],
            }
        else:
            return {
                "lr": [1e-3, 2e-3],
                "d_model": config["d_model"],
                "optimizer": ["adam"],
            }


# =============================================================================
# CORE RESEARCH SYSTEM
# =============================================================================

@dataclass
class Result:
    """Trial result."""
    algorithm: str
    task: str
    accuracy: float
    time: float
    config: Dict[str, Any]


class TorEq:
    """THE core research system."""
    
    def __init__(self, breadth: str = "normal"):
        self.engine = HyperOptEngine()
        self.stats = StatisticalAnalyzer()
        self.breadth = breadth
        
        self.results: List[Result] = []
        self.param_impact: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        self.running = True
        signal.signal(signal.SIGINT, lambda *args: setattr(self, 'running', False))
    
    def run(self, minutes: float = 3):
        """Main research loop."""
        console.print(Panel(
            f"[bold magenta]TorEqProp Research[/bold magenta]\n"
            f"Time: {minutes}min | Breadth: {self.breadth}",
            title="🔬 Scientific Comparison"
        ))
        
        deadline = time.time() + minutes * 60
        tasks = ["xor", "xor3", "mnist", "cartpole"] if self.breadth != "low" else ["xor", "mnist"]
        
        round_num = 0
        while self.running and time.time() < deadline:
            round_num += 1
            console.print(f"\n[bold cyan]Round {round_num}[/bold cyan]")
            
            for task in tasks:
                if not self.running or time.time() >= deadline:
                    break
                
                epochs = TaskConfig.TASKS[task]["epochs"]
                
                # Run both algorithms
                console.print(f"\n[yellow]{task}[/yellow] ({epochs} epochs)")
                
                eq = self._run_trial("eqprop", task, epochs)
                bp = self._run_trial("bp", task, epochs)
                
                if eq and bp:
                    diff = eq.accuracy - bp.accuracy
                    winner = "🔋 EqProp" if diff > 0 else "⚡ BP"
                    console.print(f"  → {winner} ({diff:+.1%})")
            
            # Show insights
            if len(self.results) >= 4:
                self._show_insights()
        
        self._final_report()
    
    def _run_trial(self, algo: str, task: str, epochs: int) -> Optional[Result]:
        """Run single trial with progress bar."""
        
        # Get task-appropriate config
        if algo == "eqprop":
            space = TaskConfig.get_eqprop_space(task, self.breadth)
        else:
            space = TaskConfig.get_bp_space(task, self.breadth)
        
        # Sample
        config = {k: random.choice(v) for k, v in space.items()}
        
        # Show config
        key_params = ["beta", "damping", "d_model", "lr"] if algo == "eqprop" else ["lr", "d_model"]
        cfg_str = ", ".join(f"{k}={config[k]}" for k in key_params if k in config)
        
        console.print(f"  {'🔋' if algo == 'eqprop' else '⚡'} {algo}: {cfg_str}", end=" ")
        
        # Create trial
        trial = HyperOptTrial(
            trial_id=f"{algo}_{task}_{int(time.time()*1000)}",
            algorithm=algo,
            config=config,
            task=task,
            seed=random.randint(0, 9999)
        )
        
        # Run with progress bar
        start = time.time()
        
        # Create progress bar
        pbar = tqdm(total=100, desc="Training", leave=False, ncols=50, 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}|')
        
        # Run evaluation (would need to hook into trainer for real progress)
        # For now, simulate progress updates
        def progress_callback(epoch, total_epochs):
            pbar.n = int((epoch / total_epochs) * 100)
            pbar.refresh()
        
        trial = self.engine.evaluator.evaluate(trial, epochs=epochs, show_progress=False)
        pbar.close()
        
        elapsed = time.time() - start
        
        if trial.status != "complete":
            console.print(f"[red]FAILED[/red]")
            return None
        
        # Create result
        result = Result(
            algorithm=algo,
            task=task,
            accuracy=trial.performance,
            time=elapsed,
            config=config
        )
        
        self.results.append(result)
        self.engine.db.add_trial(trial)
        
        # Track param impact
        for k, v in config.items():
            self.param_impact[k][str(v)].append(result.accuracy)
        
        console.print(f"[{'green' if result.accuracy > 0.6 else 'yellow'}]{result.accuracy:.3f}[/] ({elapsed:.1f}s)")
        
        return result
    
    def _show_insights(self):
        """Show current insights."""
        eq = [r for r in self.results if r.algorithm == "eqprop"]
        bp = [r for r in self.results if r.algorithm == "bp"]
        
        if not eq or not bp:
            return
        
        eq_mean = np.mean([r.accuracy for r in eq])
        bp_mean = np.mean([r.accuracy for r in bp])
        
        # Parameter importance
        importance = {}
        for param, values in self.param_impact.items():
            if len(values) >= 2:
                means = [np.mean(v) for v in values.values() if v]
                if len(means) >= 2:
                    importance[param] = np.std(means)
        
        insights = []
        
        if eq_mean > bp_mean * 1.05:
            insights.append(f"✅ EqProp leading: {eq_mean:.1%} vs {bp_mean:.1%}")
        elif bp_mean > eq_mean * 1.05:
            insights.append(f"⚠️ BP leading: {bp_mean:.1%} vs {eq_mean:.1%}")
        else:
            insights.append(f"🔄 Tight race: {eq_mean:.1%} vs {bp_mean:.1%}")
        
        if importance:
            top = sorted(importance.items(), key=lambda x: -x[1])[0]
            best_val = max(self.param_impact[top[0]].items(), 
                          key=lambda x: np.mean(x[1]) if x[1] else 0)
            insights.append(f"📊 Key: {top[0]}={best_val[0]} (avg {np.mean(best_val[1]):.1%})")
        
        console.print(Panel("\n".join(insights), title="Insights", border_style="cyan"))
    
    def _final_report(self):
        """Final comprehensive report."""
        console.print("\n" + "="*60)
        console.print("[bold]FINAL REPORT[/bold]")
        console.print("="*60)
        
        # Scoreboard
        table = Table(title="Results")
        table.add_column("Task")
        table.add_column("EqProp Best", style="green")
        table.add_column("BP Best", style="yellow")
        table.add_column("Winner")
        
        tasks = sorted(set(r.task for r in self.results))
        eq_wins = bp_wins = 0
        
        for task in tasks:
            eq_best = max([r for r in self.results if r.algorithm == "eqprop" and r.task == task],
                         key=lambda x: x.accuracy, default=None)
            bp_best = max([r for r in self.results if r.algorithm == "bp" and r.task == task],
                         key=lambda x: x.accuracy, default=None)
            
            eq_s = f"{eq_best.accuracy:.3f}" if eq_best else "-"
            bp_s = f"{bp_best.accuracy:.3f}" if bp_best else "-"
            
            if eq_best and bp_best:
                if eq_best.accuracy > bp_best.accuracy:
                    winner = "🔋 EqProp"
                    eq_wins += 1
                else:
                    winner = "⚡ BP"
                    bp_wins += 1
            else:
                winner = "-"
            
            table.add_row(task, eq_s, bp_s, winner)
        
        console.print(table)
        console.print(f"\n[bold]Score: EqProp {eq_wins}, BP {bp_wins}[/bold]")
        
        # Parameter importance
        console.print("\n[bold cyan]Parameter Importance:[/bold cyan]")
        importance = {}
        for param, values in self.param_impact.items():
            if len(values) >= 2:
                means = [np.mean(v) for v in values.values() if v]
                if len(means) >= 2:
                    importance[param] = np.std(means)
        
        for param in sorted(importance, key=lambda x: -importance[x])[:5]:
            best_val = max(self.param_impact[param].items(),
                          key=lambda x: np.mean(x[1]) if x[1] else 0)
            console.print(f"  {param}: best={best_val[0]} (impact={importance[param]:.3f})")
        
        # Best configs
        console.print("\n[bold cyan]Best EqProp Config:[/bold cyan]")
        best_eq = max([r for r in self.results if r.algorithm == "eqprop"],
                      key=lambda x: x.accuracy, default=None)
        if best_eq:
            console.print(f"  Task: {best_eq.task} → {best_eq.accuracy:.3f}")
            for k, v in sorted(best_eq.config.items()):
                console.print(f"    {k}: {v}")
        
        # Recommendations
        console.print("\n[bold green]💡 Next Steps:[/bold green]")
        if eq_wins > bp_wins:
            console.print("  ✅ EqProp showing promise - scale up training time")
        else:
            console.print("  ⚠️ Tune beta/damping parameters")
        
        # Save
        self._save_report()
    
    def _save_report(self):
        """Save markdown report."""
        Path("results").mkdir(exist_ok=True)
        
        with open("results/latest.md", "w") as f:
            f.write(f"# TorEqProp Research\n\n")
            f.write(f"Experiments: {len(self.results)}\n\n")
            
            f.write("## Results\n\n| # | Algo | Task | Accuracy | Config |\n|-|--|-|-|-|\n")
            for i, r in enumerate(self.results, 1):
                cfg = ", ".join(f"{k}={v}" for k, v in list(r.config.items())[:3])
                f.write(f"| {i} | {r.algorithm} | {r.task} | {r.accuracy:.3f} | {cfg} |\n")
        
        console.print(f"\n[green]📄 Saved: results/latest.md[/green]")


def main():
    parser = argparse.ArgumentParser(description="TorEqProp Research")
    parser.add_argument("minutes", type=float, nargs="?", default=3)
    parser.add_argument("--deep", action="store_true", help="Deep exploration")
    args = parser.parse_args()
    
    breadth = "high" if args.deep else "low" if args.minutes < 5 else "normal"
    
    console.print("""[bold magenta]
╔════════════════════════════════════════════════════╗
║  🔬 TorEqProp vs Baselines: Scientific Analysis   ║
╚════════════════════════════════════════════════════╝[/bold magenta]
""")
    
    toreq = TorEq(breadth=breadth)
    toreq.run(minutes=args.minutes)


if __name__ == "__main__":
    main()
