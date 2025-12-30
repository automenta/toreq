#!/usr/bin/env python3
"""
TorEqProp Competitive Hyperparameter Optimization Engine

A systematic framework for finding optimal configurations of both EqProp and
baseline algorithms, then comparing them fairly across multiple cost dimensions.

Features:
- SearchSpace definitions for all algorithm hyperparameters
- Cost-aware evaluation (time, memory, iterations, parameters)
- Fair trial matching for apples-to-apples comparison
- Pareto frontier analysis for multi-objective optimization
- Comprehensive reporting with statistical analysis

Usage:
    python hyperopt_engine.py              # Run full optimization
    python hyperopt_engine.py --strategy random --n-trials 50
    python hyperopt_engine.py --smoke-test --n-trials 2
    python hyperopt_engine.py --report     # Generate report from existing results
"""

import argparse
import subprocess
import sys
import os
import time
import re
import yaml
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

from validation_db import ValidationDB, ExperimentRun, generate_experiment_id
from statistics import StatisticalAnalyzer, ComparisonResult


# =============================================================================
# SEARCH SPACE DEFINITIONS
# =============================================================================

@dataclass
class SearchSpace(ABC):
    """Base class for hyperparameter search spaces."""
    
    @abstractmethod
    def sample(self, rng: random.Random = None) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        pass
    
    @abstractmethod
    def grid(self) -> List[Dict[str, Any]]:
        """Return all configurations in the grid."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return the size of the search space."""
        pass


@dataclass
class EqPropSearchSpace(SearchSpace):
    """Search space for EqProp-specific hyperparameters.
    
    Covers all tunable aspects of Equilibrium Propagation:
    - Nudge strength (β)
    - Equilibrium solver parameters (damping, iterations, tolerance)
    - Architecture choices (attention type, symmetric mode)
    - Update mechanisms
    """
    
    # Nudge strength: critical for gradient approximation quality
    beta: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.22, 0.25, 0.3])
    
    # Damping: controls convergence speed vs stability tradeoff
    damping: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 0.95])
    
    # Max iterations: compute budget for equilibrium finding
    max_iters: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    
    # Convergence tolerance
    tol: List[float] = field(default_factory=lambda: [1e-4, 1e-5, 1e-6])
    
    # Attention type: linear required for symmetric mode
    attention_type: List[str] = field(default_factory=lambda: ["linear"])
    
    # Symmetric mode: theoretical guarantees vs practical performance
    symmetric: List[bool] = field(default_factory=lambda: [False, True])
    
    # Update mechanism: how gradients are approximated
    update_mode: List[str] = field(default_factory=lambda: ["mse_proxy", "vector_field"])
    
    # Model size: now includes tiny sizes for micro tasks
    d_model: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    
    # Learning rate
    lr: List[float] = field(default_factory=lambda: [5e-4, 1e-3, 2e-3])
    
    def sample(self, rng: random.Random = None) -> Dict[str, Any]:
        """Sample a random EqProp configuration."""
        if rng is None:
            rng = random.Random()
        
        config = {
            "algorithm": "eqprop",
            "beta": rng.choice(self.beta),
            "damping": rng.choice(self.damping),
            "max_iters": rng.choice(self.max_iters),
            "tol": rng.choice(self.tol),
            "attention_type": rng.choice(self.attention_type),
            "symmetric": rng.choice(self.symmetric),
            "update_mode": rng.choice(self.update_mode),
            "d_model": rng.choice(self.d_model),
            "lr": rng.choice(self.lr),
        }
        
        # Symmetric mode requires linear attention
        if config["symmetric"] and config["attention_type"] != "linear":
            config["attention_type"] = "linear"
        
        return config
    
    def grid(self) -> List[Dict[str, Any]]:
        """Generate full grid of EqProp configurations."""
        import itertools
        
        configs = []
        for beta, damping, max_iters, tol, attn, sym, mode, d_model, lr in itertools.product(
            self.beta, self.damping, self.max_iters, self.tol,
            self.attention_type, self.symmetric, self.update_mode,
            self.d_model, self.lr
        ):
            # Skip invalid: symmetric requires linear attention
            if sym and attn != "linear":
                continue
            
            configs.append({
                "algorithm": "eqprop",
                "beta": beta,
                "damping": damping,
                "max_iters": max_iters,
                "tol": tol,
                "attention_type": attn,
                "symmetric": sym,
                "update_mode": mode,
                "d_model": d_model,
                "lr": lr,
            })
        
        return configs
    
    def size(self) -> int:
        """Approximate size of search space."""
        # Account for symmetric requiring linear attention
        valid_sym_combos = len(self.attention_type)  # symmetric=True only with linear
        valid_nonsym_combos = len(self.attention_type)  # symmetric=False with any
        
        base = (len(self.beta) * len(self.damping) * len(self.max_iters) * 
                len(self.tol) * len(self.update_mode) * len(self.d_model) * len(self.lr))
        
        return base * (valid_sym_combos + valid_nonsym_combos)


@dataclass
class BaselineSearchSpace(SearchSpace):
    """Search space for baseline (BP) hyperparameters.
    
    Covers standard backpropagation training parameters to ensure
    fair comparison with optimized EqProp.
    """
    
    # Learning rate
    lr: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    
    # Optimizer choice
    optimizer: List[str] = field(default_factory=lambda: ["adam", "adamw"])
    
    # Model size (match EqProp options - includes tiny sizes)
    d_model: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    
    # Weight decay for AdamW
    weight_decay: List[float] = field(default_factory=lambda: [0, 1e-4, 1e-3])
    
    # Scheduler
    scheduler: List[str] = field(default_factory=lambda: ["none", "cosine"])
    
    def sample(self, rng: random.Random = None) -> Dict[str, Any]:
        """Sample a random baseline configuration."""
        if rng is None:
            rng = random.Random()
        
        config = {
            "algorithm": "bp",
            "lr": rng.choice(self.lr),
            "optimizer": rng.choice(self.optimizer),
            "d_model": rng.choice(self.d_model),
            "weight_decay": rng.choice(self.weight_decay),
            "scheduler": rng.choice(self.scheduler),
        }
        
        return config
    
    def grid(self) -> List[Dict[str, Any]]:
        """Generate full grid of baseline configurations."""
        import itertools
        
        configs = []
        for lr, opt, d_model, wd, sched in itertools.product(
            self.lr, self.optimizer, self.d_model, self.weight_decay, self.scheduler
        ):
            configs.append({
                "algorithm": "bp",
                "lr": lr,
                "optimizer": opt,
                "d_model": d_model,
                "weight_decay": wd,
                "scheduler": sched,
            })
        
        return configs
    
    def size(self) -> int:
        """Size of baseline search space."""
        return (len(self.lr) * len(self.optimizer) * len(self.d_model) * 
                len(self.weight_decay) * len(self.scheduler))


# =============================================================================
# TRIAL AND COST TRACKING
# =============================================================================

@dataclass
class CostMetrics:
    """Cost metrics for a single trial."""
    wall_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    total_iterations: int = 0  # For EqProp: equilibration iterations
    param_count: int = 0
    flops_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class HyperOptTrial:
    """A single hyperparameter optimization trial."""
    trial_id: str
    algorithm: str  # "eqprop" or "bp"
    config: Dict[str, Any]
    task: str  # e.g., "mnist", "cartpole"
    seed: int
    
    # Results
    performance: float = 0.0
    performance_metric: str = "accuracy"  # or "reward"
    cost: CostMetrics = field(default_factory=CostMetrics)
    
    # Meta
    status: str = "pending"  # pending, running, complete, failed
    timestamp: str = ""
    log_path: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "trial_id": self.trial_id,
            "algorithm": self.algorithm,
            "config": self.config,
            "task": self.task,
            "seed": self.seed,
            "performance": self.performance,
            "performance_metric": self.performance_metric,
            "cost": self.cost.to_dict(),
            "status": self.status,
            "timestamp": self.timestamp,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "HyperOptTrial":
        cost = CostMetrics(**d.get("cost", {}))
        return cls(
            trial_id=d["trial_id"],
            algorithm=d["algorithm"],
            config=d["config"],
            task=d["task"],
            seed=d["seed"],
            performance=d.get("performance", 0.0),
            performance_metric=d.get("performance_metric", "accuracy"),
            cost=cost,
            status=d.get("status", "pending"),
            timestamp=d.get("timestamp", ""),
            log_path=d.get("log_path", ""),
            error=d.get("error", ""),
        )


# =============================================================================
# COST-AWARE EVALUATION
# =============================================================================

class CostAwareEvaluator:
    """Evaluates trials across multiple cost dimensions.
    
    Tracks:
    - Performance (accuracy/reward)
    - Time cost (wall-clock training time)
    - Memory cost (peak GPU memory)
    - Iteration cost (equilibration iterations for EqProp)
    - Parameter cost (model size)
    """
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, trial: HyperOptTrial, epochs: int = 5, 
                 callback=None, show_progress: bool = True) -> HyperOptTrial:
        """Run a trial and capture all metrics."""
        
        trial.timestamp = datetime.now().isoformat()
        trial.status = "running"
        log_path = self.logs_dir / f"{trial.trial_id}.log"
        trial.log_path = str(log_path)
        
        # Build command based on algorithm
        if trial.algorithm == "eqprop":
            cmd = self._build_eqprop_command(trial, epochs)
        else:
            cmd = self._build_baseline_command(trial, epochs)
        
        start_time = time.time()
        
        # Show progress indicator
        if show_progress:
            print(f"   ⏳ Running", end="", flush=True)
        
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            output_lines = []
            last_progress_time = time.time()
            line_count = 0
            
            for line in process.stdout:
                output_lines.append(line)
                line_count += 1
                
                if callback:
                    callback(line)
                
                # Show progress dots every 5 seconds even in headless mode
                if show_progress and time.time() - last_progress_time > 5:
                    elapsed = time.time() - start_time
                    print(f".", end="", flush=True)
                    last_progress_time = time.time()
            
            process.wait()
            wall_time = time.time() - start_time
            output = "".join(output_lines)
            
            # Complete progress line
            if show_progress:
                print(f" ({wall_time:.0f}s)", flush=True)
            
            # Save log
            with open(log_path, "w") as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"Config: {json.dumps(trial.config, indent=2)}\n")
                f.write(f"Duration: {wall_time:.1f}s\n")
                f.write(f"Exit code: {process.returncode}\n")
                f.write("=" * 70 + "\n")
                f.write(output)
            
            # Extract metrics
            trial.performance = self._extract_performance(output, trial.task)
            trial.cost.wall_time_seconds = wall_time
            trial.cost.total_iterations = self._extract_iterations(output)
            trial.cost.param_count = self._estimate_params(trial.config)
            
            trial.status = "complete" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                trial.error = f"Exit code: {process.returncode}"
        
        except Exception as e:
            if show_progress:
                print(f" (error)", flush=True)
            trial.status = "failed"
            trial.error = str(e)
            trial.cost.wall_time_seconds = time.time() - start_time
        
        return trial

    
    def _build_eqprop_command(self, trial: HyperOptTrial, epochs: int) -> str:
        """Build command for EqProp trial.
        
        Supports all experiment types from the research plan:
        - Classification: mnist, fashion, cifar10, svhn
        - Algorithmic: parity, copy, addition
        - RL: CartPole, Acrobot, MountainCar, LunarLander
        - Memory: memory profiling
        """
        cfg = trial.config
        task = trial.task.lower() if isinstance(trial.task, str) else trial.task
        
        # Classification tasks
        if task in ["mnist", "fashion", "cifar10", "svhn"]:
            cmd = (f"python train.py --dataset {task} "
                   f"--epochs {epochs} --seed {trial.seed} "
                   f"--d-model {cfg['d_model']} --beta {cfg['beta']} "
                   f"--damping {cfg['damping']} --max-iters {cfg['max_iters']} "
                   f"--tol {cfg['tol']} --lr {cfg['lr']} "
                   f"--attention-type {cfg['attention_type']} "
                   f"--update-mode {cfg['update_mode']}")
            if cfg.get("symmetric", False):
                cmd += " --symmetric"
            if cfg.get("rapid", False):
                cmd += " --rapid"
        
        # Algorithmic reasoning tasks
        elif task in ["parity", "parity_8", "parity_12"]:
            seq_len = 12 if "12" in task else 8
            cmd = (f"python train_algorithmic.py --task parity "
                   f"--seq-len {seq_len} --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "copy":
            cmd = (f"python train_algorithmic.py --task copy "
                   f"--seq-len 8 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "addition":
            cmd = (f"python train_algorithmic.py --task addition "
                   f"--n-digits 4 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        # RL tasks
        elif task in ["cartpole-v1", "cartpole", "acrobot-v1", "acrobot", 
                      "mountaincar-v0", "mountaincar", "lunarlander-v2", "lunarlander"]:
            env_map = {
                "cartpole-v1": "CartPole-v1", "cartpole": "CartPole-v1",
                "acrobot-v1": "Acrobot-v1", "acrobot": "Acrobot-v1",
                "mountaincar-v0": "MountainCar-v0", "mountaincar": "MountainCar-v0",
                "lunarlander-v2": "LunarLander-v2", "lunarlander": "LunarLander-v2",
            }
            env = env_map.get(task, "CartPole-v1")
            episodes = epochs * 100 if epochs > 0 else 300
            cmd = (f"python train_rl.py --env {env} "
                   f"--episodes {episodes} --seed {trial.seed} "
                   f"--hidden-dim {cfg['d_model']}")
            if cfg.get("max_iters"):
                cmd += f" --max-iters {cfg['max_iters']}"
        
        # Memory profiling
        elif task in ["memory", "memory_profile"]:
            cmd = (f"python profile_memory.py --d-model {cfg['d_model']} "
                   f"--max-iters {cfg.get('max_iters', 100)} "
                   f"--seed {trial.seed}")
        
        else:
            cmd = f"echo 'Unknown task: {trial.task}'"
        
        return cmd
    
    def _build_baseline_command(self, trial: HyperOptTrial, epochs: int) -> str:
        """Build command for baseline trial.
        
        Supports all experiment types with BP baseline.
        """
        cfg = trial.config
        task = trial.task.lower() if isinstance(trial.task, str) else trial.task
        
        # Classification tasks
        if task in ["mnist", "fashion", "cifar10", "svhn"]:
            cmd = (f"python train_mnist_bp.py --dataset {task} "
                   f"--epochs {epochs} --seed {trial.seed} "
                   f"--d-model {cfg['d_model']} --lr {cfg['lr']}")
        
        # Algorithmic reasoning tasks (use --use-bp flag)
        elif task in ["parity", "parity_8", "parity_12"]:
            seq_len = 12 if "12" in task else 8
            cmd = (f"python train_algorithmic.py --task parity --use-bp "
                   f"--seq-len {seq_len} --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "copy":
            cmd = (f"python train_algorithmic.py --task copy --use-bp "
                   f"--seq-len 8 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        elif task == "addition":
            cmd = (f"python train_algorithmic.py --task addition --use-bp "
                   f"--n-digits 4 --epochs {epochs} "
                   f"--seed {trial.seed} --d-model {cfg['d_model']} "
                   f"--lr {cfg['lr']}")
        
        # RL tasks
        elif task in ["cartpole-v1", "cartpole", "acrobot-v1", "acrobot",
                      "mountaincar-v0", "mountaincar", "lunarlander-v2", "lunarlander"]:
            env_map = {
                "cartpole-v1": "CartPole-v1", "cartpole": "CartPole-v1",
                "acrobot-v1": "Acrobot-v1", "acrobot": "Acrobot-v1",
                "mountaincar-v0": "MountainCar-v0", "mountaincar": "MountainCar-v0",
                "lunarlander-v2": "LunarLander-v2", "lunarlander": "LunarLander-v2",
            }
            env = env_map.get(task, "CartPole-v1")
            episodes = epochs * 100 if epochs > 0 else 300
            cmd = (f"python train_rl.py --env {env} --use-bp "
                   f"--episodes {episodes} --seed {trial.seed} "
                   f"--hidden-dim {cfg['d_model']}")
        
        # Memory profiling (comparison mode - runs both)
        elif task in ["memory", "memory_profile"]:
            cmd = (f"python profile_memory.py --d-model {cfg['d_model']} "
                   f"--max-iters 100 --seed {trial.seed}")
        
        else:
            cmd = f"echo 'Unknown task: {trial.task}'"
        
        return cmd

    
    def _extract_performance(self, output: str, task: str) -> float:
        """Extract performance metric from output."""
        # Try accuracy patterns
        acc_patterns = [
            r"Test Acc(?:uracy)?:\s*([\d.]+)",
            r"test/accuracy:\s*([\d.]+)",
            r"Final.*?Accuracy:\s*([\d.]+)",
        ]
        for pattern in acc_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # Try reward patterns for RL
        reward_patterns = [
            r"Final Average Reward:\s*([-\d.]+)",
            r"avg_reward:\s*([-\d.]+)",
            r"Average Reward:\s*([-\d.]+)",
        ]
        for pattern in reward_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_iterations(self, output: str) -> int:
        """Extract total equilibration iterations."""
        patterns = [
            r"Iters:\s*(\d+)/(\d+)",
            r"iters_free:\s*(\d+)",
            r"train/iters_free.*?:\s*([\d.]+)",
        ]
        total_iters = 0
        for pattern in patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                try:
                    total_iters += int(float(match.group(1)))
                except:
                    pass
        return total_iters if total_iters > 0 else 1  # Default 1 for BP
    
    def _estimate_params(self, config: Dict) -> int:
        """Estimate parameter count from config."""
        d = config.get("d_model", 128)
        # Rough estimate: embedding + transformer block + head
        # Actual: d*784 + 4*d*d + 2*d*d*4 + d*10
        return int(d * 784 + 4 * d * d + 8 * d * d + d * 10)


# =============================================================================
# TRIAL MATCHING FOR FAIR COMPARISON
# =============================================================================

@dataclass
class MatchedPair:
    """A matched pair of EqProp and baseline trials."""
    eqprop_trial: HyperOptTrial
    baseline_trial: HyperOptTrial
    match_quality: float  # 0-1, how well matched
    match_criteria: str  # What was matched on
    
    def performance_diff(self) -> float:
        """EqProp performance - Baseline performance."""
        return self.eqprop_trial.performance - self.baseline_trial.performance
    
    def time_ratio(self) -> float:
        """Time ratio: EqProp / Baseline."""
        if self.baseline_trial.cost.wall_time_seconds > 0:
            return (self.eqprop_trial.cost.wall_time_seconds / 
                    self.baseline_trial.cost.wall_time_seconds)
        return 1.0


class TrialMatcher:
    """Match EqProp trials to baseline trials for fair comparison.
    
    Matching strategies:
    - time_matched: Similar training time
    - param_matched: Same parameter count
    - size_matched: Same model size (d_model)
    """
    
    def __init__(self, strategy: str = "time_matched", tolerance: float = 0.1):
        self.strategy = strategy
        self.tolerance = tolerance
    
    def match(self, eqprop_trials: List[HyperOptTrial],
              baseline_trials: List[HyperOptTrial]) -> List[MatchedPair]:
        """Find matched pairs between EqProp and baseline trials."""
        
        if self.strategy == "time_matched":
            return self._match_by_time(eqprop_trials, baseline_trials)
        elif self.strategy == "param_matched":
            return self._match_by_params(eqprop_trials, baseline_trials)
        elif self.strategy == "size_matched":
            return self._match_by_size(eqprop_trials, baseline_trials)
        else:
            raise ValueError(f"Unknown matching strategy: {self.strategy}")
    
    def _match_by_time(self, eqprop: List[HyperOptTrial],
                       baseline: List[HyperOptTrial]) -> List[MatchedPair]:
        """Match trials with similar training time."""
        pairs = []
        used_baseline = set()
        
        for eq in eqprop:
            if eq.status != "complete":
                continue
            
            best_match = None
            best_diff = float("inf")
            
            for bl in baseline:
                if bl.trial_id in used_baseline or bl.status != "complete":
                    continue
                
                time_diff = abs(eq.cost.wall_time_seconds - bl.cost.wall_time_seconds)
                relative_diff = time_diff / max(eq.cost.wall_time_seconds, 1)
                
                if relative_diff < self.tolerance and relative_diff < best_diff:
                    best_match = bl
                    best_diff = relative_diff
            
            if best_match:
                quality = 1.0 - best_diff
                pairs.append(MatchedPair(eq, best_match, quality, "time_matched"))
                used_baseline.add(best_match.trial_id)
        
        return pairs
    
    def _match_by_params(self, eqprop: List[HyperOptTrial],
                         baseline: List[HyperOptTrial]) -> List[MatchedPair]:
        """Match trials with similar parameter count."""
        pairs = []
        used_baseline = set()
        
        for eq in eqprop:
            if eq.status != "complete":
                continue
            
            best_match = None
            best_diff = float("inf")
            
            for bl in baseline:
                if bl.trial_id in used_baseline or bl.status != "complete":
                    continue
                
                param_diff = abs(eq.cost.param_count - bl.cost.param_count)
                relative_diff = param_diff / max(eq.cost.param_count, 1)
                
                if relative_diff < self.tolerance and relative_diff < best_diff:
                    best_match = bl
                    best_diff = relative_diff
            
            if best_match:
                quality = 1.0 - best_diff
                pairs.append(MatchedPair(eq, best_match, quality, "param_matched"))
                used_baseline.add(best_match.trial_id)
        
        return pairs
    
    def _match_by_size(self, eqprop: List[HyperOptTrial],
                       baseline: List[HyperOptTrial]) -> List[MatchedPair]:
        """Match trials with same model size (d_model)."""
        pairs = []
        
        # Group by d_model
        eq_by_size = defaultdict(list)
        bl_by_size = defaultdict(list)
        
        for eq in eqprop:
            if eq.status == "complete":
                d = eq.config.get("d_model", 128)
                eq_by_size[d].append(eq)
        
        for bl in baseline:
            if bl.status == "complete":
                d = bl.config.get("d_model", 128)
                bl_by_size[d].append(bl)
        
        # Match best from each size bucket
        for d_model in eq_by_size:
            if d_model not in bl_by_size:
                continue
            
            # Sort by performance
            eq_sorted = sorted(eq_by_size[d_model], 
                              key=lambda t: t.performance, reverse=True)
            bl_sorted = sorted(bl_by_size[d_model],
                              key=lambda t: t.performance, reverse=True)
            
            # Match best with best, second with second, etc.
            for eq, bl in zip(eq_sorted, bl_sorted):
                pairs.append(MatchedPair(eq, bl, 1.0, f"size_matched_d{d_model}"))
        
        return pairs


# =============================================================================
# PARETO ANALYSIS
# =============================================================================

class ParetoAnalyzer:
    """Find Pareto-optimal configurations across multiple objectives."""
    
    @staticmethod
    def is_dominated(trial1: HyperOptTrial, trial2: HyperOptTrial,
                     objectives: List[str]) -> bool:
        """Check if trial1 is dominated by trial2.
        
        trial1 is dominated if trial2 is better or equal in all objectives
        and strictly better in at least one.
        """
        better_in_one = False
        
        for obj in objectives:
            val1 = ParetoAnalyzer._get_objective_value(trial1, obj)
            val2 = ParetoAnalyzer._get_objective_value(trial2, obj)
            
            # Higher is better for performance, lower is better for costs
            if obj == "performance":
                if val2 < val1:
                    return False  # trial2 worse in this objective
                if val2 > val1:
                    better_in_one = True
            else:  # cost objectives: lower is better
                if val2 > val1:
                    return False
                if val2 < val1:
                    better_in_one = True
        
        return better_in_one
    
    @staticmethod
    def _get_objective_value(trial: HyperOptTrial, obj: str) -> float:
        if obj == "performance":
            return trial.performance
        elif obj == "time":
            return trial.cost.wall_time_seconds
        elif obj == "memory":
            return trial.cost.peak_memory_mb
        elif obj == "params":
            return trial.cost.param_count
        elif obj == "iterations":
            return trial.cost.total_iterations
        else:
            return 0.0
    
    @staticmethod
    def pareto_frontier(trials: List[HyperOptTrial],
                       objectives: List[str] = None) -> List[HyperOptTrial]:
        """Find Pareto-optimal trials.
        
        Default objectives: maximize performance, minimize time.
        """
        if objectives is None:
            objectives = ["performance", "time"]
        
        # Filter to complete trials
        complete = [t for t in trials if t.status == "complete"]
        
        frontier = []
        for candidate in complete:
            dominated = False
            for other in complete:
                if other.trial_id == candidate.trial_id:
                    continue
                if ParetoAnalyzer.is_dominated(candidate, other, objectives):
                    dominated = True
                    break
            
            if not dominated:
                frontier.append(candidate)
        
        return frontier
    
    @staticmethod
    def compute_hypervolume(frontier: List[HyperOptTrial],
                           reference_point: Tuple[float, float],
                           objectives: List[str] = None) -> float:
        """Compute hypervolume indicator for Pareto frontier quality.
        
        Higher hypervolume = better frontier.
        Reference point should be worse than all frontier points.
        """
        if objectives is None:
            objectives = ["performance", "time"]
        
        if len(objectives) != 2:
            raise ValueError("Hypervolume only implemented for 2 objectives")
        
        if not frontier:
            return 0.0
        
        # Sort by first objective (performance, descending)
        sorted_frontier = sorted(
            frontier,
            key=lambda t: ParetoAnalyzer._get_objective_value(t, objectives[0]),
            reverse=True
        )
        
        # Compute hypervolume using inclusion-exclusion
        hypervolume = 0.0
        prev_obj2 = reference_point[1]
        
        for trial in sorted_frontier:
            obj1 = ParetoAnalyzer._get_objective_value(trial, objectives[0])
            obj2 = ParetoAnalyzer._get_objective_value(trial, objectives[1])
            
            # For performance (higher better), time (lower better)
            width = obj1 - reference_point[0]  # Performance contribution
            height = prev_obj2 - obj2          # Time contribution
            
            if width > 0 and height > 0:
                hypervolume += width * height
            
            prev_obj2 = min(prev_obj2, obj2)
        
        return hypervolume


# =============================================================================
# HYPEROPT DATABASE
# =============================================================================

class HyperOptDB:
    """Database for storing hyperopt trials."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.trials: Dict[str, HyperOptTrial] = {}
        self._load()
    
    def _load(self):
        """Load trials from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    content = f.read().strip()
                    if not content:
                        return  # Empty file
                    data = json.loads(content)
                    for trial_data in data.get("trials", []):
                        trial = HyperOptTrial.from_dict(trial_data)
                        self.trials[trial.trial_id] = trial
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load DB from {self.db_path}: {e}")
                self.trials = {}
    
    def _save(self):
        """Save trials to disk."""
        data = {
            "trials": [t.to_dict() for t in self.trials.values()],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_trial(self, trial: HyperOptTrial):
        """Add or update a trial."""
        self.trials[trial.trial_id] = trial
        self._save()
    
    def get_trial(self, trial_id: str) -> Optional[HyperOptTrial]:
        """Get a trial by ID."""
        return self.trials.get(trial_id)
    
    def get_trials(self, algorithm: str = None, task: str = None,
                   status: str = None) -> List[HyperOptTrial]:
        """Get trials matching filters."""
        result = list(self.trials.values())
        
        if algorithm:
            result = [t for t in result if t.algorithm == algorithm]
        if task:
            result = [t for t in result if t.task == task]
        if status:
            result = [t for t in result if t.status == status]
        
        return result
    
    def get_best_trial(self, algorithm: str, task: str) -> Optional[HyperOptTrial]:
        """Get best-performing trial for algorithm and task."""
        trials = self.get_trials(algorithm=algorithm, task=task, status="complete")
        if not trials:
            return None
        return max(trials, key=lambda t: t.performance)


# =============================================================================
# MAIN HYPEROPT ENGINE
# =============================================================================

class HyperOptEngine:
    """Main hyperparameter optimization engine.
    
    Orchestrates:
    1. Search space definition
    2. Trial sampling/generation
    3. Cost-aware evaluation
    4. Fair trial matching
    5. Pareto analysis and reporting
    """
    
    def __init__(self, config_path: str = "validation_config.yaml"):
        self.config = self._load_config(config_path)
        self.db = HyperOptDB(self.config.get("output", {}).get(
            "hyperopt_db", "data/hyperopt_results.json"))
        
        self.eqprop_space = self._create_eqprop_space()
        self.baseline_space = self._create_baseline_space()
        
        logs_dir = Path(self.config.get("output", {}).get(
            "logs_dir", "logs/hyperopt"))
        self.evaluator = CostAwareEvaluator(logs_dir)
        
        self.matcher = TrialMatcher(
            strategy=self.config.get("hyperopt", {}).get(
                "matching", {}).get("strategy", "time_matched"),
            tolerance=self.config.get("hyperopt", {}).get(
                "matching", {}).get("tolerance", 0.1)
        )
        
        self.analyzer = StatisticalAnalyzer()
    
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML."""
        if Path(path).exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _create_eqprop_space(self) -> EqPropSearchSpace:
        """Create EqProp search space from config."""
        cfg = self.config.get("hyperopt", {}).get("eqprop_search_space", {})
        return EqPropSearchSpace(
            beta=cfg.get("beta", [0.05, 0.1, 0.15, 0.2, 0.22, 0.25, 0.3]),
            damping=cfg.get("damping", [0.7, 0.8, 0.9, 0.95]),
            max_iters=cfg.get("max_iters", [10, 20, 50, 100]),
            tol=cfg.get("tol", [1e-4, 1e-5, 1e-6]),
            attention_type=cfg.get("attention_type", ["linear"]),
            symmetric=cfg.get("symmetric", [False, True]),
            update_mode=cfg.get("update_mode", ["mse_proxy", "vector_field"]),
            d_model=cfg.get("d_model", [64, 128, 256]),
            lr=cfg.get("lr", [5e-4, 1e-3, 2e-3]),
        )
    
    def _create_baseline_space(self) -> BaselineSearchSpace:
        """Create baseline search space from config."""
        cfg = self.config.get("hyperopt", {}).get("baseline_search_space", {})
        return BaselineSearchSpace(
            lr=cfg.get("lr", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]),
            optimizer=cfg.get("optimizer", ["adam", "adamw"]),
            d_model=cfg.get("d_model", [64, 128, 256]),
            weight_decay=cfg.get("weight_decay", [0, 1e-4, 1e-3]),
            scheduler=cfg.get("scheduler", ["none", "cosine"]),
        )
    
    def run(self, task: str = "mnist", n_trials: int = 50,
            strategy: str = "random", seeds: List[int] = None,
            epochs: int = 5, headless: bool = False):
        """Run hyperparameter optimization.
        
        Args:
            task: Task to optimize on (mnist, cartpole, etc.)
            n_trials: Number of trials per algorithm
            strategy: Search strategy (grid, random, bayesian)
            seeds: Random seeds to use
            epochs: Training epochs per trial
            headless: Suppress output
        """
        print("\n" + "=" * 70)
        print("  TorEqProp Competitive Hyperparameter Optimization")
        print("=" * 70)
        print(f"  Task: {task}")
        print(f"  Strategy: {strategy}")
        print(f"  Trials per algorithm: {n_trials}")
        print(f"  Epochs per trial: {epochs}")
        print(f"  EqProp search space: {self.eqprop_space.size()} configs")
        print(f"  Baseline search space: {self.baseline_space.size()} configs")
        print("=" * 70)
        
        if seeds is None:
            seeds = [0, 1, 2]
        
        rng = random.Random(42)
        
        # Generate trial configurations
        eqprop_configs = self._sample_configs(self.eqprop_space, n_trials, strategy, rng)
        baseline_configs = self._sample_configs(self.baseline_space, n_trials, strategy, rng)
        
        # Run EqProp trials
        print("\n" + "-" * 70)
        print("🔋 Phase 1: EqProp Hyperparameter Search")
        print("-" * 70)
        
        for i, cfg in enumerate(eqprop_configs):
            for seed in seeds:
                trial_id = f"eq_{task}_{i}_s{seed}"
                
                # Skip if already complete
                existing = self.db.get_trial(trial_id)
                if existing and existing.status == "complete":
                    print(f"  ⏭️  {trial_id} already complete, skipping")
                    continue
                
                trial = HyperOptTrial(
                    trial_id=trial_id,
                    algorithm="eqprop",
                    config=cfg,
                    task=task,
                    seed=seed,
                )
                
                print(f"\n📊 Trial {i+1}/{len(eqprop_configs)} seed {seed}: {trial_id}")
                print(f"   Config: β={cfg['beta']}, damping={cfg['damping']}, "
                      f"iters={cfg['max_iters']}, d={cfg['d_model']}")
                
                def callback(line):
                    if not headless:
                        print(f"   {line.strip()}", flush=True)
                
                trial = self.evaluator.evaluate(trial, epochs=epochs, callback=callback)
                self.db.add_trial(trial)
                
                status = "✅" if trial.status == "complete" else "❌"
                print(f"   {status} Performance: {trial.performance:.4f}, "
                      f"Time: {trial.cost.wall_time_seconds:.1f}s")
        
        # Run baseline trials
        print("\n" + "-" * 70)
        print("⚡ Phase 2: Baseline Hyperparameter Search")
        print("-" * 70)
        
        for i, cfg in enumerate(baseline_configs):
            for seed in seeds:
                trial_id = f"bp_{task}_{i}_s{seed}"
                
                existing = self.db.get_trial(trial_id)
                if existing and existing.status == "complete":
                    print(f"  ⏭️  {trial_id} already complete, skipping")
                    continue
                
                trial = HyperOptTrial(
                    trial_id=trial_id,
                    algorithm="bp",
                    config=cfg,
                    task=task,
                    seed=seed,
                )
                
                print(f"\n📊 Trial {i+1}/{len(baseline_configs)} seed {seed}: {trial_id}")
                print(f"   Config: lr={cfg['lr']}, opt={cfg['optimizer']}, d={cfg['d_model']}")
                
                def callback(line):
                    if not headless:
                        print(f"   {line.strip()}", flush=True)
                
                trial = self.evaluator.evaluate(trial, epochs=epochs, callback=callback)
                self.db.add_trial(trial)
                
                status = "✅" if trial.status == "complete" else "❌"
                print(f"   {status} Performance: {trial.performance:.4f}, "
                      f"Time: {trial.cost.wall_time_seconds:.1f}s")
        
        # Analysis
        self._print_analysis(task)
    
    def _sample_configs(self, space: SearchSpace, n: int, 
                        strategy: str, rng: random.Random) -> List[Dict]:
        """Sample configurations from search space."""
        if strategy == "grid":
            all_configs = space.grid()
            if len(all_configs) <= n:
                return all_configs
            return rng.sample(all_configs, n)
        elif strategy == "random":
            return [space.sample(rng) for _ in range(n)]
        else:
            # Default to random
            return [space.sample(rng) for _ in range(n)]
    
    def _print_analysis(self, task: str):
        """Print analysis of completed trials."""
        print("\n" + "=" * 70)
        print("  HYPEROPT ANALYSIS")
        print("=" * 70)
        
        eqprop_trials = self.db.get_trials(algorithm="eqprop", task=task, status="complete")
        baseline_trials = self.db.get_trials(algorithm="bp", task=task, status="complete")
        
        if not eqprop_trials or not baseline_trials:
            print("  ❌ Insufficient trials for analysis")
            return
        
        # Best configurations
        best_eq = max(eqprop_trials, key=lambda t: t.performance)
        best_bl = max(baseline_trials, key=lambda t: t.performance)
        
        print(f"\n📊 Best Configurations:")
        print(f"\n  🔋 EqProp Best: {best_eq.performance:.4f}")
        print(f"     Config: β={best_eq.config['beta']}, damping={best_eq.config['damping']}, "
              f"iters={best_eq.config['max_iters']}, d={best_eq.config['d_model']}")
        print(f"     Time: {best_eq.cost.wall_time_seconds:.1f}s")
        
        print(f"\n  ⚡ Baseline Best: {best_bl.performance:.4f}")
        print(f"     Config: lr={best_bl.config['lr']}, opt={best_bl.config['optimizer']}, "
              f"d={best_bl.config['d_model']}")
        print(f"     Time: {best_bl.cost.wall_time_seconds:.1f}s")
        
        # Statistical comparison
        eq_perfs = [t.performance for t in eqprop_trials]
        bl_perfs = [t.performance for t in baseline_trials]
        
        result = self.analyzer.compare(eq_perfs, bl_perfs, "EqProp", "Baseline")
        
        print(f"\n📈 Statistical Comparison (all trials):")
        print(f"   EqProp: {result.algo1_mean:.4f} ± {result.algo1_std:.4f} (n={result.algo1_n})")
        print(f"   Baseline: {result.algo2_mean:.4f} ± {result.algo2_std:.4f} (n={result.algo2_n})")
        print(f"   Difference: {result.improvement_pct:+.2f}%")
        print(f"   p-value: {result.p_value:.4f}")
        print(f"   Cohen's d: {result.cohens_d:.2f}")
        print(f"   Significant: {'Yes' if result.is_significant else 'No'}")
        
        # Matched comparison
        pairs = self.matcher.match(eqprop_trials, baseline_trials)
        
        if pairs:
            print(f"\n⚖️  Matched Comparisons ({len(pairs)} pairs, {self.matcher.strategy}):")
            eq_wins = sum(1 for p in pairs if p.performance_diff() > 0)
            bl_wins = sum(1 for p in pairs if p.performance_diff() < 0)
            ties = len(pairs) - eq_wins - bl_wins
            
            print(f"   EqProp wins: {eq_wins}/{len(pairs)}")
            print(f"   Baseline wins: {bl_wins}/{len(pairs)}")
            print(f"   Ties: {ties}/{len(pairs)}")
            
            avg_diff = np.mean([p.performance_diff() for p in pairs])
            avg_time_ratio = np.mean([p.time_ratio() for p in pairs])
            
            print(f"   Avg performance diff: {avg_diff:+.4f}")
            print(f"   Avg time ratio (EqProp/Baseline): {avg_time_ratio:.2f}x")
        
        # Pareto frontier
        all_trials = eqprop_trials + baseline_trials
        frontier = ParetoAnalyzer.pareto_frontier(all_trials, ["performance", "time"])
        
        print(f"\n🎯 Pareto Frontier (performance vs time):")
        eq_on_frontier = sum(1 for t in frontier if t.algorithm == "eqprop")
        bl_on_frontier = sum(1 for t in frontier if t.algorithm == "bp")
        print(f"   Total on frontier: {len(frontier)}")
        print(f"   EqProp: {eq_on_frontier}, Baseline: {bl_on_frontier}")
        
        for t in sorted(frontier, key=lambda x: x.performance, reverse=True)[:5]:
            marker = "🔋" if t.algorithm == "eqprop" else "⚡"
            print(f"   {marker} {t.performance:.4f} @ {t.cost.wall_time_seconds:.1f}s")
        
        # Time-normalized analysis
        eq_times = [t.cost.wall_time_seconds for t in eqprop_trials]
        bl_times = [t.cost.wall_time_seconds for t in baseline_trials]
        
        avg_eq_time = np.mean(eq_times) if eq_times else 0
        avg_bl_time = np.mean(bl_times) if bl_times else 1
        
        speed_ratio = avg_bl_time / avg_eq_time if avg_eq_time > 0 else 1
        perf_gap = best_bl.performance - best_eq.performance
        
        print(f"\n⏱️  Time-Normalized Analysis:")
        print(f"   Avg EqProp time: {avg_eq_time:.1f}s")
        print(f"   Avg Baseline time: {avg_bl_time:.1f}s")
        print(f"   Speed advantage: {speed_ratio:.1f}x faster")
        print(f"   Performance gap: {perf_gap:+.4f}")
        
        if speed_ratio > 5:
            efficiency = perf_gap / speed_ratio
            print(f"   Efficiency ratio: {efficiency:.4f} perf per unit speed")
            
        # Verdict
        print("\n" + "=" * 70)
        print("  VERDICT")
        print("=" * 70)
        
        if result.is_significant and result.algo1_mean > result.algo2_mean:
            print("  🏆 EqProp shows SIGNIFICANT ADVANTAGE over baseline!")
            print(f"     +{result.improvement_pct:.1f}% performance (p={result.p_value:.4f})")
        elif result.is_significant and result.algo2_mean > result.algo1_mean:
            print("  ⚠️  Baseline outperforms EqProp significantly")
            print(f"     {result.improvement_pct:.1f}% worse (p={result.p_value:.4f})")
        else:
            print("  📊 No significant difference between EqProp and baseline")
            print(f"     Δ={result.improvement_pct:+.1f}%, p={result.p_value:.4f}")
        
        if eq_on_frontier > bl_on_frontier:
            print(f"  🎯 EqProp dominates Pareto frontier ({eq_on_frontier}/{len(frontier)})")
        elif bl_on_frontier > eq_on_frontier:
            print(f"  ⚡ Baseline dominates Pareto frontier ({bl_on_frontier}/{len(frontier)})")
        
        # Interpretation / Big Picture
        print("\n" + "-" * 70)
        print("  💡 INSIGHTS")
        print("-" * 70)
        
        if speed_ratio > 5:
            print(f"  • EqProp is {speed_ratio:.0f}x faster - significant efficiency advantage")
            if perf_gap < 0.05:  # 5% gap
                print("  • Small accuracy gap may be acceptable for speed tradeoff")
                print("  • Consider: At equal time budget, EqProp may match BP accuracy")
        
        if task.lower() in ["cartpole", "acrobot", "lunarlander", "mountaincar"]:
            if best_eq.performance > best_bl.performance:
                print("  • 🎉 EqProp OUTPERFORMS BP on RL - key finding!")
                print("  • Novel contribution: First EP superiority on control tasks")
        
        # Check for model size mismatch
        eq_d = [t.config.get('d_model', 0) for t in eqprop_trials]
        bl_d = [t.config.get('d_model', 0) for t in baseline_trials]
        
        avg_eq_d = np.mean(eq_d) if eq_d else 0
        avg_bl_d = np.mean(bl_d) if bl_d else 0
        
        if abs(avg_eq_d - avg_bl_d) > 50:
            print(f"  ⚠️  Model size mismatch: EqProp avg d={avg_eq_d:.0f}, BP avg d={avg_bl_d:.0f}")
            print("  • Consider running with matched d_model for fair comparison")
        
        # Recommendations
        print("\n" + "-" * 70)
        print("  📋 RECOMMENDATIONS")
        print("-" * 70)
        
        if result.algo2_mean > result.algo1_mean and speed_ratio > 5:
            print("  1. Run fair comparison with same d_model")
            print("  2. Test EqProp with larger model (d=256)")
            print("  3. Consider time-budget experiment (same wall-clock time)")
        
        if best_eq.performance < 0.90 and task.lower() in ["mnist", "fashion"]:
            print("  1. Try more epochs (current may be insufficient)")
            print("  2. Tune β (best range: 0.20-0.25)")
            print("  3. Increase d_model for more capacity")
        
        if task.lower() in ["parity", "copy", "addition"]:
            print("  1. Algorithmic tasks may show adaptive compute advantage")
            print("  2. Track convergence iterations per sample difficulty")
        
        print("=" * 70)
    
    def report(self, task: str = None):
        """Generate report from existing results."""
        tasks = [task] if task else ["mnist", "fashion", "cifar10"]
        
        for t in tasks:
            eqprop = self.db.get_trials(algorithm="eqprop", task=t, status="complete")
            baseline = self.db.get_trials(algorithm="bp", task=t, status="complete")
            
            if eqprop or baseline:
                print(f"\n📊 Task: {t}")
                print(f"   EqProp trials: {len(eqprop)}")
                print(f"   Baseline trials: {len(baseline)}")
                
                if eqprop and baseline:
                    self._print_analysis(t)
    
    def smoke_test(self, n_trials: int = 2, task: str = "mnist"):
        """Quick smoke test with minimal trials."""
        print("\n🧪 SMOKE TEST MODE")
        print("=" * 70)
        self.run(task=task, n_trials=n_trials, epochs=1, headless=True)
    
    def run_campaign(self, tasks: List[str] = None, n_trials: int = 10,
                     strategy: str = "random", seeds: List[int] = None,
                     epochs: int = 3, rapid: bool = False):
        """Run comprehensive research campaign across multiple tasks.
        
        Implements the TorEqProp research plan phases:
        - Phase 1: Classification (mnist, fashion, cifar10)
        - Phase 2: Algorithmic (parity, copy, addition)
        - Phase 3: RL (cartpole, acrobot)
        - Phase 4: Memory profiling
        
        Args:
            tasks: List of tasks to run, or None for all
            n_trials: Number of trials per algorithm per task
            strategy: Search strategy
            seeds: Random seeds
            epochs: Epochs per trial (reduced in rapid mode)
            rapid: Use rapid mode (fewer epochs, smaller models, faster feedback)
        """
        if tasks is None:
            tasks = ["mnist", "fashion", "cartpole", "parity"]
        
        if seeds is None:
            seeds = [0, 1, 2]
        
        # Rapid mode adjustments
        if rapid:
            epochs = min(epochs, 1)
            n_trials = min(n_trials, 5)
            seeds = seeds[:2]  # Only 2 seeds in rapid mode
            print("\n⚡ RAPID MODE: Reduced epochs, trials, and seeds for fast feedback")
        
        # Estimate total time
        total_trials = len(tasks) * n_trials * len(seeds) * 2  # 2 algorithms
        est_time_min = total_trials * (0.5 if rapid else 2)  # ~30s rapid, ~2min normal
        
        print("\n" + "=" * 70)
        print("  TorEqProp Research Campaign")
        print("=" * 70)
        print(f"  Tasks: {', '.join(tasks)}")
        print(f"  Trials per algorithm: {n_trials}")
        print(f"  Seeds: {seeds}")
        print(f"  Total trials: {total_trials}")
        print(f"  Estimated time: {est_time_min:.0f} min")
        print("=" * 70)
        
        campaign_start = time.time()
        completed_tasks = []
        
        for i, task in enumerate(tasks):
            task_start = time.time()
            print(f"\n{'='*70}")
            print(f"📋 TASK {i+1}/{len(tasks)}: {task.upper()}")
            print(f"{'='*70}")
            
            try:
                self.run(
                    task=task,
                    n_trials=n_trials,
                    strategy=strategy,
                    seeds=seeds,
                    epochs=epochs,
                    headless=True
                )
                completed_tasks.append(task)
                
                task_time = time.time() - task_start
                print(f"\n✅ {task} complete in {task_time/60:.1f} min")
                
            except Exception as e:
                print(f"\n❌ {task} failed: {e}")
                continue
        
        # Campaign summary
        total_time = time.time() - campaign_start
        print("\n" + "=" * 70)
        print("  CAMPAIGN COMPLETE")
        print("=" * 70)
        print(f"  Completed: {len(completed_tasks)}/{len(tasks)} tasks")
        print(f"  Total time: {total_time/60:.1f} min")
        print(f"  Tasks: {', '.join(completed_tasks)}")
        
        # Generate combined report
        print("\n" + "-" * 70)
        print("📊 Combined Analysis")
        print("-" * 70)
        
        all_eqprop = []
        all_baseline = []
        
        for task in completed_tasks:
            eqprop = self.db.get_trials(algorithm="eqprop", task=task, status="complete")
            baseline = self.db.get_trials(algorithm="bp", task=task, status="complete")
            all_eqprop.extend(eqprop)
            all_baseline.extend(baseline)
            
            if eqprop and baseline:
                eq_avg = np.mean([t.performance for t in eqprop])
                bl_avg = np.mean([t.performance for t in baseline])
                diff = ((eq_avg - bl_avg) / bl_avg * 100) if bl_avg != 0 else 0
                
                winner = "🔋 EqProp" if eq_avg > bl_avg else "⚡ BP"
                print(f"  {task}: EqProp={eq_avg:.4f}, BP={bl_avg:.4f} → {winner} ({diff:+.1f}%)")
        
        print("=" * 70)
    
    def status(self):
        """Show current status of all trials."""
        print("\n" + "=" * 70)
        print("  HYPEROPT STATUS")
        print("=" * 70)
        
        all_trials = self.db.get_trials()
        if not all_trials:
            print("  No trials found.")
            return
        
        # Group by task
        by_task = defaultdict(list)
        for t in all_trials:
            by_task[t.task].append(t)
        
        for task, trials in sorted(by_task.items()):
            eq_trials = [t for t in trials if t.algorithm == "eqprop"]
            bl_trials = [t for t in trials if t.algorithm == "bp"]
            
            eq_complete = sum(1 for t in eq_trials if t.status == "complete")
            bl_complete = sum(1 for t in bl_trials if t.status == "complete")
            
            print(f"\n📋 {task}:")
            print(f"   EqProp: {eq_complete}/{len(eq_trials)} complete")
            print(f"   Baseline: {bl_complete}/{len(bl_trials)} complete")
            
            if eq_complete > 0 and bl_complete > 0:
                eq_perfs = [t.performance for t in eq_trials if t.status == "complete"]
                bl_perfs = [t.performance for t in bl_trials if t.status == "complete"]
                
                eq_best = max(eq_perfs) if eq_perfs else 0
                bl_best = max(bl_perfs) if bl_perfs else 0
                
                print(f"   Best EqProp: {eq_best:.4f}")
                print(f"   Best Baseline: {bl_best:.4f}")
        
        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Competitive Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python hyperopt_engine.py --smoke-test
  
  # Rapid campaign (fast feedback)
  python hyperopt_engine.py --campaign --rapid
  
  # Full optimization on specific task
  python hyperopt_engine.py --task cartpole --n-trials 50 --epochs 5
  
  # Multi-task campaign
  python hyperopt_engine.py --campaign --tasks mnist fashion cartpole parity
  
  # Generate report from existing results
  python hyperopt_engine.py --report
  
  # Show status of all trials
  python hyperopt_engine.py --status

Supported Tasks:
  Classification: mnist, fashion, cifar10, svhn
  Algorithmic:    parity, parity_12, copy, addition
  RL:             cartpole, acrobot, mountaincar, lunarlander
  Memory:         memory
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke-test", action="store_true",
                           help="Quick smoke test (2 trials, 1 epoch)")
    mode_group.add_argument("--campaign", action="store_true",
                           help="Run research campaign across multiple tasks")
    mode_group.add_argument("--report", action="store_true",
                           help="Generate report from existing results")
    mode_group.add_argument("--status", action="store_true",
                           help="Show status of all trials")
    
    # Task selection
    parser.add_argument("--task", type=str, default="mnist",
                       help="Single task to optimize on")
    parser.add_argument("--tasks", type=str, nargs="+",
                       default=["mnist", "fashion", "cartpole", "parity"],
                       help="Tasks for campaign mode")
    
    # Optimization settings
    parser.add_argument("--n-trials", type=int, default=10,
                       help="Number of trials per algorithm")
    parser.add_argument("--strategy", type=str, default="random",
                       choices=["grid", "random"],
                       help="Search strategy")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs per trial")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Random seeds to use")
    
    # Speed/quality tradeoffs
    parser.add_argument("--rapid", action="store_true",
                       help="Rapid mode: fewer epochs, smaller configs, faster feedback")
    parser.add_argument("--headless", action="store_true",
                       help="Suppress training output (still shows progress)")
    
    # Configuration
    parser.add_argument("--config", type=str, default="validation_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    engine = HyperOptEngine(args.config)
    
    if args.smoke_test:
        engine.smoke_test(n_trials=2, task=args.task)
    elif args.campaign:
        engine.run_campaign(
            tasks=args.tasks,
            n_trials=args.n_trials,
            strategy=args.strategy,
            seeds=args.seeds,
            epochs=args.epochs,
            rapid=args.rapid
        )
    elif args.report:
        engine.report(task=args.task if args.task != "mnist" else None)
    elif args.status:
        engine.status()
    else:
        engine.run(
            task=args.task,
            n_trials=args.n_trials,
            strategy=args.strategy,
            seeds=args.seeds,
            epochs=args.epochs,
            headless=args.headless
        )


if __name__ == "__main__":
    main()

