#!/usr/bin/env python3
"""
TorEqProp Autonomous Validation Engine - Complete Multi-Phase Version

Validates ALL experiment types:
- Phase 1: Classification (MNIST, Fashion, CIFAR-10, SVHN)
- Phase 2: Algorithmic (parity, copy, addition)
- Phase 3: Reinforcement Learning (CartPole, Acrobot, MountainCar, LunarLander)
- Phase 4: Extended Training (high-accuracy push)
- Phase 5: Memory Profiling

Usage:
    python validation_engine.py              # Run all phases
    python validation_engine.py --phase 1    # Classification only
    python validation_engine.py --phase 3    # RL only
    python validation_engine.py --status     # Show progress
"""

import argparse
import subprocess
import sys
import os
import time
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

from validation_db import ValidationDB, ExperimentRun, generate_experiment_id
from statistics import StatisticalAnalyzer, FairnessChecker, ComparisonResult
from readme_updater import ReadmeUpdater


@dataclass
class ExperimentSpec:
    """Specification for a single experiment."""
    experiment_id: str
    phase: str               # classification, algorithmic, rl, extended, memory
    name: str                # Dataset/task/environment name
    algorithm: str           # eqprop or bp
    seed: int
    command: str
    success_threshold: float
    metric_name: str
    metric_pattern: str
    priority: int


class MultiPhaseScheduler:
    """Schedules experiments across all phases."""
    
    def __init__(self, config: dict, db: ValidationDB):
        self.config = config
        self.db = db
    
    def get_all_specs(self) -> List[ExperimentSpec]:
        """Generate all experiment specifications."""
        specs = []
        
        # Phase 1: Classification
        if self.config.get("classification", {}).get("enabled", True):
            specs.extend(self._get_classification_specs())
        
        # Phase 2: Algorithmic
        if self.config.get("algorithmic", {}).get("enabled", True):
            specs.extend(self._get_algorithmic_specs())
        
        # Phase 3: RL
        if self.config.get("rl", {}).get("enabled", True):
            specs.extend(self._get_rl_specs())
        
        # Phase 4: Extended
        if self.config.get("extended", {}).get("enabled", True):
            specs.extend(self._get_extended_specs())
        
        # Phase 5: Memory
        if self.config.get("memory", {}).get("enabled", True):
            specs.extend(self._get_memory_specs())
        
        return specs
    
    def _get_classification_specs(self) -> List[ExperimentSpec]:
        specs = []
        cfg = self.config["classification"]
        seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
        metric_pattern = cfg.get("metric_pattern", r"Test Acc:\s*([\d.]+)")
        
        for dataset in cfg.get("datasets", []):
            for algo in cfg.get("algorithms", []):
                for seed in seeds:
                    cmd = algo["command_template"].format(
                        dataset=dataset["name"],
                        epochs=dataset["epochs"],
                        seed=seed
                    )
                    specs.append(ExperimentSpec(
                        experiment_id=f"cls_{dataset['name']}_{algo['name']}_s{seed}",
                        phase="classification",
                        name=dataset["name"],
                        algorithm=algo["name"],
                        seed=seed,
                        command=cmd,
                        success_threshold=dataset["success_threshold"],
                        metric_name="test_accuracy",
                        metric_pattern=metric_pattern,
                        priority=dataset["priority"]
                    ))
        return specs
    
    def _get_algorithmic_specs(self) -> List[ExperimentSpec]:
        specs = []
        cfg = self.config["algorithmic"]
        seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
        metric_pattern = cfg.get("metric_pattern", r"Test Accuracy:\s*([\d.]+)")
        
        for task in cfg.get("tasks", []):
            for algo in cfg.get("algorithms", []):
                for seed in seeds:
                    cmd = algo["command_template"].format(
                        task=task["task"],
                        seq_len=task.get("seq_len", 8),
                        epochs=task["epochs"],
                        seed=seed
                    )
                    # Handle addition task which uses n_digits instead of seq_len
                    if task["task"] == "addition":
                        cmd = cmd.replace("--seq-len", "--n-digits")
                        cmd = cmd.replace(str(task.get("seq_len", 8)), str(task.get("n_digits", 4)))
                    
                    specs.append(ExperimentSpec(
                        experiment_id=f"algo_{task['name']}_{algo['name']}_s{seed}",
                        phase="algorithmic",
                        name=task["name"],
                        algorithm=algo["name"],
                        seed=seed,
                        command=cmd,
                        success_threshold=task["success_threshold"],
                        metric_name="test_accuracy",
                        metric_pattern=metric_pattern,
                        priority=task["priority"]
                    ))
        return specs
    
    def _get_rl_specs(self) -> List[ExperimentSpec]:
        specs = []
        cfg = self.config["rl"]
        seeds = cfg.get("seeds", list(range(10)))
        metric_pattern = cfg.get("metric_pattern", r"Final Average Reward:\s*([\d.]+)")
        
        for env in cfg.get("environments", []):
            for algo in cfg.get("algorithms", []):
                for seed in seeds:
                    cmd = algo["command_template"].format(
                        env=env["name"],
                        episodes=env["episodes"],
                        seed=seed
                    )
                    specs.append(ExperimentSpec(
                        experiment_id=f"rl_{env['name'].lower().replace('-', '_')}_{algo['name']}_s{seed}",
                        phase="rl",
                        name=env["name"],
                        algorithm=algo["name"],
                        seed=seed,
                        command=cmd,
                        success_threshold=env["success_threshold"],
                        metric_name="avg_reward",
                        metric_pattern=metric_pattern,
                        priority=env["priority"]
                    ))
        return specs
    
    def _get_extended_specs(self) -> List[ExperimentSpec]:
        specs = []
        cfg = self.config.get("extended", {})
        if not cfg.get("enabled", True):
            return specs
            
        seeds = cfg.get("seeds", [0, 1, 2])
        
        for exp in cfg.get("experiments", []):
            for algo in cfg.get("algorithms", []):
                for seed in seeds:
                    cmd = algo["command_template"].format(
                        dataset=exp["dataset"],
                        epochs=exp["epochs"],
                        d_model=exp["d_model"],
                        seed=seed
                    )
                    specs.append(ExperimentSpec(
                        experiment_id=f"ext_{exp['name']}_{algo['name']}_s{seed}",
                        phase="extended",
                        name=exp["name"],
                        algorithm=algo["name"],
                        seed=seed,
                        command=cmd,
                        success_threshold=exp["success_threshold"],
                        metric_name="test_accuracy",
                        metric_pattern=r"Test Acc:\s*([\d.]+)",
                        priority=exp["priority"]
                    ))
        return specs
    
    def _get_memory_specs(self) -> List[ExperimentSpec]:
        specs = []
        cfg = self.config.get("memory", {})
        if not cfg.get("enabled", True):
            return specs
            
        seeds = cfg.get("seeds", [0, 1, 2])
        cmd_template = cfg.get("command_template", "python profile_memory.py --d-model {d_model} --max-iters 100 --seed {seed}")
        metric_pattern = cfg.get("metric_pattern", r"Ratio\s*([\d.]+)")
        
        for size in cfg.get("model_sizes", []):
            for seed in seeds:
                cmd = cmd_template.format(d_model=size["d_model"], seed=seed)
                specs.append(ExperimentSpec(
                    experiment_id=f"mem_{size['name']}_s{seed}",
                    phase="memory",
                    name=size["name"],
                    algorithm="comparison",  # Memory compares both in one run
                    seed=seed,
                    command=cmd,
                    success_threshold=1.0,  # Ratio < 1 means EqProp uses less memory
                    metric_name="memory_ratio",
                    metric_pattern=metric_pattern,
                    priority=size["priority"]
                ))
        return specs
    
    def get_next_experiment(self, phase_filter: Optional[List[str]] = None) -> Optional[ExperimentSpec]:
        """Get next experiment to run based on priority and gaps."""
        all_specs = self.get_all_specs()
        
        # Filter by phase if specified
        if phase_filter:
            all_specs = [s for s in all_specs if s.phase in phase_filter]
        
        # Sort by priority, then by phase order
        phase_order = {"classification": 1, "algorithmic": 2, "rl": 3, "extended": 4, "memory": 5}
        all_specs.sort(key=lambda s: (phase_order.get(s.phase, 99), s.priority, s.seed))
        
        # Find first incomplete experiment
        for spec in all_specs:
            run = self.db.get_run(spec.experiment_id)
            if run is None or run.status not in ("complete", "running"):
                return spec
        
        return None  # All done!
    
    def get_progress(self, phase_filter: Optional[List[str]] = None) -> Dict:
        """Get progress across all phases."""
        all_specs = self.get_all_specs()
        
        if phase_filter:
            all_specs = [s for s in all_specs if s.phase in phase_filter]
        
        total = len(all_specs)
        completed = sum(1 for s in all_specs 
                       if self.db.get_run(s.experiment_id) 
                       and self.db.get_run(s.experiment_id).status == "complete")
        
        # By phase
        phases = {}
        for phase in ["classification", "algorithmic", "rl", "extended", "memory"]:
            phase_specs = [s for s in all_specs if s.phase == phase]
            phase_complete = sum(1 for s in phase_specs 
                                if self.db.get_run(s.experiment_id) 
                                and self.db.get_run(s.experiment_id).status == "complete")
            if phase_specs:
                phases[phase] = {"completed": phase_complete, "total": len(phase_specs)}
        
        return {
            "overall_progress": completed / total if total > 0 else 1.0,
            "completed": completed,
            "total": total,
            "gaps": total - completed,
            "phases": phases
        }


class ExperimentExecutor:
    """Executes experiments and captures results."""
    
    def __init__(self, config: dict, db: ValidationDB):
        self.config = config
        self.db = db
        self.logs_dir = Path(config["output"]["logs_dir"])
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, spec: ExperimentSpec, callback=None) -> ExperimentRun:
        """Execute an experiment and return results."""
        
        # Create run record
        run = ExperimentRun(
            experiment_id=spec.experiment_id,
            algorithm=spec.algorithm,
            environment=spec.name,
            seed=spec.seed,
            timestamp=datetime.now().isoformat(),
            config={"command": spec.command, "phase": spec.phase},
            status="running"
        )
        self.db.add_run(run)
        
        # Prepare log file
        log_path = self.logs_dir / f"{spec.experiment_id}.log"
        
        start_time = time.time()
        
        try:
            # Run with unbuffered output
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                spec.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                if callback:
                    callback(line)
            
            process.wait()
            walltime = time.time() - start_time
            
            output = "".join(output_lines)
            
            # Save log
            with open(log_path, "w") as f:
                f.write(f"Command: {spec.command}\n")
                f.write(f"Duration: {walltime:.1f}s\n")
                f.write(f"Exit code: {process.returncode}\n")
                f.write("=" * 70 + "\n")
                f.write(output)
            
            # Extract metrics
            metric_value = self._extract_metric(output, spec.metric_pattern)
            solved = self._check_solved(output, spec)
            
            # Update run record
            run.primary_metric = metric_value
            run.secondary_metrics = {
                "solved": 1.0 if solved else 0.0,
                spec.metric_name: metric_value
            }
            run.solved = solved
            run.walltime_seconds = walltime
            run.status = "complete" if process.returncode == 0 else "failed"
            run.log_path = str(log_path)
            
            if process.returncode != 0:
                run.error = f"Exit code: {process.returncode}"
            
        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.walltime_seconds = time.time() - start_time
        
        self.db.add_run(run)
        return run
    
    def _extract_metric(self, output: str, pattern: str) -> float:
        """Extract metric from output using pattern."""
        try:
            # Try the specified pattern
            match = re.search(pattern, output)
            if match:
                return float(match.group(1))
            
            # Fallback patterns
            fallbacks = [
                r"Test Acc(?:uracy)?:\s*([\d.]+)",
                r"Final Average Reward:\s*([\d.]+)",
                r"Best (?:Average )?(?:Reward|Accuracy):\s*([\d.]+)",
                r"avg_reward:\s*([\d.]+)",
            ]
            for fb in fallbacks:
                match = re.search(fb, output, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except:
            pass
        return 0.0
    
    def _check_solved(self, output: str, spec: ExperimentSpec) -> bool:
        """Check if experiment was solved."""
        if "SOLVED" in output.upper():
            return True
        
        metric = self._extract_metric(output, spec.metric_pattern)
        if spec.phase == "rl":
            return metric >= spec.success_threshold
        else:
            return metric >= spec.success_threshold


class ValidationEngine:
    """Main validation engine orchestrator."""
    
    def __init__(self, config_path: str = "validation_config.yaml"):
        self.config = self._load_config(config_path)
        self.db = ValidationDB(self.config["output"]["results_db"])
        self.scheduler = MultiPhaseScheduler(self.config, self.db)
        self.executor = ExperimentExecutor(self.config, self.db)
        self.analyzer = StatisticalAnalyzer(
            significance_level=self.config["statistics"]["significance_level"],
            min_effect_size=self.config["statistics"]["min_effect_size"],
            breakthrough_p_threshold=self.config["statistics"]["breakthrough_p_threshold"]
        )
        self.readme_updater = ReadmeUpdater(self.config["output"]["readme_path"])
    
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML."""
        with open(path) as f:
            return yaml.safe_load(f)
    
    def run(self, headless: bool = False, max_experiments: int = 0, 
            phases: Optional[List[str]] = None):
        """Main run loop."""
        print("\n" + "=" * 70)
        print("  TorEqProp Autonomous Validation Engine v2.0")
        print("  Multi-Phase: Classification | Algorithmic | RL | Extended | Memory")
        print("=" * 70)
        
        # Map phase numbers to names
        phase_map = {
            "1": "classification", "classification": "classification",
            "2": "algorithmic", "algorithmic": "algorithmic",
            "3": "rl", "rl": "rl",
            "4": "extended", "extended": "extended",
            "5": "memory", "memory": "memory"
        }
        phase_filter = None
        if phases:
            phase_filter = [phase_map.get(str(p).lower(), p) for p in phases]
            print(f"  Filtering to phases: {phase_filter}")
        
        # Show current status
        self._print_status(phase_filter)
        
        experiments_run = 0
        
        while True:
            # Check for next experiment
            spec = self.scheduler.get_next_experiment(phase_filter)
            
            if spec is None:
                print("\n✅ Validation complete! All experiments finished.")
                break
            
            if max_experiments > 0 and experiments_run >= max_experiments:
                print(f"\n⏸️  Stopping after {max_experiments} experiments (--max-experiments)")
                break
            
            # Run experiment
            print(f"\n{'='*70}")
            print(f"🚀 [{spec.phase.upper()}] Starting: {spec.experiment_id}")
            print(f"   Name: {spec.name}")
            print(f"   Algorithm: {spec.algorithm}")
            print(f"   Seed: {spec.seed}")
            print(f"   Command: {spec.command[:70]}...")
            print("=" * 70)
            
            def output_callback(line):
                if not headless:
                    print(line, end="", flush=True)
            
            run = self.executor.run(spec, callback=output_callback)
            experiments_run += 1
            
            # Report result
            status_icon = "✅" if run.status == "complete" else "❌"
            print(f"\n{status_icon} {run.experiment_id}: {run.primary_metric:.4f}")
            print(f"   Duration: {run.walltime_seconds:.1f}s")
            if run.solved:
                print(f"   🎉 SUCCESS - met threshold!")
            
            # Update statistics and README periodically
            if experiments_run % 5 == 0:
                self._update_validated_results()
            
            time.sleep(0.5)
        
        # Final report
        self._update_validated_results()
        self._print_final_report(phase_filter)
    
    def _print_status(self, phase_filter: Optional[List[str]] = None):
        """Print current validation status."""
        progress = self.scheduler.get_progress(phase_filter)
        
        print(f"\n📊 Current Status:")
        print(f"   Total experiments: {progress['completed']}/{progress['total']} ({progress['overall_progress']:.0%})")
        
        print(f"\n📈 Phase Progress:")
        phase_names = {
            "classification": "Classification",
            "algorithmic": "Algorithmic",
            "rl": "RL",
            "extended": "Extended",
            "memory": "Memory"
        }
        for phase, p in progress["phases"].items():
            pct = p["completed"] / p["total"] * 100 if p["total"] > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            status = "✅" if pct == 100 else "🔄" if pct > 0 else "⏳"
            print(f"   {status} {phase_names.get(phase, phase):15} [{bar}] {p['completed']}/{p['total']}")
    
    def _update_validated_results(self):
        """Update validated results and README."""
        results = {}
        stats_config = self.config["statistics"]
        min_seeds = stats_config.get("min_seeds_for_validation", 3)
        
        # Collect results by (phase, name)
        all_specs = self.scheduler.get_all_specs()
        groups = {}
        for spec in all_specs:
            key = (spec.phase, spec.name)
            if key not in groups:
                groups[key] = {"eqprop": [], "bp": []}
            
            run = self.db.get_run(spec.experiment_id)
            if run and run.status == "complete":
                if spec.algorithm in groups[key]:
                    groups[key][spec.algorithm].append(run.primary_metric)
        
        # Compute statistics where we have enough data
        for (phase, name), metrics in groups.items():
            eqprop_vals = metrics["eqprop"]
            bp_vals = metrics["bp"]
            
            if len(eqprop_vals) >= min_seeds and len(bp_vals) >= min_seeds:
                result = self.analyzer.compare(eqprop_vals, bp_vals)
                results[f"{phase}/{name}"] = result
                
                if result.is_breakthrough:
                    print(f"\n🎯 BREAKTHROUGH: {phase}/{name}")
                    print(f"   EqProp: {result.algo1_mean:.2f}±{result.algo1_std:.2f}")
                    print(f"   BP: {result.algo2_mean:.2f}±{result.algo2_std:.2f}")
                    print(f"   Improvement: {result.improvement_pct:+.1f}% (p={result.p_value:.4f})")
        
        if results:
            progress = self.scheduler.get_progress()
            self.readme_updater.update(results, {
                "total_experiments": progress["total"],
                "completed": progress["completed"]
            })
    
    def _print_final_report(self, phase_filter: Optional[List[str]] = None):
        """Print final validation report."""
        print("\n" + "=" * 70)
        print("  VALIDATION REPORT")
        print("=" * 70)
        
        self._print_status(phase_filter)
        
        # Get all comparisons
        print("\n📊 Statistical Comparisons:")
        
        all_specs = self.scheduler.get_all_specs()
        groups = {}
        for spec in all_specs:
            key = (spec.phase, spec.name)
            if key not in groups:
                groups[key] = {"eqprop": [], "bp": []}
            
            run = self.db.get_run(spec.experiment_id)
            if run and run.status == "complete":
                if spec.algorithm in groups[key]:
                    groups[key][spec.algorithm].append(run.primary_metric)
        
        for (phase, name), metrics in sorted(groups.items()):
            eqprop_vals = metrics["eqprop"]
            bp_vals = metrics["bp"]
            
            if eqprop_vals and bp_vals:
                result = self.analyzer.compare(eqprop_vals, bp_vals)
                status = "🏆" if result.is_breakthrough else "✅" if result.is_significant else "📊"
                print(f"\n{status} {phase}/{name}:")
                print(f"   EqProp: {result.algo1_mean:.2f}±{result.algo1_std:.2f} (n={result.algo1_n})")
                print(f"   BP:     {result.algo2_mean:.2f}±{result.algo2_std:.2f} (n={result.algo2_n})")
                print(f"   Δ: {result.improvement_pct:+.1f}%, p={result.p_value:.4f}, d={result.cohens_d:.2f}")
        
        print("\n" + "=" * 70)
    
    def status(self, phases: Optional[List[str]] = None):
        """Show current status only."""
        print("\n" + "=" * 70)
        print("  TorEqProp Validation Status")
        print("=" * 70)
        
        phase_map = {"1": "classification", "2": "algorithmic", "3": "rl", "4": "extended", "5": "memory"}
        phase_filter = [phase_map.get(str(p), p) for p in phases] if phases else None
        
        self._print_status(phase_filter)
        self._print_final_report(phase_filter)


def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Autonomous Validation Engine - All Phases",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--headless", action="store_true",
                       help="Run without showing experiment output")
    parser.add_argument("--status", action="store_true",
                       help="Show current status only")
    parser.add_argument("--phase", nargs="+", 
                       help="Run specific phases (1-5 or classification/algorithmic/rl/extended/memory)")
    parser.add_argument("--max-experiments", type=int, default=0,
                       help="Maximum experiments to run (0 = unlimited)")
    parser.add_argument("--config", type=str, default="validation_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    engine = ValidationEngine(args.config)
    
    if args.status:
        engine.status(args.phase)
    else:
        engine.run(headless=args.headless, max_experiments=args.max_experiments, 
                  phases=args.phase)


if __name__ == "__main__":
    main()
