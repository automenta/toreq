#!/usr/bin/env python3
"""
TorEqProp Experiment Orchestrator

Turn-key discovery process that:
1. Intelligently allocates compute resources
2. Explores the application space fairly and thoroughly
3. Provides rapid, detailed feedback
4. Steers research toward maximum success

Usage:
    python run_discovery.py                    # Full discovery campaign
    python run_discovery.py --phase 1          # Run specific phase
    python run_discovery.py --quick            # Quick validation (1 epoch each)
    python run_discovery.py --dry-run          # Show what would run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import traceback


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    name: str
    command: str
    expected_time_min: float
    success_metric: str
    success_threshold: float
    phase: int
    priority: str  # HIGH, MEDIUM, LOW
    hypothesis: str
    
    
# All experiments organized by phase
EXPERIMENTS = [
    # Phase 1: Rapid Dataset Sweep
    ExperimentConfig(
        name="MNIST Rapid",
        command="python train.py --dataset mnist --rapid --epochs 3",
        expected_time_min=5,
        success_metric="test_accuracy",
        success_threshold=0.80,
        phase=1,
        priority="HIGH",
        hypothesis="Baseline validation - should achieve ~85% in 3 epochs"
    ),
    ExperimentConfig(
        name="FashionMNIST Rapid",
        command="python train.py --dataset fashion --rapid --epochs 3",
        expected_time_min=5,
        success_metric="test_accuracy",
        success_threshold=0.70,
        phase=1,
        priority="HIGH",
        hypothesis="Harder than MNIST - expect ~75% in 3 epochs"
    ),
    ExperimentConfig(
        name="CIFAR-10 Rapid",
        command="python train.py --dataset cifar10 --rapid --epochs 3",
        expected_time_min=8,
        success_metric="test_accuracy",
        success_threshold=0.35,
        phase=1,
        priority="HIGH",
        hypothesis="Significant complexity jump - expect ~45% if working"
    ),
    ExperimentConfig(
        name="SVHN Rapid",
        command="python train.py --dataset svhn --rapid --epochs 3",
        expected_time_min=8,
        success_metric="test_accuracy",
        success_threshold=0.40,
        phase=1,
        priority="MEDIUM",
        hypothesis="Real-world digits - expect ~50% if working"
    ),
    
    # Phase 2: Algorithmic Reasoning
    ExperimentConfig(
        name="Parity N=8",
        command="python train_algorithmic.py --task parity --seq-len 8 --epochs 10",
        expected_time_min=5,
        success_metric="test_accuracy",
        success_threshold=0.90,
        phase=2,
        priority="HIGH",
        hypothesis="Adaptive compute test - track iterations vs difficulty"
    ),
    ExperimentConfig(
        name="Parity N=12",
        command="python train_algorithmic.py --task parity --seq-len 12 --epochs 15",
        expected_time_min=8,
        success_metric="test_accuracy",
        success_threshold=0.85,
        phase=2,
        priority="MEDIUM",
        hypothesis="Longer sequences should need more iterations"
    ),
    ExperimentConfig(
        name="Copy Task",
        command="python train_algorithmic.py --task copy --seq-len 8 --epochs 5",
        expected_time_min=3,
        success_metric="test_accuracy",
        success_threshold=0.95,
        phase=2,
        priority="MEDIUM",
        hypothesis="Easy baseline - should converge quickly with uniform iterations"
    ),
    ExperimentConfig(
        name="Addition 4-digit",
        command="python train_algorithmic.py --task addition --n-digits 4 --epochs 20",
        expected_time_min=10,
        success_metric="test_accuracy",
        success_threshold=0.50,
        phase=2,
        priority="HIGH",
        hypothesis="Carry propagation is inherently sequential - test adaptive compute"
    ),
    
    # Phase 3: Reinforcement Learning
    ExperimentConfig(
        name="CartPole EqProp",
        command="python train_rl.py --env CartPole-v1 --episodes 500",
        expected_time_min=15,
        success_metric="avg_reward",
        success_threshold=195.0,
        phase=3,
        priority="HIGH",
        hypothesis="Can EqProp gradients solve classic control?"
    ),
    ExperimentConfig(
        name="CartPole BP Baseline",
        command="python train_rl.py --env CartPole-v1 --episodes 500 --use-bp",
        expected_time_min=10,
        success_metric="avg_reward",
        success_threshold=195.0,
        phase=3,
        priority="HIGH",
        hypothesis="BP baseline for comparison"
    ),
    
    # Phase 4: Accuracy Push
    ExperimentConfig(
        name="MNIST Extended (100 epochs)",
        command="python train.py --d-model 256 --beta 0.22 --epochs 100 --dropout 0.1 --compile",
        expected_time_min=180,
        success_metric="test_accuracy",
        success_threshold=0.945,
        phase=4,
        priority="HIGH",
        hypothesis="Extended training should reach 94.5%+ based on trajectory"
    ),
    ExperimentConfig(
        name="MNIST Scaled (d=512)",
        command="python train.py --d-model 512 --n-heads 16 --d-ff 2048 --beta 0.22 --epochs 50 --compile",
        expected_time_min=240,
        success_metric="test_accuracy",
        success_threshold=0.950,
        phase=4,
        priority="MEDIUM",
        hypothesis="Larger model capacity should push past 95%"
    ),
    
    # Phase 5: Memory Profiling
    ExperimentConfig(
        name="Memory Profile d=256",
        command="python profile_memory.py --d-model 256 --max-iters 100",
        expected_time_min=5,
        success_metric="memory_ratio",
        success_threshold=1.5,
        phase=5,
        priority="MEDIUM",
        hypothesis="Baseline memory measurement"
    ),
    ExperimentConfig(
        name="Memory Profile d=1024",
        command="python profile_memory.py --d-model 1024 --max-iters 100",
        expected_time_min=10,
        success_metric="memory_ratio",
        success_threshold=0.8,
        phase=5,
        priority="HIGH",
        hypothesis="O(1) advantage should emerge at scale"
    ),
    ExperimentConfig(
        name="Memory Profile d=2048",
        command="python profile_memory.py --d-model 2048 --max-iters 100",
        expected_time_min=20,
        success_metric="memory_ratio",
        success_threshold=0.5,
        phase=5,
        priority="HIGH",
        hypothesis="Clear O(1) memory advantage at large scale"
    ),
]


# ============================================================================
# Result Tracking
# ============================================================================

@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    name: str
    phase: int
    status: str  # SUCCESS, FAILURE, ERROR, SKIPPED
    metric_value: Optional[float]
    threshold: float
    passed: bool
    duration_sec: float
    timestamp: str
    output_log: str
    error: Optional[str] = None
    insights: Optional[str] = None


class ResultsTracker:
    """Track and display experiment results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        self.start_time = datetime.now()
        
    def add_result(self, result: ExperimentResult):
        self.results.append(result)
        self._save_results()
        self._print_status(result)
        
    def _save_results(self):
        """Save results to JSON for later analysis."""
        results_file = self.output_dir / "discovery_results.json"
        data = {
            "start_time": self.start_time.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results]
        }
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
            
    def _print_status(self, result: ExperimentResult):
        """Print formatted status update."""
        icon = "✅" if result.passed else "❌" if result.status == "FAILURE" else "⚠️"
        print(f"\n{'='*70}")
        print(f"{icon} {result.name} [{result.status}]")
        print(f"   Metric: {result.metric_value:.4f} vs threshold {result.threshold:.4f}")
        print(f"   Duration: {result.duration_sec:.1f}s")
        if result.insights:
            print(f"   Insight: {result.insights}")
        print(f"{'='*70}\n")
        
    def print_summary(self):
        """Print final summary of all experiments."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if r.status == "FAILURE")
        errors = sum(1 for r in self.results if r.status == "ERROR")
        
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        
        print("\n" + "="*70)
        print("📊 DISCOVERY CAMPAIGN SUMMARY")
        print("="*70)
        print(f"Total experiments: {total}")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⚠️  Errors: {errors}")
        print(f"Total duration: {duration:.1f} minutes")
        print()
        
        # Per-phase summary
        phases = sorted(set(r.phase for r in self.results))
        for phase in phases:
            phase_results = [r for r in self.results if r.phase == phase]
            phase_passed = sum(1 for r in phase_results if r.passed)
            print(f"Phase {phase}: {phase_passed}/{len(phase_results)} passed")
            
        # Key insights
        print("\n🔑 KEY INSIGHTS:")
        for r in self.results:
            if r.insights:
                icon = "✅" if r.passed else "❌"
                print(f"  {icon} {r.name}: {r.insights}")
                
        # Recommendations
        print("\n📋 RECOMMENDATIONS:")
        failed_high_priority = [r for r in self.results if not r.passed and r.status != "ERROR"]
        if failed_high_priority:
            print("  Investigate failures:")
            for r in failed_high_priority[:3]:
                print(f"    - {r.name}: got {r.metric_value:.4f}, needed {r.threshold:.4f}")
        
        passed_results = [r for r in self.results if r.passed]
        if passed_results:
            print("  Promising directions to scale:")
            for r in sorted(passed_results, key=lambda x: x.metric_value or 0, reverse=True)[:3]:
                print(f"    - {r.name}: {r.metric_value:.4f}")
                
        print("="*70 + "\n")
        
        # Save summary
        summary_file = self.output_dir / "discovery_summary.md"
        self._save_markdown_summary(summary_file)
        print(f"Full results saved to: {self.output_dir}/")
        
    def _save_markdown_summary(self, path: Path):
        """Save markdown summary."""
        with open(path, "w") as f:
            f.write("# Discovery Campaign Results\n\n")
            f.write(f"**Date**: {self.start_time.strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Results by Phase\n\n")
            phases = sorted(set(r.phase for r in self.results))
            for phase in phases:
                f.write(f"### Phase {phase}\n\n")
                f.write("| Experiment | Status | Metric | Threshold | Insight |\n")
                f.write("|------------|--------|--------|-----------|--------|\n")
                for r in self.results:
                    if r.phase == phase:
                        status = "✅" if r.passed else "❌"
                        metric = f"{r.metric_value:.4f}" if r.metric_value else "N/A"
                        insight = r.insights or "-"
                        f.write(f"| {r.name} | {status} | {metric} | {r.threshold:.4f} | {insight} |\n")
                f.write("\n")


# ============================================================================
# Experiment Runner
# ============================================================================

def parse_output_for_metrics(output: str, metric_name: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse experiment output for key metrics and insights."""
    metric_value = None
    insight = None
    
    lines = output.strip().split("\n")
    
    # Look for test accuracy
    if metric_name == "test_accuracy":
        for line in reversed(lines):
            if "Test Acc:" in line or "test/accuracy" in line:
                try:
                    # Extract number after "Acc:" or "accuracy:"
                    parts = line.split("Acc:")[-1].split("accuracy:")[-1]
                    value = float(parts.strip().split()[0].strip(","))
                    if value <= 1.0:
                        metric_value = value
                    else:
                        metric_value = value / 100  # Handle percentage
                    break
                except (ValueError, IndexError):
                    continue
                    
        # Generate insight based on accuracy
        if metric_value:
            if metric_value > 0.90:
                insight = "Strong performance"
            elif metric_value > 0.70:
                insight = "Moderate performance - may improve with scaling"
            elif metric_value > 0.50:
                insight = "Learning signal present - needs tuning"
            else:
                insight = "Weak signal - may need architecture changes"
                
    # Look for reward (RL)
    elif metric_name == "avg_reward":
        for line in reversed(lines):
            if "Average Reward:" in line or "avg_reward" in line:
                try:
                    parts = line.split(":")[-1]
                    metric_value = float(parts.strip().split()[0])
                    break
                except (ValueError, IndexError):
                    continue
                    
    # Look for memory ratio
    elif metric_name == "memory_ratio":
        for line in reversed(lines):
            if "Memory Ratio:" in line or "EqProp/BP:" in line:
                try:
                    parts = line.split(":")[-1]
                    metric_value = float(parts.strip().split()[0].replace("x", ""))
                    break
                except (ValueError, IndexError):
                    continue
                    
        if metric_value:
            if metric_value < 0.5:
                insight = "Clear O(1) advantage!"
            elif metric_value < 1.0:
                insight = "Memory advantage present"
            else:
                insight = "No memory advantage at this scale"
                
    return metric_value, insight


def run_experiment(config: ExperimentConfig, tracker: ResultsTracker, dry_run: bool = False) -> ExperimentResult:
    """Run a single experiment and track results."""
    print(f"\n{'='*70}")
    print(f"🚀 Starting: {config.name}")
    print(f"   Command: {config.command}")
    print(f"   Expected time: ~{config.expected_time_min} min")
    print(f"   Hypothesis: {config.hypothesis}")
    print(f"{'='*70}\n")
    
    if dry_run:
        return ExperimentResult(
            name=config.name,
            phase=config.phase,
            status="SKIPPED",
            metric_value=None,
            threshold=config.success_threshold,
            passed=False,
            duration_sec=0,
            timestamp=datetime.now().isoformat(),
            output_log="[DRY RUN]",
            insights="Dry run - not executed"
        )
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(
            config.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=config.expected_time_min * 60 * 3  # 3x timeout
        )
        
        duration = time.time() - start_time
        output = result.stdout + result.stderr
        
        # Save full log
        log_file = tracker.output_dir / f"{config.name.replace(' ', '_').lower()}.log"
        with open(log_file, "w") as f:
            f.write(f"Command: {config.command}\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write("="*70 + "\n")
            f.write(output)
            
        # Parse metrics
        metric_value, insight = parse_output_for_metrics(output, config.success_metric)
        
        # Determine success
        if result.returncode != 0:
            status = "ERROR"
            passed = False
            insight = f"Exit code {result.returncode}"
        elif metric_value is None:
            status = "ERROR"
            passed = False
            insight = "Could not parse metric from output"
        elif metric_value >= config.success_threshold:
            status = "SUCCESS"
            passed = True
        else:
            status = "FAILURE"
            passed = False
            
        return ExperimentResult(
            name=config.name,
            phase=config.phase,
            status=status,
            metric_value=metric_value,
            threshold=config.success_threshold,
            passed=passed,
            duration_sec=duration,
            timestamp=datetime.now().isoformat(),
            output_log=str(log_file),
            insights=insight
        )
        
    except subprocess.TimeoutExpired:
        return ExperimentResult(
            name=config.name,
            phase=config.phase,
            status="ERROR",
            metric_value=None,
            threshold=config.success_threshold,
            passed=False,
            duration_sec=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            output_log="",
            error="Timeout",
            insights="Experiment timed out - may need more time"
        )
        
    except Exception as e:
        return ExperimentResult(
            name=config.name,
            phase=config.phase,
            status="ERROR",
            metric_value=None,
            threshold=config.success_threshold,
            passed=False,
            duration_sec=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            output_log="",
            error=str(e),
            insights=f"Exception: {type(e).__name__}"
        )


# ============================================================================
# Main Orchestrator
# ============================================================================

def estimate_campaign_time(experiments: List[ExperimentConfig]) -> float:
    """Estimate total campaign time in minutes."""
    return sum(e.expected_time_min for e in experiments)


def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Discovery Campaign - Turn-key experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_discovery.py                    # Full campaign
  python run_discovery.py --phase 1          # Only Phase 1 (dataset sweep)
  python run_discovery.py --phase 1 2        # Phases 1 and 2
  python run_discovery.py --quick            # Quick validation (1 epoch)
  python run_discovery.py --dry-run          # Show what would run
  python run_discovery.py --priority HIGH    # Only HIGH priority experiments
        """
    )
    parser.add_argument("--phase", type=int, nargs="+", help="Run specific phase(s)")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode (1 epoch)")
    parser.add_argument("--dry-run", action="store_true", help="Show experiments without running")
    parser.add_argument("--priority", choices=["HIGH", "MEDIUM", "LOW"], help="Filter by priority")
    parser.add_argument("--output-dir", type=str, default="logs/discovery", help="Output directory")
    
    args = parser.parse_args()
    
    # Filter experiments
    experiments = EXPERIMENTS.copy()
    
    if args.phase:
        experiments = [e for e in experiments if e.phase in args.phase]
        
    if args.priority:
        experiments = [e for e in experiments if e.priority == args.priority]
        
    if args.quick:
        # Modify to run just 1 epoch for quick validation
        quick_experiments = []
        for e in experiments:
            if "epochs" in e.command:
                new_cmd = e.command.replace("--epochs 100", "--epochs 1")
                new_cmd = new_cmd.replace("--epochs 50", "--epochs 1")
                new_cmd = new_cmd.replace("--epochs 20", "--epochs 1")
                new_cmd = new_cmd.replace("--epochs 15", "--epochs 1")
                new_cmd = new_cmd.replace("--epochs 10", "--epochs 1")
                new_cmd = new_cmd.replace("--epochs 5", "--epochs 1")
                new_cmd = new_cmd.replace("--epochs 3", "--epochs 1")
                quick_experiments.append(ExperimentConfig(
                    name=e.name + " [QUICK]",
                    command=new_cmd,
                    expected_time_min=2,
                    success_metric=e.success_metric,
                    success_threshold=e.success_threshold * 0.5,  # Lower threshold
                    phase=e.phase,
                    priority=e.priority,
                    hypothesis=e.hypothesis
                ))
            else:
                quick_experiments.append(e)
        experiments = quick_experiments
        
    if not experiments:
        print("No experiments match the specified criteria.")
        return
        
    # Print campaign overview
    total_time = estimate_campaign_time(experiments)
    print("\n" + "="*70)
    print("🔬 TorEqProp Discovery Campaign")
    print("="*70)
    print(f"Experiments: {len(experiments)}")
    print(f"Estimated time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
    print(f"Output directory: {args.output_dir}")
    print()
    
    print("Experiments to run:")
    for i, e in enumerate(experiments, 1):
        print(f"  {i}. [{e.phase}] {e.name} ({e.priority}) ~{e.expected_time_min}min")
    print()
    
    if args.dry_run:
        print("[DRY RUN] Would run the above experiments. Use without --dry-run to execute.")
        return
        
    # Confirm before running
    print("Press Enter to start, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
        
    # Run experiments
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    tracker = ResultsTracker(output_dir)
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running {experiment.name}...")
        result = run_experiment(experiment, tracker, dry_run=False)
        tracker.add_result(result)
        
    # Print summary
    tracker.print_summary()


if __name__ == "__main__":
    main()
