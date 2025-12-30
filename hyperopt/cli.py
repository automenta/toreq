from .engine import HyperOptEngine, HyperOptDB
from .evaluator import CostAwareEvaluator
import argparse

def main():
    parser = argparse.ArgumentParser(description="TorEqProp Competitive Hyperparameter Optimization Engine")
    
    # Modes
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument("--campaign", action="store_true", help="Run full multi-task campaign")
    parser.add_argument("--ultra-fast", action="store_true", help="Use ultra-fast settings for smoke test")
    
    # Configuration
    parser.add_argument("--task", type=str, default="mnist", 
                       help="Task to optimize (mnist, fashion, cifar10, parity, etc.)")
    parser.add_argument("--strategy", type=str, default="random",
                       choices=["grid", "random", "sobol", "lhs"],
                       help="Sampling strategy")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of trials per algorithm")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Epochs per trial")
    parser.add_argument("--time-budget", type=float, default=None,
                       help="Time budget in seconds per trial (Fair Comparison Mode)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Random seeds")
    parser.add_argument("--config", type=str, default="validation_config.yaml",
                       help="Path to configuration file")
    
    # Filters for reporting
    parser.add_argument("--algo", type=str, choices=["eqprop", "bp"],
                       help="Filter report by algorithm")
    
    # Flags
    parser.add_argument("--headless", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    engine = HyperOptEngine(config_path=args.config)
    
    if args.report:
        engine._print_analysis(args.task)
        return

    if args.smoke_test:
        print("🧪 SMOKE TEST MODE")
        if args.ultra_fast:
            print("⚡ ULTRA-FAST: d_model=8, micro task, <10s per trial, single seed")
            # Override search space for speed
            engine.eqprop_space.d_model = [8, 16]
            engine.eqprop_space.max_iters = [10]
            engine.baseline_space.d_model = [8, 16]
            
            # Should use a micro task
            if args.task == "mnist": # default
                args.task = "xor"
                
            engine.run(task=args.task, n_trials=2, strategy=args.strategy, 
                      epochs=1, seeds=[0], headless=args.headless)
        else:
            engine.run(task=args.task, n_trials=2, strategy=args.strategy, 
                      epochs=1, seeds=[0], headless=args.headless)
        return

    if args.campaign:
        print("🚀 STARTING CAMPAIGN MODE")
        tasks = ["xor", "xor3", "tiny_lm", "mnist"]
        for t in tasks:
            print(f"\n>> Running Task: {t}")
            engine.run(task=t, n_trials=args.n_trials, strategy=args.strategy,
                      epochs=args.epochs, seeds=args.seeds, headless=args.headless)
        return

    # Normal run
    engine.run(task=args.task, n_trials=args.n_trials, strategy=args.strategy,
              epochs=args.epochs, seeds=args.seeds, headless=args.headless, 
              time_budget=args.time_budget)

if __name__ == "__main__":
    main()
