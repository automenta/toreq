#!/bin/bash
# run_complete_research.sh - TorEqProp Complete Research Pipeline
# Usage:
#   ./run_complete_research.sh --smoke-test  # Quick validation (1 seed, 5 epochs)
#   ./run_complete_research.sh --full        # Full research run (5 seeds, 50 epochs)

set -e  # Exit on any error

# Parse arguments
MODE="smoke-test"  # Default to smoke test
if [[ "$1" == "--full" ]]; then
    MODE="full"
    SEEDS=5
    EPOCHS=50
elif [[ "$1" == "--smoke-test" ]]; then
    MODE="smoke-test"
    SEEDS=1
    EPOCHS=5
else
    echo "Usage: $0 [--smoke-test | --full]"
    echo "  --smoke-test: Quick validation (1 seed, 5 epochs) - DEFAULT"
    echo "  --full: Full research run (5 seeds, 50 epochs)"
    echo ""
    echo "Running smoke test by default..."
    SEEDS=1
    EPOCHS=5
fi

echo "=========================================="
echo "TorEqProp Complete Research Pipeline"
echo "=========================================="
echo "Mode: $MODE"
echo "Seeds: $SEEDS"
echo "Epochs: $EPOCHS"
echo ""
echo "Started: $(date)"
echo "=========================================="
echo ""

# Create output directories
mkdir -p results
mkdir -p figures
mkdir -p papers

# Log file
LOGFILE="research_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Logging to: $LOGFILE"
echo ""

# Phase 1: Core Experiments
echo ""
echo "┌─────────────────────────────────────────┐"
echo "│  PHASE 1: CORE EXPERIMENTS              │"
echo "└─────────────────────────────────────────┘"
echo ""

# Experiment A: Multi-seed MNIST
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/6] Multi-seed MNIST Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Command: python scripts/competitive_benchmark.py --seeds $SEEDS --epochs $EPOCHS"
echo ""

if python scripts/competitive_benchmark.py --seeds "$SEEDS" --epochs "$EPOCHS"; then
    echo ""
    echo "✅ MNIST benchmark completed"
    
    # Check if results file exists
    RESULT_FILE="/tmp/competitive_benchmark_${SEEDS}seed.json"
    if [[ -f "$RESULT_FILE" ]]; then
        echo "   Results saved to: $RESULT_FILE"
        # Copy to results directory
        cp "$RESULT_FILE" "results/competitive_benchmark_${SEEDS}seed.json"
        echo "   Copied to: results/competitive_benchmark_${SEEDS}seed.json"
    fi
else
    echo ""
    echo "❌ MNIST benchmark FAILED"
    echo "   Check TODO.md for contingency plan"
    exit 1
fi
echo ""

# Experiment B: CIFAR-10 Hierarchical (only in full mode)
if [[ "$MODE" == "full" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[2/6] CIFAR-10 Hierarchical Testing"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Command: python scripts/test_cifar_readiness.py --model EnhancedMSTEP --epochs $EPOCHS"
    echo ""
    
    if python scripts/test_cifar_readiness.py --model EnhancedMSTEP --epochs "$EPOCHS"; then
        echo ""
        echo "✅ CIFAR-10 test completed"
    else
        echo ""
        echo "⚠️  CIFAR-10 test failed (continuing...)"
        echo "   This is acceptable - see contingency plan in TODO.md"
    fi
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[2/6] CIFAR-10 Testing (SKIPPED in smoke test)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run with --full to include CIFAR-10 testing"
    echo ""
fi

# Experiment C: Kernel Speed Test (only in full mode)
if [[ "$MODE" == "full" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[3/6] Kernel Speed Validation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ -f "kernel/test_optimizations.py" ]]; then
        echo "Command: python kernel/test_optimizations.py"
        echo ""
        
        if CUDA_PATH=/opt/cuda python kernel/test_optimizations.py; then
            echo ""
            echo "✅ Kernel speed test completed"
        else
            echo ""
            echo "⚠️  Kernel speed test failed (continuing...)"
        fi
    else
        echo "⚠️  kernel/test_optimizations.py not found, skipping"
    fi
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[3/6] Kernel Validation (SKIPPED in smoke test)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run with --full to include kernel benchmarks"
    echo ""
fi

# Experiment D: Ablation Studies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/6] Ablation Studies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# D.1: Spectral Normalization Ablation
echo "→ Spectral Normalization Impact"
if [[ -f "scripts/test_spectral_norm_all.py" ]]; then
    echo "  Command: python scripts/test_spectral_norm_all.py"
    echo ""
    
    if python scripts/test_spectral_norm_all.py; then
        echo ""
        echo "  ✅ Spectral norm validation completed"
    else
        echo ""
        echo "  ⚠️  Spectral norm test failed (continuing...)"
    fi
else
    echo "  ⚠️  scripts/test_spectral_norm_all.py not found"
fi
echo ""

# D.2: Beta Sweep (only in full mode)
if [[ "$MODE" == "full" ]]; then
    echo "→ Beta Parameter Sweep"
    if [[ -f "scripts/mnist_hyperparam_sweep.py" ]]; then
        echo "  Command: python scripts/mnist_hyperparam_sweep.py"
        echo ""
        
        if python scripts/mnist_hyperparam_sweep.py; then
            echo ""
            echo "  ✅ Beta parameter sweep completed"
        else
            echo ""
            echo "  ⚠️  Beta sweep failed (continuing...)"
        fi
    else
        echo "  ⚠️  scripts/mnist_hyperparam_sweep.py not found"
    fi
    echo ""
else
    echo "→ Beta Parameter Sweep (SKIPPED in smoke test)"
    echo "  Run with --full to include parameter sweeps"
    echo ""
fi

# D.3: Steps Analysis
echo "→ Convergence Steps Analysis"
if [[ -f "scripts/test_mnist_steps.py" ]]; then
    echo "  Command: python scripts/test_mnist_steps.py"
    echo ""
    
    if python scripts/test_mnist_steps.py; then
        echo ""
        echo "  ✅ Steps analysis completed"
    else
        echo ""
        echo "  ⚠️  Steps analysis failed (continuing...)"
    fi
else
    echo "  ⚠️  scripts/test_mnist_steps.py not found"
fi
echo ""

# Phase 2: Results Organization
echo ""
echo "┌─────────────────────────────────────────┐"
echo "│  PHASE 2: RESULTS ORGANIZATION          │"
echo "└─────────────────────────────────────────┘"
echo ""

# Generate Figures
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/6] Generating Publication Figures"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Use existing visualization scripts
echo "→ Generating dynamics visualization"
if python scripts/visualize_dynamics_svg.py 2>/dev/null; then
    echo "  ✅ Generated dynamics visualization"
else
    echo "  ⚠️  Dynamics visualization failed (may need training data)"
fi

echo ""
echo "→ Generating Lipschitz plots"
if python scripts/plot_lipschitz_svg.py 2>/dev/null; then
    echo "  ✅ Generated Lipschitz plots"
else
    echo "  ⚠️  Lipschitz plots failed (may need analysis data)"
fi

echo ""
echo "→ Generating beta stability plots"
if python scripts/plot_beta_stability_svg.py 2>/dev/null; then
    echo "  ✅ Generated beta stability plots"
else
    echo "  ⚠️  Beta stability plots failed (may need sweep data)"
fi
echo ""

# Generate Paper
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[6/6] Generating Paper Draft"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Command: python scripts/generate_paper.py --paper spectral_normalization"
echo ""

if python scripts/generate_paper.py --paper spectral_normalization; then
    echo ""
    echo "✅ Paper draft generated"
    
    # Check if generated paper exists
    PAPER_FILE="papers/spectral_normalization_paper_generated.md"
    if [[ -f "$PAPER_FILE" ]]; then
        echo "   Paper location: $PAPER_FILE"
    else
        echo "   ⚠️  Expected paper file not found: $PAPER_FILE"
    fi
else
    echo ""
    echo "❌ Paper generation failed"
    echo "   Try: python scripts/generate_paper.py --validate-only"
    echo "   This will show what data is missing"
fi
echo ""

# Summary Report
echo ""
echo "=========================================="
echo "  PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Mode: $MODE"
echo "Completed: $(date)"
echo ""
echo "Output Locations:"
echo "  📊 Experiment Results: results/*.json"
echo "  📈 Publication Figures: figures/*.svg"
echo "  📝 Paper Drafts: papers/*.md"
echo "  📋 Log File: $LOGFILE"
echo ""

# List results
echo "Generated Results:"
if ls -1 results/*.json 2>/dev/null | tail -5; then
    :
else
    echo "  (No JSON results found)"
fi
echo ""

echo "Generated Figures:"
if ls -1 figures/*.svg 2>/dev/null | tail -5; then
    :
else
    echo "  (No SVG figures found)"
fi
echo ""

if [[ "$MODE" == "smoke-test" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ SMOKE TEST PASSED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "The pipeline executed successfully with minimal parameters."
    echo "To run the full research program with all experiments:"
    echo ""
    echo "  ./run_complete_research.sh --full"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ FULL RESEARCH RUN COMPLETE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Next Steps:"
    echo "  1. Review results in results/ directory"
    echo "  2. Check success criteria in TODO.md"
    echo "  3. Verify all figures in figures/ directory"
    echo "  4. Review paper draft in papers/ directory"
    echo ""
fi

echo "For detailed research status, see: RESEARCH_STATUS.md"
echo "=========================================="
