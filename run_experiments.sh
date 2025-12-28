#!/bin/bash
# TorEqProp Experiment Runner
# Usage: ./run_experiments.sh [experiment_name]
# 
# Available experiments:
#   accuracy     - Run corrected β=0.25 training for best accuracy
#   multiseed    - Run 5-seed validation
#   memory       - Profile memory at different scales  
#   adaptive     - Analyze adaptive compute behavior
#   all          - Run all experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Experiment 1: Corrected β=0.25 Training
run_accuracy() {
    log "🎯 Running accuracy experiment (β=0.25 fixed)..."
    
    python train.py \
        --d-model 256 \
        --n-heads 8 \
        --d-ff 1024 \
        --beta 0.25 \
        --damping 0.8 \
        --lr 0.002 \
        --epochs 15 \
        --dropout 0.1 \
        --compile \
        2>&1 | tee logs/accuracy_beta025.log
    
    log "✅ Accuracy experiment complete. Results in logs/accuracy_beta025.log"
}

# Experiment 2: Multi-seed Validation
run_multiseed() {
    log "🎲 Running multi-seed validation (5 seeds)..."
    
    mkdir -p logs/multiseed
    
    for seed in 1 2 3 4 5; do
        log "  Seed $seed/5..."
        python train.py \
            --d-model 256 \
            --n-heads 8 \
            --d-ff 1024 \
            --beta 0.25 \
            --damping 0.8 \
            --lr 0.002 \
            --epochs 10 \
            --dropout 0.1 \
            --seed $seed \
            --compile \
            2>&1 | tee logs/multiseed/seed_${seed}.log
    done
    
    log "✅ Multi-seed validation complete. Results in logs/multiseed/"
    
    # Summary
    echo ""
    log "📊 Summary:"
    grep -h "Test Acc" logs/multiseed/*.log | tail -5
}

# Experiment 3: Memory Profiling
run_memory() {
    log "💾 Running memory profiling..."
    
    mkdir -p logs
    
    python profile_memory.py 2>&1 | tee logs/memory_profile.log
    
    log "✅ Memory profiling complete. Results in logs/memory_profile.log"
}

# Experiment 4: Adaptive Compute Analysis
run_adaptive() {
    log "🧠 Running adaptive compute analysis..."
    
    mkdir -p logs
    
    python analyze_adaptive_compute.py 2>&1 | tee logs/adaptive_compute.log
    
    log "✅ Adaptive compute analysis complete. Results in logs/adaptive_compute.log"
}

# Run gradient equivalence verification
run_gradient() {
    log "📐 Running gradient equivalence verification..."
    
    mkdir -p logs
    
    python test_gradient_equiv.py 2>&1 | tee logs/gradient_equiv.log
    
    log "✅ Gradient verification complete. Results in logs/gradient_equiv.log"
}

# Create logs directory
mkdir -p logs

# Main
case "${1:-all}" in
    accuracy)
        run_accuracy
        ;;
    multiseed)
        run_multiseed
        ;;
    memory)
        run_memory
        ;;
    adaptive)
        run_adaptive
        ;;
    gradient)
        run_gradient
        ;;
    all)
        log "🚀 Running all experiments..."
        run_accuracy
        run_multiseed
        run_memory
        run_adaptive
        log "🎉 All experiments complete!"
        ;;
    *)
        echo "Usage: $0 {accuracy|multiseed|memory|adaptive|gradient|all}"
        exit 1
        ;;
esac
