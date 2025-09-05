#!/bin/bash
#
# Complete setup and run script for Causal Attention Geometry
# Hillary Danan's Research Implementation
#
# This script handles everything from setup to results

set -e  # Exit on error

echo "======================================================================"
echo "CAUSAL ATTENTION GEOMETRY - COMPLETE SETUP AND RUN"
echo "Hillary Danan's Research Implementation"
echo "======================================================================"

# Function to print colored output
print_status() {
    echo -e "\033[1;34m[STATUS]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Step 1: Setup directories
print_status "Setting up directory structure..."
mkdir -p src experiments results data
touch src/__init__.py experiments/__init__.py
print_success "Directories created"

# Step 2: Check Python version
print_status "Checking Python version..."
python3 --version
if [ $? -eq 0 ]; then
    print_success "Python3 found"
else
    print_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Step 3: Install dependencies
print_status "Installing dependencies..."
pip install -q --upgrade pip

# Core packages
pip install -q numpy scipy torch transformers spacy tqdm matplotlib pandas seaborn

# Download spacy model
print_status "Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm -q 2>/dev/null || {
    print_status "Retrying spaCy model download..."
    python3 -m spacy download en_core_web_sm
}
print_success "Dependencies installed"

# Step 4: Clean any corrupt results
print_status "Cleaning any previous results..."
rm -f results/*.json 2>/dev/null || true
rm -f results/*.txt 2>/dev/null || true
rm -f results/*.png 2>/dev/null || true
print_success "Clean workspace ready"

# Step 5: Test imports
print_status "Testing basic imports..."
python3 -c "
import numpy as np
import torch
import transformers
import spacy
print('✓ All core imports successful')
" || {
    print_error "Import test failed. Running debug..."
    python3 debug.py
    exit 1
}

# Step 6: Make sure source files are in place
print_status "Checking source files..."
if [ ! -f "src/causal_attention.py" ]; then
    print_error "src/causal_attention.py not found!"
    print_error "Please ensure all artifact files are saved to their locations"
    exit 1
fi

if [ ! -f "experiments/run_all.py" ]; then
    print_error "experiments/run_all.py not found!"
    print_error "Please ensure all experiment files are in experiments/"
    exit 1
fi
print_success "Source files verified"

# Step 7: Run experiments
print_status "Starting experiments..."
echo ""
echo "======================================================================"
echo "RUNNING EXPERIMENTS"
echo "======================================================================"

# Run with error handling
{
    python3 experiments/run_all.py --n-samples=64 --model=bert-base-uncased
} || {
    print_error "Experiments failed. Trying individual runs..."
    
    echo ""
    print_status "Running Hypothesis 1: Counterfactual Divergence..."
    python3 experiments/test_counterfactual.py --n-samples=64 || print_error "H1 failed"
    
    echo ""
    print_status "Running Hypothesis 2: Feedback Loop Density..."
    python3 experiments/test_feedback_loop.py --n-samples=64 || print_error "H2 failed"
    
    echo ""
    print_status "Running Hypothesis 3: Layer Specificity..."
    python3 experiments/test_layer_specificity.py --n-samples=64 || print_error "H3 failed"
    
    echo ""
    print_status "Generating summary from available results..."
    python3 experiments/run_all.py --skip-experiments || print_error "Summary generation failed"
}

# Step 8: Display results
echo ""
echo "======================================================================"
echo "RESULTS"
echo "======================================================================"

if [ -f "results/summary_statistics.txt" ]; then
    print_success "Experiments complete! Summary:"
    echo ""
    head -n 50 results/summary_statistics.txt
    echo ""
    echo "[... see full results in results/summary_statistics.txt ...]"
else
    print_error "No summary generated. Check individual result files:"
    ls -la results/ 2>/dev/null || echo "No results found"
fi

# Step 9: List output files
echo ""
echo "======================================================================"
echo "OUTPUT FILES"
echo "======================================================================"
ls -lh results/ 2>/dev/null || echo "No results directory"

echo ""
echo "======================================================================"
print_success "Script complete!"
echo ""
echo "To view full results:"
echo "  cat results/summary_statistics.txt"
echo ""
echo "To view visualizations:"
echo "  open results/h3_layer_effects_plot.png"
echo ""
echo "<4577> Dick Tracy's geometric detective work complete! 💕🚀"
echo "======================================================================