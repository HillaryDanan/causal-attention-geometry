# Quick Start Guide - Causal Attention Geometry

## Step 1: Setup Project Structure

Make sure all files are in the correct locations:

```
causal-attention-geometry/
├── src/
│   ├── __init__.py
│   └── causal_attention.py          # Core analysis module
├── experiments/
│   ├── __init__.py
│   ├── test_counterfactual.py       # Hypothesis 1 test
│   ├── test_feedback_loop.py        # Hypothesis 2 test
│   ├── test_layer_specificity.py    # Hypothesis 3 test
│   └── run_all.py                   # Run all experiments
├── results/                         # Will be created automatically
├── data/                            # Will be created automatically
├── requirements.txt
├── setup.py                         # Setup script
├── install_dependencies.py          # Dependency installer
├── test_imports.py                  # Test basic functionality
├── clean_and_run.sh                # Clean and run all
└── README.md
```

## Step 2: Install Dependencies

Run the automated installer:

```bash
python3 install_dependencies.py
```

Or manually install:

```bash
pip install numpy scipy torch transformers spacy tqdm matplotlib pandas seaborn
python3 -m spacy download en_core_web_sm
```

## Step 3: Test Installation

Verify everything is working:

```bash
python3 test_imports.py
```

You should see all green checkmarks (✓).

## Step 4: Run Experiments

### Option A: Run All Experiments (Recommended)

```bash
python3 experiments/run_all.py
```

### Option B: Run Individual Experiments

```bash
# Hypothesis 1: Counterfactual Divergence
python3 experiments/test_counterfactual.py --n-samples=64

# Hypothesis 2: Feedback Loop Density
python3 experiments/test_feedback_loop.py --n-samples=64

# Hypothesis 3: Layer Specificity
python3 experiments/test_layer_specificity.py --n-samples=64 --visualize
```

### Option C: Clean and Run Everything

```bash
bash clean_and_run.sh
```

## Step 5: View Results

Results are saved in the `results/` directory:

```bash
# View summary
cat results/summary_statistics.txt

# View individual results
cat results/h1_counterfactual_results.json
cat results/h2_feedback_results.json
cat results/h3_layer_analysis.json

# View visualization
open results/h3_layer_effects_plot.png
```

## Troubleshooting

### Issue: Module not found errors

```bash
# Ensure you're in the project root
cd causal-attention-geometry

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: CUDA/GPU errors

```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Memory errors

Use a smaller model:

```bash
python3 experiments/run_all.py --model distilbert-base-uncased
```

### Issue: JSON serialization errors

Clean results and retry:

```bash
rm -f results/*.json
python3 experiments/run_all.py
```

### Issue: spaCy model not found

```bash
python3 -m spacy download en_core_web_sm
```

## Parameters

- `--n-samples`: Number of samples (default: 64)
- `--model`: Model to test (default: bert-base-uncased)
- `--visualize`: Generate plots (for layer specificity)
- `--skip-experiments`: Only generate summary from existing results

## Expected Runtime

- Each experiment: 5-10 minutes
- Full suite: 15-30 minutes
- Depends on hardware and model size

## Success Indicators

✓ All three hypotheses tested  
✓ Statistical results with p-values and effect sizes  
✓ Clear interpretation of findings  
✓ Results saved to JSON and text files  
✓ Visualization generated for layer analysis

## Need Help?

If you encounter issues:

1. Check error messages carefully
2. Ensure all dependencies are installed
3. Verify file structure matches expected layout
4. Try running individual experiments first
5. Use test_imports.py to diagnose issues

---

<4577> Let's discover those geometric patterns! 💕🚀