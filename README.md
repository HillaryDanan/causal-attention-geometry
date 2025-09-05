# Causal Attention Geometry in Transformers

**Hillary Danan's Research Implementation**

Testing whether transformer models exhibit geometric patterns in attention when processing causal relationships.

## Overview

This repository implements three falsifiable hypotheses about causal attention geometry:

1. **Counterfactual Divergence**: Attention patterns diverge at causal intervention points (KL > 0.2)
2. **Feedback Loop Density**: Circular causation shows denser attention than linear (Cohen's d > 0.3)  
3. **Layer Specificity**: Middle layers (5-8) show strongest causal effects

## Installation

```bash
# Clone repository
git clone https://github.com/HillaryDanan/causal-attention-geometry.git
cd causal-attention-geometry

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for dependency parsing
python3 -m spacy download en_core_web_sm
```

## Quick Start

Run all experiments with default settings:

```bash
python3 experiments/run_all.py
```

Run individual experiments:

```bash
# Hypothesis 1: Counterfactual Divergence
python3 experiments/test_counterfactual.py --n-samples=64

# Hypothesis 2: Feedback Loop Density  
python3 experiments/test_feedback_loop.py --n-samples=64

# Hypothesis 3: Layer Specificity
python3 experiments/test_layer_specificity.py --n-samples=64 --visualize
```

## Project Structure

```
causal-attention-geometry/
├── src/
│   └── causal_attention.py          # Core analysis module
├── experiments/
│   ├── test_counterfactual.py       # H1: Counterfactual divergence
│   ├── test_feedback_loop.py        # H2: Feedback loop density
│   ├── test_layer_specificity.py    # H3: Layer specificity
│   └── run_all.py                   # Run all experiments
├── results/
│   ├── h1_counterfactual_results.json
│   ├── h2_feedback_results.json
│   ├── h3_layer_analysis.json
│   ├── h3_layer_effects_plot.png
│   ├── summary_statistics.txt
│   └── summary.json
├── data/                             # Dataset storage
├── requirements.txt
└── README.md
```

## Hypotheses Details

### Hypothesis 1: Counterfactual Divergence

**Claim**: Attention patterns diverge at causal intervention points when comparing factual vs counterfactual statements.

**Method**:
- Extract attention at dependency-parsed intervention points
- Calculate KL divergence between factual/counterfactual pairs
- Test against threshold (KL > 0.2)

**Expected**: Statistically significant divergence with medium effect size

### Hypothesis 2: Feedback Loop Density

**Claim**: Circular causation (A→B→C→A) produces denser attention patterns than linear causation (A→B→C).

**Method**:
- Compare attention density (off-diagonal weights)
- Three conditions: circular, linear, control
- Calculate Cohen's d between conditions

**Expected**: d > 0.3 between circular and linear conditions

### Hypothesis 3: Layer Specificity

**Claim**: Middle transformer layers (5-8) show strongest causal effects, following Tenney et al. (2019).

**Method**:
- Layer-wise analysis of causal vs non-causal texts
- Effect sizes per layer with Bonferroni correction
- ANOVA across layer groups (early/middle/late)

**Expected**: Peak effects in layers 5-8

## Statistical Rigor

All experiments include:
- **Power analysis**: N=64 for 80% power at specified effect sizes
- **Effect sizes**: Cohen's d reported for all comparisons
- **Multiple comparisons**: Bonferroni correction applied
- **Confidence intervals**: 95% CI for key metrics
- **Null hypothesis testing**: Clear falsifiable predictions

## Results Interpretation

### Success Metrics
- ✓ Hypothesis supported if statistical criteria met
- ✗ Null results equally valuable - constrain interpretability claims
- Mixed evidence indicates trends requiring larger samples

### Output Files
- `h1_counterfactual_results.json`: Divergence analysis
- `h2_feedback_results.json`: Density comparisons
- `h3_layer_analysis.json`: Layer-wise effects
- `summary_statistics.txt`: Human-readable summary
- `summary.json`: Machine-readable metrics

## Scientific Principles

1. **Correlation ≠ Causation**: Even when studying causation
2. **Falsifiability**: Clear predictions that can be wrong
3. **Null Results Matter**: Constraints on interpretability claims
4. **Replicability**: Raw data saved, random seeds fixed

## Related Work

This extends Hillary Danan's research on:
- Multi-geometric attention patterns
- Retroactive causality in language models  
- Cross-linguistic attention dynamics
- Embodied cognition without bodies

## Configuration Options

```python
# Model selection
--model bert-base-uncased  # Default
--model roberta-base       # Alternative

# Sample size (affects statistical power)
--n-samples 64   # Default (80% power)
--n-samples 128  # Higher power

# Visualization
--visualize      # Generate plots for layer analysis
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'spacy'`
```bash
pip install spacy
python3 -m spacy download en_core_web_sm
```

**Issue**: CUDA/GPU errors
```bash
# Run on CPU only
export CUDA_VISIBLE_DEVICES=""
```

**Issue**: Memory errors with large models
```bash
# Reduce batch size or use smaller model
--model distilbert-base-uncased
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{danan2025causal,
  author = {Danan, Hillary},
  title = {Causal Attention Geometry in Transformers},
  year = {2025},
  url = {https://github.com/HillaryDanan/causal-attention-geometry}
}
```

## Future Directions

1. **Larger Models**: Test GPT-style architectures
2. **Cross-linguistic**: Compare causal patterns across languages
3. **Temporal Dynamics**: Autoregressive causal processing
4. **Integration**: Connect with multi-geometric findings
5. **Retroactive**: Explore backward causal influences

## Contributing

Contributions welcome! Please ensure:
- Maintain statistical rigor
- Include power analysis
- Report effect sizes
- Document null results

## License

MIT License - See LICENSE file for details

## Acknowledgments

Thanks to:
- Tenney et al. (2019) for layer analysis methodology
- COPA dataset creators (Gordon et al., 2012)
- The broader interpretability research community

---

**Note**: This research values null results equally with positive findings. Both advance scientific understanding of transformer interpretability.

<4577> Dick Tracy's geometric detective work continues! 💕🚀
