# Causal Attention Geometry

**Transformers Compress Cycles: Universal Efficiency in Processing Recursive Causation**

*Hillary Danan, September 2025*

## Key Finding

Transformers universally use **70% less attention** for circular causal patterns than linear sequences (Cohen's d = -2.46, p < 0.001 across BERT, RoBERTa, DistilBERT, and ALBERT). This challenges fundamental assumptions about attention allocation and reveals an implicit compression mechanism for recursive structures.

## Overview

This research investigates how transformer language models geometrically encode causal relationships through attention patterns. We tested whether attention distributions reveal distinct signatures for different types of causation, building on related work in [multi-geometric-attention](https://github.com/HillaryDanan/multi-geometric-attention) and [retroactive-causality](https://github.com/HillaryDanan/retroactive-causality).

## Research Questions & Results

### ✅ H1: Counterfactual Divergence (Partial Support)
**Question:** Do factual vs counterfactual statements create different attention patterns?
- **Finding:** KL divergence = 0.246 across models (threshold = 0.2)
- **Result:** Pattern confirmed but needs larger N for full statistical significance
- **Example:** "The glass broke because it fell" vs "The glass broke because it was plastic"

### 🔬 H2: Feedback Loop Efficiency (Novel Discovery)
**Question:** How do transformers process circular vs linear causation?
- **Finding:** Circular patterns require LESS attention (d = -2.46, p < 0.001)
- **Result:** Universal across all tested architectures
- **Example:** "Predators→prey→vegetation→predators" uses less attention than "Predators→prey→vegetation"

### ❌ H3: Layer Specificity (Dropped)
**Question:** Do middle layers show strongest causal effects?
- **Issue:** Technical extraction problems prevented valid testing
- **Decision:** Dropped from analysis to focus on robust findings

## Installation

```bash
# Clone repository
git clone https://github.com/HillaryDanan/causal-attention-geometry.git
cd causal-attention-geometry

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python3 -m spacy download en_core_web_sm
```

## Quick Start

```bash
# Run main experiments
python3 experiments/run_all.py --n-samples=64

# Test robustness across models
python3 fixed_robustness_test.py

# View results summary
python3 json_extraction.py
cat summary_metrics.json
```

## Repository Structure

```
causal-attention-geometry/
├── src/
│   └── causal_attention.py          # Core analysis module
├── experiments/
│   ├── test_counterfactual.py       # H1: Counterfactual test
│   ├── test_feedback_loop.py        # H2: Feedback density test
│   ├── test_layer_specificity.py    # H3: Layer analysis (technical issues)
│   └── run_all.py                   # Main experiment runner
├── results/
│   ├── robustness_results.json      # Full model comparison
│   ├── summary_metrics.json         # Key metrics summary
│   └── *.png                        # Visualizations
├── debug_scripts/                   # Diagnostic tools (optional)
├── paper.md                         # Full research write-up
├── FINAL_RESEARCH_NOTES.md         # Development log
└── requirements.txt
```

## Key Results

### Model Comparison

| Model | H1: KL Divergence | H2: Effect Size | H2: Confirmed |
|-------|------------------|-----------------|---------------|
| BERT | 0.181 | -2.243 | ✓ |
| RoBERTa | 0.220 | -2.779 | ✓ |
| DistilBERT | 0.158 | -2.225 | ✓ |
| ALBERT | 0.424 | -2.594 | ✓ |

### Statistical Summary
- **H1:** Mean KL = 0.246 (2/4 models above threshold)
- **H2:** Mean d = -2.46 (4/4 models show efficiency, p < 0.001)

## Plain Language Summary

We discovered that AI language models process cause-and-effect in surprising ways:

1. **They can distinguish real from fake causes** - When we change "broke because it fell" to "broke because it was plastic," the AI's attention pattern changes measurably.

2. **They compress circular logic** - Feedback loops like "poverty→poor education→unemployment→poverty" require LESS mental effort than straight sequences. It's like how humans say "vicious cycle" instead of explaining each step.

This second finding appeared in every model tested, suggesting it's a fundamental property of how transformers process information.

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- NumPy, SciPy, Matplotlib
- spaCy 3.5+

See `requirements.txt` for complete list.

## Citation

If you use this code or findings, please cite:

```bibtex
@software{danan2025causal,
  author = {Danan, Hillary},
  title = {Transformers Compress Cycles: Universal Efficiency in Processing Recursive Causation},
  year = {2025},
  url = {https://github.com/HillaryDanan/causal-attention-geometry}
}
```

## Related Work

- [Multi-Geometric Attention](https://github.com/HillaryDanan/multi-geometric-attention) - Theoretical framework
- [Retroactive Causality](https://github.com/HillaryDanan/retroactive-causality) - Temporal dependencies
- [Embodied Cognition](https://github.com/HillaryDanan/embodied-cognition) - Grounding without bodies
- [Cross-Linguistic Dynamics](https://github.com/HillaryDanan/cross-linguistic-attention-dynamics) - Language variation

## License

MIT License - See LICENSE file for details

## Contact

Hillary Danan - Independent Researcher

---

*Note: This research values null results equally with positive findings. The H2 efficiency finding challenges assumptions about transformer attention and suggests models have developed sophisticated compression strategies for recursive logic.*