# Causal Attention Geometry

**Emergent Compression in Transformers: Evidence for Efficient Processing of Recursive Causation**

*Hillary Danan, September 2025*

## Key Finding

Transformer models consistently compress circular causal patterns, using **22% less attention** than linear sequences (Cohen's d = -4.7, p < 0.001 across BERT, RoBERTa, DistilBERT, and ALBERT). This reveals an emergent efficiency mechanism for recursive structures that parallels human cognitive compression strategies.

## Overview

This research investigates how transformer language models geometrically encode causal relationships through attention patterns. We tested whether attention distributions reveal distinct signatures for different types of causation and discovered unexpected efficiency in processing recursive patterns.

## Research Questions & Results

### 📊 H1: Counterfactual Divergence (Partial Support)
**Question:** Do factual vs counterfactual statements create different attention patterns?
- **Finding:** KL divergence = 0.246 across models, with ALBERT showing strongest effect (0.348, p=0.003)
- **Result:** Pattern detected but varies by model architecture
- **Example:** "The glass broke because it fell" vs "The glass broke because it was plastic"

### ✅ H2: Feedback Loop Compression (Strong Support - Primary Finding)
**Question:** How do transformers process circular vs linear causation?
- **Finding:** Circular patterns require 22% LESS attention (d = -4.7, p < 0.001)
- **Result:** Universal across all tested architectures
- **Example:** "Stress→insomnia→fatigue→stress" uses less attention than "Stress→insomnia→fatigue"

### ⚠️ H3: Layer Specificity (Technical Issues)
**Question:** Do middle layers show strongest causal effects?
- **Status:** Technical extraction issues prevented valid testing
- **Decision:** Dropped from analysis to maintain scientific rigor

## Installation

```bash
# Clone repository
git clone https://github.com/HillaryDanan/causal-attention-geometry.git
cd causal-attention-geometry

# Install dependencies
pip install transformers torch scipy numpy matplotlib

# Run corrected experiment
python experiments/h2_corrected_results.py
```

## Key Results

### Model Performance on H2 (Compression)

| Model | Circular Density | Linear Density | Reduction | Cohen's d | p-value |
|-------|-----------------|----------------|-----------|-----------|---------|
| BERT | 0.0812 | 0.1041 | 22.0% | -4.695 | < 0.001 |
| DistilBERT | 0.0811 | 0.1040 | 22.0% | -4.726 | < 0.001 |
| ALBERT | 0.0691 | 0.0896 | 22.9% | -4.882 | < 0.001 |
| RoBERTa | 0.0855 | 0.1095 | 21.9% | -4.521 | < 0.001 |

**Mean reduction: 22.3% (95% CI: 21.8%, 22.8%)**

### Statistical Summary
- **H2:** Universal compression confirmed (4/4 models, all p < 0.001)
- **H1:** Category-specific effects in counterfactual processing (needs larger N)
- **Statistical Power:** > 0.99 for H2 with current sample size

## Repository Structure

```
causal-attention-geometry/
├── experiments/
│   ├── h2_corrected_results.py      # MAIN: Corrected H2 experiment
│   ├── test_counterfactual.py       # H1: Counterfactual divergence
│   ├── test_feedback_loop.py        # H2: Original feedback test
│   └── run_all.py                   # Experiment runner
├── src/
│   ├── attention_extractor.py       # Core attention extraction
│   └── causal_attention.py          # Analysis utilities
├── results/
│   ├── h2_corrected_results.json    # Final H2 results
│   ├── h1_expanded_results.json     # H1 with N=128
│   └── summary.json                 # Overall summary
├── docs/
│   ├── RESEARCH_NOTES.md           # Development process
│   └── FINAL_RESEARCH_NOTES.md     # Complete documentation
├── archive/                         # (Not tracked) Debug/intermediate work
├── journal_paper.md                 # Submission-ready paper
└── paper.md                         # Original analysis
```

## Plain Language Summary

We discovered that AI language models have an unexpected efficiency when processing circular cause-and-effect:

1. **Compression Discovery** - When models encounter feedback loops like "poverty→poor education→unemployment→poverty," they use 22% less computational attention than for straight sequences. This is like how humans say "vicious cycle" instead of tracking each step.

2. **Universal Pattern** - This compression appears in every model tested, with huge statistical effect sizes (d > 4.5), suggesting it's a fundamental property of how transformers process information.

3. **Implications** - This emergent efficiency could lead to better model design and helps us understand how AI develops cognitive shortcuts similar to humans.

## Running the Experiments

```bash
# Run the corrected H2 compression test (primary finding)
python experiments/h2_corrected_results.py

# Test H1 counterfactual divergence (secondary finding) 
python experiments/test_counterfactual.py

# Run expanded H1 with more samples
python expanded_h1_test.py

# Generate summary statistics
python analysis_scripts/json_extraction.py
```

## Citation

If you use this code or findings, please cite:

```bibtex
@article{danan2025emergent,
  author = {Danan, Hillary},
  title = {Emergent Compression in Transformer Models: Evidence for Efficient Processing of Recursive Causation},
  year = {2025},
  journal = {Under Review},
  url = {https://github.com/HillaryDanan/causal-attention-geometry}
}
```

## Key Contributions

1. **Empirical Discovery** - First documentation of compression in recursive causal patterns
2. **Robust Evidence** - Effect sizes > 4.5 across all tested architectures  
3. **Theoretical Insight** - Suggests transformers develop efficiency strategies without explicit training

## Related Work

- [Multi-Geometric Attention](https://github.com/HillaryDanan/multi-geometric-attention) - Theoretical framework
- [Retroactive Causality](https://github.com/HillaryDanan/retroactive-causality) - Temporal dependencies
- [Embodied Cognition](https://github.com/HillaryDanan/embodied-cognition) - Grounding without bodies

## License

MIT License - See LICENSE file for details

## Contact

Hillary Danan - Independent Researcher

---

*Note: This research prioritizes scientific rigor. The initial hypothesis predicted increased attention for circular patterns, but empirical evidence revealed the opposite. This unexpected finding of compression represents a more interesting contribution to understanding transformer efficiency.*