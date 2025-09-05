# Causal Attention Geometry

*Hillary Danan, September 2025*

## Overview

This repository investigates whether transformer models exhibit geometric patterns in attention distributions when processing causal relationships. Building on findings from related work in [embodied-cognition](https://github.com/HillaryDanan/embodied-cognition), [retroactive-causality](https://github.com/HillaryDanan/retroactive-causality), and [multi-geometric-attention](https://github.com/HillaryDanan/multi-geometric-attention), this project tests whether attention patterns reflect the underlying topology of causal structures.

## Core Research Questions

1. Do counterfactual statements produce measurable divergence in attention patterns at intervention points?
2. Do circular causal relationships (feedback loops) exhibit different attention density than linear chains?
3. Are causal attention patterns layer-specific, following the syntactic hierarchy found by Tenney et al. (2019)?

## Hypotheses

### H1: Counterfactual Divergence
- **Prediction**: KL divergence > 0.2 at causal intervention points
- **Rationale**: Based on Pearl's (2009) parallel worlds framework
- **Falsifiable**: If divergence ≤ 0.2 or p > 0.05, hypothesis rejected

### H2: Feedback Loop Density
- **Prediction**: Circular causation shows denser attention (Cohen's d > 0.3)
- **Rationale**: Feedback requires maintaining multiple dependencies
- **Falsifiable**: If d ≤ 0.3 or no significant difference, hypothesis rejected

### H3: Layer Specificity
- **Prediction**: Middle layers (5-8) show strongest causal effects
- **Rationale**: Following Tenney et al. (2019) on syntactic emergence
- **Falsifiable**: If effects uniform or in early/late layers, hypothesis rejected

## Methodology

### Dataset
- COPA (Choice of Plausible Alternatives) - Gordon et al., 2012
- 1000 causal reasoning examples
- Power analysis: N=64 per condition for 80% power at d=0.5

### Control Conditions
1. **Temporal**: Replace causal markers with temporal sequence
2. **Scrambled**: Preserve syntax, break causal logic  
3. **Conjunction**: Replace causal connectives with "and"

### Statistical Analysis
- Two-sample t-tests for group comparisons
- Cohen's d for effect size
- Bonferroni correction for multiple comparisons
- Layer-wise analysis to avoid aggregation artifacts

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

## Usage

```python
# Run main experiment
python3 run_experiments.py

# Run specific hypothesis test
python3 experiments/test_counterfactual.py
```

## Project Structure

```
causal-attention-geometry/
├── src/
│   ├── attention_extractor.py      # Extract attention patterns
│   ├── layerwise_analyzer.py       # Layer-specific analysis
│   └── statistical_tests.py        # Statistical utilities
├── experiments/
│   ├── test_counterfactual.py      # H1: Counterfactual divergence
│   ├── test_feedback_loops.py      # H2: Circular causation
│   └── test_layer_effects.py       # H3: Layer specificity
├── data/
│   └── copa_processed.json         # Processed COPA examples
└── results/
    └── [timestamp]_results.json    # Experimental results
```

## Expected Outcomes

### If Hypotheses Supported:
- Transformers implicitly learn causal topology
- Attention geometry reflects causal structure
- Foundation for geometric optimization of causal reasoning

### If Hypotheses Rejected:
- Attention patterns don't capture causal semantics
- Need alternative mechanisms for causal representation
- Valuable null result constraining interpretability claims

## Related Work

This project extends geometric insights from:
- [Multi-Geometric Attention Theory](https://github.com/HillaryDanan/multi-geometric-attention) - theoretical framework
- [Retroactive Causality](https://github.com/HillaryDanan/retroactive-causality) - temporal dependencies  
- [Cross-Linguistic Attention Dynamics](https://github.com/HillaryDanan/cross-linguistic-attention-dynamics) - language variation

## Citations

- Pearl, J. (2009). *Causality*. Cambridge University Press.
- Gordon, A. et al. (2012). SemEval-2012 Task 7: COPA. *SemEval*.
- Tenney, I. et al. (2019). BERT Rediscovers Classical NLP. *ACL*.
- Vig, J. & Belinkov, Y. (2019). Analyzing Structure in Transformer Representations. *ACL*.

## Contact

Hillary Danan - Independent Researcher

## License

MIT
