"""
Test H1: Counterfactual Divergence
Hillary Danan, September 2025
"""

import sys
sys.path.append('../src')
from attention_extractor import AttentionExtractor
import numpy as np
from scipy import stats

def test_counterfactual_hypothesis():
    """Test if counterfactuals show attention divergence"""
    
    print("="*60)
    print("H1: COUNTERFACTUAL DIVERGENCE TEST")
    print("="*60)
    
    extractor = AttentionExtractor()
    
    # Test cases (minimal set for initial testing)
    test_pairs = [
        ("The glass fell and broke", 
         "If the glass hadn't fallen, it wouldn't have broken", 3),
        ("She studied and passed",
         "If she hadn't studied, she would have failed", 2),
        ("It rained so the ground is wet",
         "If it hadn't rained, the ground would be dry", 2)
    ]
    
    divergences = []
    
    for factual, counterfactual, pos in test_pairs:
        div = extractor.measure_divergence(factual, counterfactual, pos)
        divergences.append(div)
        print(f"\nFactual: {factual}")
        print(f"Counter: {counterfactual}")
        print(f"KL Divergence: {div:.3f}")
    
    # Statistical test
    mean_div = np.mean(divergences)
    t_stat, p_value = stats.ttest_1samp(divergences, 0.2)
    
    print("\n" + "="*60)
    print(f"Mean divergence: {mean_div:.3f}")
    print(f"Hypothesis: divergence > 0.2")
    print(f"p-value: {p_value:.4f}")
    
    if mean_div > 0.2 and p_value < 0.05:
        print("✓ H1 SUPPORTED")
    else:
        print("✗ H1 NOT SUPPORTED")
        
    return divergences

if __name__ == "__main__":
    test_counterfactual_hypothesis()
