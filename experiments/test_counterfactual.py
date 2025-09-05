#!/usr/bin/env python3
"""
Test Hypothesis 1: Counterfactual Divergence
Hillary Danan's causal attention geometry research

Claim: Attention patterns diverge at causal intervention points
Expected: KL divergence > 0.2 between factual/counterfactual pairs
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from causal_attention import CausalAttentionAnalyzer


def load_copa_dataset(n_samples: int = 64) -> List[Tuple[str, str]]:
    """
    Load or generate COPA-style counterfactual pairs.
    
    Returns pairs of (factual, counterfactual) sentences.
    """
    # Example COPA-style pairs for testing
    # In production, load from actual COPA dataset
    copa_pairs = [
        ("The glass broke because it fell off the table.", 
         "The glass broke because it was made of plastic."),
        
        ("She got wet because it was raining.", 
         "She got wet because she wore a raincoat."),
        
        ("The plant died because it wasn't watered.", 
         "The plant died because it got too much water."),
        
        ("He was late because there was traffic.", 
         "He was late because he left early."),
        
        ("The food spoiled because the power went out.", 
         "The food spoiled because it was in the freezer."),
        
        ("She smiled because she heard good news.", 
         "She smiled because she heard bad news."),
        
        ("The computer crashed because of a virus.", 
         "The computer crashed because it was new."),
        
        ("He studied hard because he wanted to pass.", 
         "He studied hard because he wanted to fail."),
        
        ("The ice melted because it was hot outside.", 
         "The ice melted because it was frozen solid."),
        
        ("She apologized because she made a mistake.", 
         "She apologized because she was right."),
        
        ("The alarm went off because there was smoke.", 
         "The alarm went off because the batteries died."),
        
        ("He exercised because he wanted to be healthy.", 
         "He exercised because he wanted to be lazy."),
        
        ("The meeting was cancelled because the boss was sick.", 
         "The meeting was cancelled because everyone attended."),
        
        ("She wore a coat because it was cold.", 
         "She wore a coat because it was summer."),
        
        ("The car stopped because it ran out of gas.", 
         "The car stopped because the tank was full."),
        
        ("He laughed because the joke was funny.", 
         "He laughed because the joke was serious."),
    ]
    
    # Extend dataset to reach required N
    extended_pairs = []
    while len(extended_pairs) < n_samples:
        for pair in copa_pairs:
            if len(extended_pairs) >= n_samples:
                break
            extended_pairs.append(pair)
            
            # Generate variations
            factual, counter = pair
            
            # Variation 1: Change tense
            past_factual = factual.replace("because", "since")
            past_counter = counter.replace("because", "since")
            extended_pairs.append((past_factual, past_counter))
            
            # Variation 2: Add context
            context_factual = f"Yesterday, {factual.lower()}"
            context_counter = f"Yesterday, {counter.lower()}"
            extended_pairs.append((context_factual, context_counter))
            
            # Variation 3: Different causal marker
            thus_factual = factual.replace("because", "therefore")
            thus_counter = counter.replace("because", "therefore")
            extended_pairs.append((thus_factual, thus_counter))
    
    return extended_pairs[:n_samples]


def run_counterfactual_experiment(analyzer: CausalAttentionAnalyzer,
                                 n_samples: int = 64) -> Dict:
    """
    Run the counterfactual divergence experiment.
    
    Tests whether attention patterns diverge at causal intervention points.
    """
    print(f"Testing Hypothesis 1: Counterfactual Divergence")
    print(f"Required N={n_samples} for 80% power at d=0.5")
    print("-" * 60)
    
    # Load dataset
    pairs = load_copa_dataset(n_samples)
    print(f"Loaded {len(pairs)} factual/counterfactual pairs")
    
    # Store results for each pair
    all_results = []
    divergences_by_layer = {i: [] for i in range(12)}
    
    for idx, (factual, counterfactual) in enumerate(pairs):
        if idx % 10 == 0:
            print(f"Processing pair {idx+1}/{len(pairs)}...")
        
        # Test divergence
        result = analyzer.test_counterfactual_divergence(factual, counterfactual)
        all_results.append(result)
        
        # Collect layer-wise divergences
        for layer_idx, divergence in result['layer_divergences'].items():
            divergences_by_layer[int(layer_idx)].append(divergence)
    
    # Aggregate statistics
    all_mean_divergences = [r['mean_kl_divergence'] for r in all_results]
    overall_mean = np.mean(all_mean_divergences)
    overall_std = np.std(all_mean_divergences)
    
    # Calculate confidence interval
    from scipy import stats
    confidence_level = 0.95
    ci = stats.t.interval(confidence_level, len(all_mean_divergences)-1, 
                          loc=overall_mean, 
                          scale=overall_std/np.sqrt(len(all_mean_divergences)))
    
    # Test against null hypothesis (divergence = 0.2)
    t_stat, p_value = stats.ttest_1samp(all_mean_divergences, 0.2)
    
    # Effect size (Cohen's d)
    cohens_d = (overall_mean - 0.2) / overall_std if overall_std > 0 else 0
    
    # Layer-wise analysis
    layer_means = {}
    layer_stds = {}
    for layer_idx, divergences in divergences_by_layer.items():
        if divergences:
            layer_means[layer_idx] = np.mean(divergences)
            layer_stds[layer_idx] = np.std(divergences)
    
    # Find layers with highest divergence
    best_layers = sorted(layer_means.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Compile final results
    final_results = {
        "hypothesis": "counterfactual_divergence",
        "n_samples": len(pairs),
        "mean_kl_divergence": float(overall_mean),
        "std_kl_divergence": float(overall_std),
        "confidence_interval_95": [float(ci[0]), float(ci[1])],
        "null_hypothesis_value": 0.2,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "hypothesis_supported": bool(overall_mean > 0.2 and p_value < 0.05),
        "layer_analysis": {
            "mean_by_layer": {str(k): float(v) for k, v in layer_means.items()},
            "std_by_layer": {str(k): float(v) for k, v in layer_stds.items()},
            "top_diverging_layers": [{"layer": int(l), "mean_divergence": float(d)} 
                                    for l, d in best_layers]
        },
        "interpretation": interpret_results(overall_mean, p_value, cohens_d)
    }
    
    return final_results


def interpret_results(mean_divergence: float, p_value: float, cohens_d: float) -> str:
    """
    Provide scientific interpretation of results.
    """
    if mean_divergence > 0.2 and p_value < 0.05:
        interpretation = (
            f"HYPOTHESIS SUPPORTED: Attention patterns show significant divergence "
            f"(mean KL={mean_divergence:.3f}) at causal intervention points between "
            f"factual and counterfactual conditions (p={p_value:.4f}). "
            f"Effect size d={cohens_d:.3f} indicates a "
            f"{'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect. "
            f"This suggests transformers geometrically distinguish causal structures."
        )
    elif mean_divergence > 0.2:
        interpretation = (
            f"MIXED EVIDENCE: While mean divergence ({mean_divergence:.3f}) exceeds "
            f"threshold, results are not statistically significant (p={p_value:.4f}). "
            f"Effect size d={cohens_d:.3f}. Larger sample size needed for conclusive results."
        )
    else:
        interpretation = (
            f"HYPOTHESIS NOT SUPPORTED: Mean divergence ({mean_divergence:.3f}) below "
            f"expected threshold of 0.2 (p={p_value:.4f}, d={cohens_d:.3f}). "
            f"Attention patterns do not show expected geometric divergence at "
            f"causal intervention points. This null result constrains claims about "
            f"transformer causal understanding."
        )
    
    return interpretation


def main():
    parser = argparse.ArgumentParser(
        description="Test counterfactual divergence in causal attention patterns"
    )
    parser.add_argument("--n-samples", type=int, default=64,
                       help="Number of samples to test (default: 64)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model to test (default: bert-base-uncased)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("CAUSAL ATTENTION GEOMETRY - HYPOTHESIS 1")
    print("Testing Counterfactual Divergence")
    print("Hillary Danan's Research Implementation")
    print("=" * 60)
    
    # Initialize analyzer
    print(f"\nInitializing model: {args.model}")
    analyzer = CausalAttentionAnalyzer(model_name=args.model)
    
    # Calculate required sample size
    required_n = analyzer.calculate_power_analysis(effect_size=0.5, power=0.8)
    print(f"Power analysis: N={required_n} required for 80% power at d=0.5")
    
    if args.n_samples < required_n:
        print(f"WARNING: Using N={args.n_samples} (less than required {required_n})")
        print(f"Results may be underpowered!")
    
    # Run experiment
    print("\nRunning experiment...")
    results = run_counterfactual_experiment(analyzer, n_samples=args.n_samples)
    
    # Save results
    output_file = output_path / "h1_counterfactual_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Mean KL Divergence: {results['mean_kl_divergence']:.4f} ± {results['std_kl_divergence']:.4f}")
    print(f"95% CI: [{results['confidence_interval_95'][0]:.4f}, {results['confidence_interval_95'][1]:.4f}]")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Cohen's d: {results['cohens_d']:.3f}")
    print(f"Hypothesis Supported: {results['hypothesis_supported']}")
    print(f"\nTop Diverging Layers:")
    for layer_info in results['layer_analysis']['top_diverging_layers']:
        print(f"  Layer {layer_info['layer']}: {layer_info['mean_divergence']:.4f}")
    print(f"\n{results['interpretation']}")
    print(f"\nResults saved to: {output_file}")
    
    return 0 if results['hypothesis_supported'] else 1


if __name__ == "__main__":
    sys.exit(main())