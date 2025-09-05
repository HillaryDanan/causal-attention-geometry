#!/usr/bin/env python3
"""
Test Hypothesis 2: Feedback Loop Density
Hillary Danan's causal attention geometry research

Claim: Circular causation shows denser attention than linear
Expected: Cohen's d > 0.3 between conditions
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


def generate_feedback_loop_examples(n_samples: int = 64) -> List[Tuple[str, str, str]]:
    """
    Generate examples of circular vs linear causation with controls.
    
    Returns tuples of (circular, linear, control) texts.
    """
    examples = []
    
    # Base examples with feedback loops
    base_examples = [
        # Ecological feedback loops
        ("Predators control prey which affects vegetation which affects predators",
         "Predators control prey and then vegetation changes",
         "Predators and prey and vegetation exist"),
        
        ("Rising temperatures cause ice melt which reduces reflection which increases temperatures",
         "Rising temperatures cause ice melt and then reflection decreases",
         "Temperatures and ice and reflection change"),
        
        ("Economic growth increases consumption which drives production which fuels growth",
         "Economic growth increases consumption and then production rises",
         "Economy and consumption and production occur"),
        
        # Psychological feedback loops
        ("Anxiety causes avoidance which increases fear which heightens anxiety",
         "Anxiety causes avoidance and then fear increases",
         "Anxiety and avoidance and fear happen"),
        
        ("Success builds confidence which improves performance which creates success",
         "Success builds confidence and then performance improves",
         "Success and confidence and performance exist"),
        
        # Social feedback loops
        ("Trust enables cooperation which builds relationships which strengthens trust",
         "Trust enables cooperation and then relationships build",
         "Trust and cooperation and relationships occur"),
        
        ("Innovation drives competition which spurs research which accelerates innovation",
         "Innovation drives competition and then research increases",
         "Innovation and competition and research happen"),
        
        # Biological feedback loops
        ("Exercise strengthens muscles which improves metabolism which enhances exercise",
         "Exercise strengthens muscles and then metabolism improves",
         "Exercise and muscles and metabolism function"),
        
        ("Inflammation triggers pain which causes stress which increases inflammation",
         "Inflammation triggers pain and then stress increases",
         "Inflammation and pain and stress exist"),
        
        # Environmental feedback loops
        ("Deforestation reduces rainfall which kills trees which accelerates deforestation",
         "Deforestation reduces rainfall and then trees die",
         "Deforestation and rainfall and trees change"),
        
        ("Pollution damages health which reduces productivity which increases pollution",
         "Pollution damages health and then productivity falls",
         "Pollution and health and productivity vary"),
        
        # Cognitive feedback loops
        ("Learning improves understanding which facilitates retention which enhances learning",
         "Learning improves understanding and then retention increases",
         "Learning and understanding and retention occur"),
        
        ("Practice develops skill which increases enjoyment which motivates practice",
         "Practice develops skill and then enjoyment increases",
         "Practice and skill and enjoyment exist"),
        
        # Market feedback loops
        ("Demand raises prices which reduces consumption which lowers demand",
         "Demand raises prices and then consumption falls",
         "Demand and prices and consumption fluctuate"),
        
        ("Investment generates returns which attracts capital which enables investment",
         "Investment generates returns and then capital arrives",
         "Investment and returns and capital move"),
        
        # Neural feedback loops
        ("Attention enhances processing which strengthens connections which focuses attention",
         "Attention enhances processing and then connections strengthen",
         "Attention and processing and connections work"),
    ]
    
    # Generate variations to reach n_samples
    for base_circular, base_linear, base_control in base_examples:
        if len(examples) >= n_samples:
            break
            
        # Original
        examples.append((base_circular, base_linear, base_control))
        
        # Variation 1: Add "because" explicitly
        circular_because = base_circular.replace(" which ", " because it ")
        linear_because = base_linear.replace(" and then ", " because ")
        control_because = base_control.replace(" and ", " also ")
        examples.append((circular_because, linear_because, control_because))
        
        # Variation 2: Past tense
        circular_past = base_circular.replace("affects", "affected").replace("causes", "caused")
        linear_past = base_linear.replace("causes", "caused").replace("changes", "changed")
        control_past = base_control.replace("exist", "existed").replace("occur", "occurred")
        examples.append((circular_past, linear_past, control_past))
        
        # Variation 3: Future tense
        circular_future = "will " + base_circular.replace(" which ", " which will ")
        linear_future = "will " + base_linear.replace(" and then ", " and then will ")
        control_future = "will " + base_control
        examples.append((circular_future, linear_future, control_future))
    
    return examples[:n_samples]


def run_feedback_density_experiment(analyzer: CausalAttentionAnalyzer,
                                   n_samples: int = 64) -> Dict:
    """
    Run the feedback loop density experiment.
    
    Tests whether circular causation shows denser attention patterns.
    """
    print(f"Testing Hypothesis 2: Feedback Loop Density")
    print(f"Required N={n_samples} for 80% power at d=0.3")
    print("-" * 60)
    
    # Generate examples
    examples = generate_feedback_loop_examples(n_samples)
    print(f"Generated {len(examples)} circular/linear/control triads")
    
    # Store results
    all_results = []
    circular_densities = []
    linear_densities = []
    control_densities = []
    
    for idx, (circular, linear, control) in enumerate(examples):
        if idx % 10 == 0:
            print(f"Processing example {idx+1}/{len(examples)}...")
        
        # Test density differences
        result = analyzer.test_feedback_loop_density(circular, linear, control)
        all_results.append(result)
        
        # Collect densities
        circular_densities.append(result['mean_circular_density'])
        linear_densities.append(result['mean_linear_density'])
        control_densities.append(result['mean_control_density'])
    
    # Calculate aggregate statistics
    from scipy import stats
    
    # Means and standard deviations
    circ_mean, circ_std = np.mean(circular_densities), np.std(circular_densities)
    lin_mean, lin_std = np.mean(linear_densities), np.std(linear_densities)
    ctrl_mean, ctrl_std = np.mean(control_densities), np.std(control_densities)
    
    # Effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(
            ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof
        )
    
    d_circ_lin = cohens_d(circular_densities, linear_densities)
    d_circ_ctrl = cohens_d(circular_densities, control_densities)
    d_lin_ctrl = cohens_d(linear_densities, control_densities)
    
    # Statistical tests
    t_circ_lin, p_circ_lin = stats.ttest_ind(circular_densities, linear_densities)
    t_circ_ctrl, p_circ_ctrl = stats.ttest_ind(circular_densities, control_densities)
    t_lin_ctrl, p_lin_ctrl = stats.ttest_ind(linear_densities, control_densities)
    
    # ANOVA for overall difference
    f_stat, p_anova = stats.f_oneway(circular_densities, linear_densities, control_densities)
    
    # Confidence intervals
    confidence_level = 0.95
    ci_circular = stats.t.interval(confidence_level, len(circular_densities)-1, 
                                   loc=circ_mean, 
                                   scale=circ_std/np.sqrt(len(circular_densities)))
    ci_linear = stats.t.interval(confidence_level, len(linear_densities)-1,
                                 loc=lin_mean,
                                 scale=lin_std/np.sqrt(len(linear_densities)))
    
    # Layer-wise analysis
    layer_effects = {}
    for result in all_results:
        for i, (circ, lin, ctrl) in enumerate(zip(
            result['layer_densities']['circular'],
            result['layer_densities']['linear'],
            result['layer_densities']['control']
        )):
            if i not in layer_effects:
                layer_effects[i] = {'circular': [], 'linear': [], 'control': []}
            layer_effects[i]['circular'].append(circ)
            layer_effects[i]['linear'].append(lin)
            layer_effects[i]['control'].append(ctrl)
    
    # Calculate layer-wise effect sizes
    layer_cohens_d = {}
    for layer_idx, densities in layer_effects.items():
        d = cohens_d(densities['circular'], densities['linear'])
        layer_cohens_d[layer_idx] = float(d)
    
    # Find layers with strongest effects
    best_layers = sorted(layer_cohens_d.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    # Compile results
    final_results = {
        "hypothesis": "feedback_loop_density",
        "n_samples": len(examples),
        "density_statistics": {
            "circular": {
                "mean": float(circ_mean),
                "std": float(circ_std),
                "95_ci": [float(ci_circular[0]), float(ci_circular[1])]
            },
            "linear": {
                "mean": float(lin_mean),
                "std": float(lin_std),
                "95_ci": [float(ci_linear[0]), float(ci_linear[1])]
            },
            "control": {
                "mean": float(ctrl_mean),
                "std": float(ctrl_std)
            }
        },
        "effect_sizes": {
            "circular_vs_linear": float(d_circ_lin),
            "circular_vs_control": float(d_circ_ctrl),
            "linear_vs_control": float(d_lin_ctrl)
        },
        "statistical_tests": {
            "circular_vs_linear": {
                "t_statistic": float(t_circ_lin),
                "p_value": float(p_circ_lin)
            },
            "circular_vs_control": {
                "t_statistic": float(t_circ_ctrl),
                "p_value": float(p_circ_ctrl)
            },
            "anova": {
                "f_statistic": float(f_stat),
                "p_value": float(p_anova)
            }
        },
        "hypothesis_supported": bool(d_circ_lin > 0.3 and p_circ_lin < 0.05),
        "layer_analysis": {
            "effect_sizes_by_layer": {str(k): v for k, v in layer_cohens_d.items()},
            "strongest_effect_layers": [{"layer": int(l), "cohens_d": float(d)} 
                                       for l, d in best_layers]
        },
        "interpretation": interpret_feedback_results(d_circ_lin, p_circ_lin, circ_mean, lin_mean)
    }
    
    return final_results


def interpret_feedback_results(effect_size: float, p_value: float, 
                              circ_mean: float, lin_mean: float) -> str:
    """
    Provide scientific interpretation of feedback loop results.
    """
    if effect_size > 0.3 and p_value < 0.05:
        interpretation = (
            f"HYPOTHESIS SUPPORTED: Circular causation shows significantly denser "
            f"attention patterns (mean={circ_mean:.4f}) than linear causation "
            f"(mean={lin_mean:.4f}), with effect size d={effect_size:.3f} "
            f"(p={p_value:.4f}). This suggests transformers allocate more "
            f"attention resources to feedback loops, potentially reflecting "
            f"their increased computational complexity."
        )
    elif effect_size > 0.3:
        interpretation = (
            f"MIXED EVIDENCE: Effect size ({effect_size:.3f}) exceeds threshold "
            f"but lacks statistical significance (p={p_value:.4f}). "
            f"Circular (mean={circ_mean:.4f}) vs linear (mean={lin_mean:.4f}). "
            f"Trend supports hypothesis but requires larger sample."
        )
    elif abs(effect_size) < 0.1:
        interpretation = (
            f"NULL RESULT: No meaningful difference in attention density between "
            f"circular (mean={circ_mean:.4f}) and linear (mean={lin_mean:.4f}) "
            f"causation (d={effect_size:.3f}, p={p_value:.4f}). "
            f"Transformers may not geometrically distinguish feedback loops "
            f"from linear causal chains."
        )
    else:
        interpretation = (
            f"HYPOTHESIS NOT SUPPORTED: Insufficient effect size ({effect_size:.3f}) "
            f"between circular (mean={circ_mean:.4f}) and linear (mean={lin_mean:.4f}) "
            f"causation (p={p_value:.4f}). Attention geometry may not reflect "
            f"causal topology as hypothesized."
        )
    
    return interpretation


def main():
    parser = argparse.ArgumentParser(
        description="Test feedback loop density in causal attention patterns"
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
    print("CAUSAL ATTENTION GEOMETRY - HYPOTHESIS 2")
    print("Testing Feedback Loop Density")
    print("Hillary Danan's Research Implementation")
    print("=" * 60)
    
    # Initialize analyzer
    print(f"\nInitializing model: {args.model}")
    analyzer = CausalAttentionAnalyzer(model_name=args.model)
    
    # Calculate required sample size for d=0.3
    required_n = analyzer.calculate_power_analysis(effect_size=0.3, power=0.8)
    print(f"Power analysis: N={required_n} required for 80% power at d=0.3")
    
    if args.n_samples < required_n:
        print(f"WARNING: Using N={args.n_samples} (less than required {required_n})")
        print(f"Results may be underpowered!")
    
    # Run experiment
    print("\nRunning experiment...")
    results = run_feedback_density_experiment(analyzer, n_samples=args.n_samples)
    
    # Save results
    output_file = output_path / "h2_feedback_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Density Means:")
    print(f"  Circular: {results['density_statistics']['circular']['mean']:.4f} ± "
          f"{results['density_statistics']['circular']['std']:.4f}")
    print(f"  Linear:   {results['density_statistics']['linear']['mean']:.4f} ± "
          f"{results['density_statistics']['linear']['std']:.4f}")
    print(f"  Control:  {results['density_statistics']['control']['mean']:.4f} ± "
          f"{results['density_statistics']['control']['std']:.4f}")
    print(f"\nEffect Sizes:")
    print(f"  Circular vs Linear:  d={results['effect_sizes']['circular_vs_linear']:.3f} "
          f"(p={results['statistical_tests']['circular_vs_linear']['p_value']:.4f})")
    print(f"  Circular vs Control: d={results['effect_sizes']['circular_vs_control']:.3f} "
          f"(p={results['statistical_tests']['circular_vs_control']['p_value']:.4f})")
    print(f"\nHypothesis Supported: {results['hypothesis_supported']}")
    print(f"\nStrongest Effect Layers:")
    for layer_info in results['layer_analysis']['strongest_effect_layers']:
        print(f"  Layer {layer_info['layer']}: d={layer_info['cohens_d']:.3f}")
    print(f"\n{results['interpretation']}")
    print(f"\nResults saved to: {output_file}")
    
    return 0 if results['hypothesis_supported'] else 1


if __name__ == "__main__":
    sys.exit(main())