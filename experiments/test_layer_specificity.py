#!/usr/bin/env python3
"""
Test Hypothesis 3: Layer Specificity
Hillary Danan's causal attention geometry research

Claim: Middle layers (5-8) show strongest causal effects
Following Tenney et al. (2019) layer-wise analysis approach
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import argparse
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from causal_attention import CausalAttentionAnalyzer


def generate_causal_texts(n_samples: int = 64) -> List[str]:
    """
    Generate diverse causal texts for testing.
    """
    causal_templates = [
        # Physical causation
        "The {} broke because {} applied too much force.",
        "Heat caused the {} to expand, which made {} change.",
        "The collision caused {} to move, affecting {}.",
        "Gravity made {} fall, which triggered {}.",
        "The pressure caused {} to crack, leading to {}.",
        
        # Biological causation
        "The virus caused {}, which led to {} symptoms.",
        "Lack of {} caused the plant to wilt, affecting {}.",
        "The mutation caused {} to change, altering {}.",
        "Exercise caused {} to strengthen, improving {}.",
        "The hormone triggered {}, which influenced {}.",
        
        # Psychological causation
        "Stress caused {} to react, which affected {}.",
        "The trauma led to {}, causing {} to develop.",
        "Happiness caused {} to improve, enhancing {}.",
        "Fear made {} avoid the situation, preventing {}.",
        "Motivation drove {} to succeed, achieving {}.",
        
        # Social causation
        "The policy caused {} to change, affecting {} behavior.",
        "Communication breakdown led to {}, causing {}.",
        "Trust enabled {} to cooperate, facilitating {}.",
        "The conflict caused {} to separate, dividing {}.",
        "Leadership inspired {} to act, mobilizing {}.",
        
        # Economic causation
        "Inflation caused {} to rise, impacting {}.",
        "The shortage led to {}, driving {} up.",
        "Investment in {} caused growth, expanding {}.",
        "The crisis caused {} to collapse, destroying {}.",
        "Demand for {} increased prices, affecting {}.",
        
        # Environmental causation
        "Pollution caused {} to deteriorate, harming {}.",
        "Climate change led to {}, affecting {} patterns.",
        "Deforestation caused {} to disappear, eliminating {}.",
        "The drought made {} scarce, limiting {}.",
        "Conservation efforts caused {} to recover, restoring {}.",
    ]
    
    # Word lists for filling templates
    objects = ["system", "structure", "mechanism", "process", "component", "element", 
              "network", "pattern", "organization", "configuration", "arrangement", "design"]
    
    agents = ["researcher", "operator", "controller", "manager", "analyst", "observer",
             "participant", "coordinator", "specialist", "technician", "expert", "scientist"]
    
    outcomes = ["failure", "success", "change", "stability", "growth", "decline",
               "improvement", "degradation", "adaptation", "transformation", "evolution", "shift"]
    
    conditions = ["temperature", "pressure", "concentration", "density", "intensity", "frequency",
                 "amplitude", "duration", "magnitude", "velocity", "acceleration", "momentum"]
    
    causal_texts = []
    
    import random
    random.seed(42)  # For reproducibility
    
    for i in range(n_samples):
        template = causal_templates[i % len(causal_templates)]
        
        # Fill template with random words
        obj1 = random.choice(objects)
        obj2 = random.choice(objects)
        agent = random.choice(agents)
        outcome = random.choice(outcomes)
        condition = random.choice(conditions)
        
        # Create variations
        if "{}" in template:
            count = template.count("{}")
            if count == 2:
                text = template.format(
                    random.choice([obj1, agent, condition]),
                    random.choice([obj2, outcome, condition])
                )
            else:
                text = template.format(obj1)
        else:
            text = template
            
        causal_texts.append(text)
        
        # Add explicit causal markers variations
        if i % 3 == 0:
            text = f"Because {text.lower()}"
        elif i % 3 == 1:
            text = f"{text} Therefore, consequences followed."
        else:
            text = f"Since {text.lower()[:-1]}, effects emerged."
            
        causal_texts.append(text)
    
    return causal_texts[:n_samples]


def generate_non_causal_texts(n_samples: int = 64) -> List[str]:
    """
    Generate non-causal control texts.
    """
    non_causal_templates = [
        # Descriptive statements
        "The {} is located near the {}.",
        "The {} appears {} and {}.",
        "There are {} types of {} available.",
        "{} and {} coexist in the environment.",
        "The {} has {} characteristics.",
        
        # Observational statements
        "The {} was observed at {}.",
        "Measurements show {} in the {}.",
        "The {} displays {} properties.",
        "Analysis reveals {} within {}.",
        "Studies document {} and {}.",
        
        # Classificatory statements
        "The {} belongs to the {} category.",
        "{} is classified as {}.",
        "Types include {} and {}.",
        "Categories encompass {} through {}.",
        "Groups contain {} alongside {}.",
        
        # Temporal statements (non-causal)
        "The {} occurred during {}.",
        "At noon, {} was present.",
        "Yesterday, {} appeared normal.",
        "The {} lasted for {} hours.",
        "Events included {} and {}.",
        
        # Spatial statements
        "The {} sits beside the {}.",
        "Above the {}, {} can be found.",
        "The {} surrounds the {}.",
        "Between {} and {} lies the area.",
        "The {} extends across {}.",
        
        # Comparative statements
        "The {} is similar to {}.",
        "{} differs from {} in size.",
        "Both {} and {} share features.",
        "The {} contrasts with {}.",
        "{} resembles {} in appearance.",
    ]
    
    # Word lists for non-causal contexts
    things = ["table", "building", "container", "device", "instrument", "material",
             "substance", "compound", "mixture", "solution", "medium", "substrate"]
    
    qualities = ["large", "small", "smooth", "rough", "bright", "dark",
                "heavy", "light", "solid", "liquid", "stable", "variable"]
    
    locations = ["center", "edge", "surface", "interior", "boundary", "interface",
                "region", "zone", "area", "section", "segment", "portion"]
    
    times = ["morning", "afternoon", "evening", "weekday", "weekend", "season",
            "quarter", "period", "interval", "moment", "instant", "duration"]
    
    non_causal_texts = []
    
    import random
    random.seed(24)  # Different seed for variety
    
    for i in range(n_samples):
        template = non_causal_templates[i % len(non_causal_templates)]
        
        # Fill template
        thing1 = random.choice(things)
        thing2 = random.choice(things)
        quality1 = random.choice(qualities)
        quality2 = random.choice(qualities)
        location = random.choice(locations)
        time = random.choice(times)
        
        # Create variations
        if "{}" in template:
            count = template.count("{}")
            if count == 2:
                text = template.format(
                    random.choice([thing1, quality1, location]),
                    random.choice([thing2, quality2, time])
                )
            elif count == 3:
                text = template.format(thing1, quality1, quality2)
            else:
                text = template.format(thing1)
        else:
            text = template
            
        non_causal_texts.append(text)
        
        # Add conjunction variations (non-causal)
        if i % 3 == 0:
            text = f"{text} Additionally, observations continue."
        elif i % 3 == 1:
            text = f"{text} Furthermore, data exists."
        else:
            text = f"{text} Also, records show patterns."
            
        non_causal_texts.append(text)
    
    return non_causal_texts[:n_samples]


def run_layer_specificity_experiment(analyzer: CausalAttentionAnalyzer,
                                    n_samples: int = 64) -> Dict:
    """
    Run the layer specificity experiment.
    
    Tests whether middle layers show strongest causal effects.
    """
    print(f"Testing Hypothesis 3: Layer Specificity")
    print(f"Testing if middle layers (5-8) show strongest causal effects")
    print(f"N={n_samples} samples per condition")
    print("-" * 60)
    
    # Generate texts
    causal_texts = generate_causal_texts(n_samples)
    non_causal_texts = generate_non_causal_texts(n_samples)
    
    print(f"Generated {len(causal_texts)} causal texts")
    print(f"Generated {len(non_causal_texts)} non-causal texts")
    
    # Test layer specificity
    print("\nAnalyzing layer-wise patterns...")
    results = analyzer.test_layer_specificity(causal_texts, non_causal_texts)
    
    # Additional analysis: variance across layers
    layer_effects = results['layer_effect_sizes']
    middle_layers = [5, 6, 7, 8]
    early_layers = [0, 1, 2, 3, 4]
    late_layers = [9, 10, 11] if 11 in layer_effects else [9, 10]
    
    # Calculate mean effects by layer group
    middle_effects = [layer_effects[i] for i in middle_layers if i in layer_effects]
    early_effects = [layer_effects[i] for i in early_layers if i in layer_effects]
    late_effects = [layer_effects[i] for i in late_layers if i in layer_effects]
    
    # Statistical comparison between layer groups
    from scipy import stats
    
    # ANOVA across layer groups
    if middle_effects and early_effects and late_effects:
        f_stat, p_anova = stats.f_oneway(middle_effects, early_effects, late_effects)
    else:
        f_stat, p_anova = 0.0, 1.0
    
    # Peak layer identification
    peak_layer = max(layer_effects.items(), key=lambda x: abs(x[1]))[0]
    peak_effect = layer_effects[peak_layer]
    
    # Calculate confidence intervals for layer groups
    def calculate_ci(data, confidence=0.95):
        if not data:
            return [0.0, 0.0]
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(len(data))
        margin = 1.96 * std_err  # Approximate for 95% CI
        return [float(mean - margin), float(mean + margin)]
    
    # Compile comprehensive results
    final_results = {
        "hypothesis": "layer_specificity",
        "n_samples_per_condition": n_samples,
        "layer_effect_sizes": {str(k): float(v) for k, v in layer_effects.items()},
        "layer_p_values": results['layer_p_values'],
        "layer_group_analysis": {
            "early_layers": {
                "layers": early_layers,
                "mean_effect": float(np.mean(early_effects)) if early_effects else 0.0,
                "std_effect": float(np.std(early_effects)) if early_effects else 0.0,
                "95_ci": calculate_ci(early_effects)
            },
            "middle_layers": {
                "layers": middle_layers,
                "mean_effect": float(np.mean(middle_effects)) if middle_effects else 0.0,
                "std_effect": float(np.std(middle_effects)) if middle_effects else 0.0,
                "95_ci": calculate_ci(middle_effects)
            },
            "late_layers": {
                "layers": late_layers,
                "mean_effect": float(np.mean(late_effects)) if late_effects else 0.0,
                "std_effect": float(np.std(late_effects)) if late_effects else 0.0,
                "95_ci": calculate_ci(late_effects)
            }
        },
        "statistical_tests": {
            "anova_across_layer_groups": {
                "f_statistic": float(f_stat),
                "p_value": float(p_anova)
            },
            "bonferroni_corrected_alpha": results['bonferroni_corrected_alpha'],
            "significant_layers": results['significant_layers']
        },
        "peak_analysis": {
            "peak_layer": int(peak_layer),
            "peak_effect_size": float(peak_effect),
            "is_middle_layer": peak_layer in middle_layers
        },
        "hypothesis_supported": results['supported'],
        "interpretation": interpret_layer_results(
            results['supported'],
            np.mean(middle_effects) if middle_effects else 0,
            np.mean(early_effects) if early_effects else 0,
            np.mean(late_effects) if late_effects else 0,
            peak_layer,
            results['significant_layers']
        )
    }
    
    return final_results


def interpret_layer_results(supported: bool, middle_mean: float, early_mean: float,
                           late_mean: float, peak_layer: int, sig_layers: List[int]) -> str:
    """
    Provide scientific interpretation of layer specificity results.
    """
    if supported and peak_layer in [5, 6, 7, 8]:
        interpretation = (
            f"HYPOTHESIS SUPPORTED: Middle layers show strongest causal effects "
            f"(mean d={middle_mean:.3f}) compared to early (d={early_mean:.3f}) "
            f"and late (d={late_mean:.3f}) layers. Peak at layer {peak_layer}. "
            f"Significant layers: {sig_layers}. This aligns with Tenney et al. (2019) "
            f"findings that middle layers capture semantic relationships."
        )
    elif peak_layer in [5, 6, 7, 8]:
        interpretation = (
            f"PARTIAL SUPPORT: Peak causal effect at layer {peak_layer} (middle), "
            f"but statistical significance not achieved after correction. "
            f"Middle mean={middle_mean:.3f}, early={early_mean:.3f}, late={late_mean:.3f}. "
            f"Trend consistent with hypothesis but requires more power."
        )
    elif abs(middle_mean - early_mean) < 0.1 and abs(middle_mean - late_mean) < 0.1:
        interpretation = (
            f"NULL RESULT: No layer specialization for causal processing detected. "
            f"Similar effects across early (d={early_mean:.3f}), middle (d={middle_mean:.3f}), "
            f"and late (d={late_mean:.3f}) layers. Peak at layer {peak_layer}. "
            f"Causal attention may be distributed rather than localized."
        )
    else:
        interpretation = (
            f"HYPOTHESIS NOT SUPPORTED: Peak causal effect at layer {peak_layer} "
            f"({'early' if peak_layer < 5 else 'late'} layer). "
            f"Middle layers (d={middle_mean:.3f}) do not show strongest effects. "
            f"Early={early_mean:.3f}, late={late_mean:.3f}. "
            f"Causal processing may occur at different depths than expected."
        )
    
    return interpretation


def visualize_layer_effects(results: Dict, output_path: Path):
    """
    Create visualization of layer-wise causal effects.
    """
    layers = sorted([int(k) for k in results['layer_effect_sizes'].keys()])
    effects = [results['layer_effect_sizes'][str(l)] for l in layers]
    p_values = [results['layer_p_values'][str(l)] for l in layers]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Effect sizes by layer
    colors = ['red' if 5 <= l <= 8 else 'blue' for l in layers]
    bars = ax1.bar(layers, effects, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='d=0.3 threshold')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel("Cohen's d")
    ax1.set_title('Causal Effect Size by Layer (Middle layers 5-8 in red)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add significance stars
    bonferroni_alpha = results['statistical_tests']['bonferroni_corrected_alpha']
    for i, (layer, p_val) in enumerate(zip(layers, p_values)):
        if p_val < bonferroni_alpha:
            ax1.text(layer, effects[i] + 0.02, '*', ha='center', fontsize=14)
    
    # Plot 2: P-values by layer (log scale)
    ax2.bar(layers, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
    ax2.axhline(y=-np.log10(0.05), color='orange', linestyle='--', alpha=0.5, label='p=0.05')
    ax2.axhline(y=-np.log10(bonferroni_alpha), color='red', linestyle='--', 
               alpha=0.5, label=f'Bonferroni corrected (p={bonferroni_alpha:.4f})')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Statistical Significance by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = output_path / "h3_layer_effects_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def main():
    parser = argparse.ArgumentParser(
        description="Test layer specificity of causal attention patterns"
    )
    parser.add_argument("--n-samples", type=int, default=64,
                       help="Number of samples per condition (default: 64)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model to test (default: bert-base-uncased)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("CAUSAL ATTENTION GEOMETRY - HYPOTHESIS 3")
    print("Testing Layer Specificity (Middle Layers 5-8)")
    print("Hillary Danan's Research Implementation")
    print("Following Tenney et al. (2019) methodology")
    print("=" * 60)
    
    # Initialize analyzer
    print(f"\nInitializing model: {args.model}")
    analyzer = CausalAttentionAnalyzer(model_name=args.model)
    
    # Note about power analysis
    print(f"\nUsing N={args.n_samples} samples per condition")
    print("Testing across 12 layers with Bonferroni correction")
    
    # Run experiment
    print("\nRunning experiment...")
    results = run_layer_specificity_experiment(analyzer, n_samples=args.n_samples)
    
    # Save results
    output_file = output_path / "h3_layer_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization
    if args.visualize:
        print("\nGenerating visualization...")
        try:
            plot_file = visualize_layer_effects(results, output_path)
            if plot_file:
                print(f"Visualization saved to: {plot_file}")
            else:
                print("Visualization skipped due to data structure issues")
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Continuing without plot...")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Print layer group means
    print("Layer Group Effects (Cohen's d):")
    early = results['layer_group_analysis']['early_layers']
    middle = results['layer_group_analysis']['middle_layers']
    late = results['layer_group_analysis']['late_layers']
    
    print(f"  Early (0-4):   {early['mean_effect']:.3f} ± {early['std_effect']:.3f}")
    print(f"  Middle (5-8):  {middle['mean_effect']:.3f} ± {middle['std_effect']:.3f}")
    print(f"  Late (9-11):   {late['mean_effect']:.3f} ± {late['std_effect']:.3f}")
    
    # Print peak analysis
    peak = results['peak_analysis']
    print(f"\nPeak Effect:")
    print(f"  Layer {peak['peak_layer']}: d={peak['peak_effect_size']:.3f}")
    print(f"  Is middle layer: {peak['is_middle_layer']}")
    
    # Print significant layers
    sig_layers = results['statistical_tests']['significant_layers']
    if sig_layers:
        print(f"\nSignificant layers (Bonferroni corrected): {sig_layers}")
    else:
        print(f"\nNo layers significant after Bonferroni correction")
    
    # Print ANOVA results
    anova = results['statistical_tests']['anova_across_layer_groups']
    print(f"\nANOVA across layer groups:")
    print(f"  F={anova['f_statistic']:.3f}, p={anova['p_value']:.4f}")
    
    print(f"\nHypothesis Supported: {results['hypothesis_supported']}")
    print(f"\n{results['interpretation']}")
    print(f"\nResults saved to: {output_file}")
    
    return 0 if results['hypothesis_supported'] else 1


if __name__ == "__main__":
    sys.exit(main())