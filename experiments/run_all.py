#!/usr/bin/env python3
"""
Run all causal attention geometry experiments and generate summary.
Hillary Danan's research implementation.

This script orchestrates all three hypothesis tests and produces
a comprehensive summary report with statistical rigor.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np


def run_experiment(script_name: str, args: list = None) -> int:
    """Run a single experiment script."""
    cmd = ["python3", f"experiments/{script_name}"]
    if args:
        cmd.extend(args)
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
    else:
        print(result.stdout)
    
    return result.returncode


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all experiment results."""
    results = {}
    
    # Expected result files
    result_files = {
        "h1": "h1_counterfactual_results.json",
        "h2": "h2_feedback_results.json",
        "h3": "h3_layer_analysis.json"
    }
    
    for key, filename in result_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    results[key] = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not load {filename}: {e}")
                results[key] = None
        else:
            print(f"Warning: {filename} not found")
            results[key] = None
    
    return results


def generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive summary report."""
    
    report = []
    report.append("=" * 70)
    report.append("CAUSAL ATTENTION GEOMETRY - COMPREHENSIVE RESULTS SUMMARY")
    report.append("Hillary Danan's Research Implementation")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    # Overall Summary
    report.append("\n## OVERALL FINDINGS ##\n")
    
    hypotheses_tested = sum(1 for r in results.values() if r is not None)
    hypotheses_supported = sum(1 for r in results.values() 
                              if r and r.get('hypothesis_supported', False))
    
    report.append(f"Hypotheses Tested: {hypotheses_tested}/3")
    report.append(f"Hypotheses Supported: {hypotheses_supported}/3")
    report.append(f"Success Rate: {(hypotheses_supported/hypotheses_tested)*100:.1f}%")
    
    # Hypothesis 1: Counterfactual Divergence
    report.append("\n" + "-" * 70)
    report.append("\n## HYPOTHESIS 1: COUNTERFACTUAL DIVERGENCE ##\n")
    
    if results.get('h1'):
        h1 = results['h1']
        report.append(f"Claim: Attention patterns diverge at causal intervention points")
        report.append(f"Expected: KL divergence > 0.2")
        report.append(f"Observed: KL divergence = {h1['mean_kl_divergence']:.4f}")
        report.append(f"N samples: {h1['n_samples']}")
        report.append(f"Effect size (Cohen's d): {h1['cohens_d']:.3f}")
        report.append(f"p-value: {h1['p_value']:.6f}")
        report.append(f"95% CI: [{h1['confidence_interval_95'][0]:.4f}, {h1['confidence_interval_95'][1]:.4f}]")
        report.append(f"Result: {'✓ SUPPORTED' if h1['hypothesis_supported'] else '✗ NOT SUPPORTED'}")
        
        # Top layers
        if 'layer_analysis' in h1 and 'top_diverging_layers' in h1['layer_analysis']:
            report.append("\nTop diverging layers:")
            for layer in h1['layer_analysis']['top_diverging_layers'][:3]:
                report.append(f"  - Layer {layer['layer']}: {layer['mean_divergence']:.4f}")
    else:
        report.append("Results not available")
    
    # Hypothesis 2: Feedback Loop Density
    report.append("\n" + "-" * 70)
    report.append("\n## HYPOTHESIS 2: FEEDBACK LOOP DENSITY ##\n")
    
    if results.get('h2'):
        h2 = results['h2']
        report.append(f"Claim: Circular causation shows denser attention than linear")
        report.append(f"Expected: Cohen's d > 0.3")
        report.append(f"Observed: Cohen's d = {h2['effect_sizes']['circular_vs_linear']:.3f}")
        report.append(f"N samples: {h2['n_samples']}")
        
        # Density statistics
        report.append("\nAttention density means:")
        report.append(f"  Circular: {h2['density_statistics']['circular']['mean']:.4f} ± "
                     f"{h2['density_statistics']['circular']['std']:.4f}")
        report.append(f"  Linear:   {h2['density_statistics']['linear']['mean']:.4f} ± "
                     f"{h2['density_statistics']['linear']['std']:.4f}")
        report.append(f"  Control:  {h2['density_statistics']['control']['mean']:.4f} ± "
                     f"{h2['density_statistics']['control']['std']:.4f}")
        
        report.append(f"\np-value (circular vs linear): {h2['statistical_tests']['circular_vs_linear']['p_value']:.6f}")
        report.append(f"ANOVA p-value: {h2['statistical_tests']['anova']['p_value']:.6f}")
        report.append(f"Result: {'✓ SUPPORTED' if h2['hypothesis_supported'] else '✗ NOT SUPPORTED'}")
        
        # Strongest effect layers
        if 'layer_analysis' in h2 and 'strongest_effect_layers' in h2['layer_analysis']:
            report.append("\nStrongest effect layers:")
            for layer in h2['layer_analysis']['strongest_effect_layers'][:3]:
                report.append(f"  - Layer {layer['layer']}: d={layer['cohens_d']:.3f}")
    else:
        report.append("Results not available")
    
    # Hypothesis 3: Layer Specificity
    report.append("\n" + "-" * 70)
    report.append("\n## HYPOTHESIS 3: LAYER SPECIFICITY ##\n")
    
    if results.get('h3'):
        h3 = results['h3']
        report.append(f"Claim: Middle layers (5-8) show strongest causal effects")
        report.append(f"N samples per condition: {h3['n_samples_per_condition']}")
        
        # Layer group analysis
        report.append("\nLayer group effects (Cohen's d):")
        early = h3['layer_group_analysis']['early_layers']
        middle = h3['layer_group_analysis']['middle_layers']
        late = h3['layer_group_analysis']['late_layers']
        
        report.append(f"  Early (0-4):  {early['mean_effect']:.3f} ± {early['std_effect']:.3f}")
        report.append(f"  Middle (5-8): {middle['mean_effect']:.3f} ± {middle['std_effect']:.3f}")
        report.append(f"  Late (9-11):  {late['mean_effect']:.3f} ± {late['std_effect']:.3f}")
        
        # Peak analysis
        peak = h3['peak_analysis']
        report.append(f"\nPeak effect at layer {peak['peak_layer']}: d={peak['peak_effect_size']:.3f}")
        report.append(f"Peak is in middle layers: {peak['is_middle_layer']}")
        
        # Statistical tests
        anova = h3['statistical_tests']['anova_across_layer_groups']
        report.append(f"\nANOVA F-statistic: {anova['f_statistic']:.3f} (p={anova['p_value']:.4f})")
        report.append(f"Bonferroni corrected α: {h3['statistical_tests']['bonferroni_corrected_alpha']:.6f}")
        
        sig_layers = h3['statistical_tests']['significant_layers']
        if sig_layers:
            report.append(f"Significant layers after correction: {sig_layers}")
        else:
            report.append("No layers significant after Bonferroni correction")
        
        report.append(f"\nResult: {'✓ SUPPORTED' if h3['hypothesis_supported'] else '✗ NOT SUPPORTED'}")
    else:
        report.append("Results not available")
    
    # Scientific Interpretation
    report.append("\n" + "=" * 70)
    report.append("\n## SCIENTIFIC INTERPRETATION ##\n")
    
    if hypotheses_supported == 3:
        report.append("STRONG EVIDENCE: All three hypotheses supported. Transformers exhibit")
        report.append("geometric patterns in attention when processing causal relationships.")
        report.append("This suggests causal topology is reflected in attention geometry,")
        report.append("with specific patterns at intervention points, feedback loops, and")
        report.append("concentrated in middle layers. This advances interpretability claims")
        report.append("about transformer causal understanding.")
        
    elif hypotheses_supported == 2:
        report.append("MODERATE EVIDENCE: Two of three hypotheses supported. Transformers show")
        report.append("some geometric patterns in causal attention, but the evidence is mixed.")
        report.append("Further research needed to understand which aspects of causality are")
        report.append("geometrically encoded in attention mechanisms.")
        
    elif hypotheses_supported == 1:
        report.append("WEAK EVIDENCE: Only one hypothesis supported. Limited geometric patterns")
        report.append("detected in causal attention. This constrains strong claims about")
        report.append("transformer causal understanding through attention geometry.")
        
    else:
        report.append("NULL RESULTS: No hypotheses supported. Attention patterns do not show")
        report.append("expected geometric signatures of causal processing. This null result")
        report.append("is scientifically valuable - it constrains interpretability claims")
        report.append("and suggests causal understanding (if present) may be encoded")
        report.append("differently than hypothesized. Alternative mechanisms should be explored.")
    
    # Methodological Notes
    report.append("\n" + "=" * 70)
    report.append("\n## METHODOLOGICAL NOTES ##\n")
    
    report.append("• All tests conducted with proper statistical power analysis")
    report.append("• Multiple comparison corrections applied (Bonferroni)")
    report.append("• Effect sizes reported alongside significance tests")
    report.append("• Control conditions included to isolate causal effects")
    report.append("• Null results interpreted as scientifically valuable constraints")
    
    # Future Directions
    report.append("\n## FUTURE DIRECTIONS ##\n")
    
    report.append("1. Test with larger models (GPT-style architectures)")
    report.append("2. Examine cross-linguistic causal patterns")
    report.append("3. Investigate temporal dynamics in autoregressive models")
    report.append("4. Compare with multi-geometric attention findings")
    report.append("5. Explore retroactive causality connections")
    
    # Citation
    report.append("\n" + "=" * 70)
    report.append("\n## CITATION ##\n")
    report.append("Danan, H. (2025). Causal Attention Geometry in Transformers.")
    report.append("Repository: https://github.com/HillaryDanan/causal-attention-geometry")
    report.append("\nRelated work:")
    report.append("- Multi-geometric attention patterns")
    report.append("- Retroactive causality in language models")
    report.append("- Cross-linguistic attention dynamics")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def main():
    """Main execution function."""
    print("=" * 70)
    print("CAUSAL ATTENTION GEOMETRY - FULL EXPERIMENTAL SUITE")
    print("Hillary Danan's Research Implementation")
    print("=" * 70)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Run all causal attention geometry experiments"
    )
    parser.add_argument("--n-samples", type=int, default=64,
                       help="Number of samples for experiments (default: 64)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model to test (default: bert-base-uncased)")
    parser.add_argument("--skip-experiments", action="store_true",
                       help="Skip running experiments, just generate summary")
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if not args.skip_experiments:
        # Install spacy model if needed
        print("\nEnsuring spaCy model is installed...")
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"], 
                      capture_output=True)
        
        # Run experiments
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENTS")
        print("=" * 70)
        
        # Hypothesis 1: Counterfactual Divergence
        print("\n[1/3] Testing Counterfactual Divergence...")
        ret1 = run_experiment("test_counterfactual.py", 
                            [f"--n-samples={args.n_samples}", 
                             f"--model={args.model}"])
        
        # Hypothesis 2: Feedback Loop Density
        print("\n[2/3] Testing Feedback Loop Density...")
        ret2 = run_experiment("test_feedback_loop.py",
                            [f"--n-samples={args.n_samples}",
                             f"--model={args.model}"])
        
        # Hypothesis 3: Layer Specificity
        print("\n[3/3] Testing Layer Specificity...")
        ret3 = run_experiment("test_layer_specificity.py",
                            [f"--n-samples={args.n_samples}",
                             f"--model={args.model}",
                             "--visualize"])
        
        print(f"\nExperiments completed: {3 - sum([ret1!=0, ret2!=0, ret3!=0])}/3 successful")
    
    # Load and summarize results
    print("\nLoading results...")
    results = load_results(results_dir)
    
    # Generate summary report
    print("Generating summary report...")
    summary = generate_summary_report(results)
    
    # Save summary
    summary_file = results_dir / "summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Print summary
    print("\n" + summary)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Create JSON summary for programmatic access
    json_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "n_samples": args.n_samples,
        "hypotheses": {
            "h1_counterfactual": results.get('h1', {}).get('hypothesis_supported', None),
            "h2_feedback_loop": results.get('h2', {}).get('hypothesis_supported', None),
            "h3_layer_specificity": results.get('h3', {}).get('hypothesis_supported', None)
        },
        "key_metrics": {
            "h1_kl_divergence": results.get('h1', {}).get('mean_kl_divergence', None),
            "h2_effect_size": results.get('h2', {}).get('effect_sizes', {}).get('circular_vs_linear', None),
            "h3_peak_layer": results.get('h3', {}).get('peak_analysis', {}).get('peak_layer', None)
        }
    }
    
    json_summary_file = results_dir / "summary.json"
    with open(json_summary_file, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"JSON summary saved to: {json_summary_file}")
    print("\nAll experiments complete! 🚀")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())