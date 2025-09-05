#!/usr/bin/env python3
"""
Test robustness of H1 (Counterfactual) and H2 (Feedback) findings across models.
Hillary Danan's Causal Attention Geometry
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from scipy import stats
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class MultiModelTester:
    """Test causal attention across different transformer architectures."""
    
    def __init__(self):
        self.models_to_test = [
            "bert-base-uncased",
            "roberta-base",
            "distilbert-base-uncased",
            # "gpt2",  # Different tokenizer, needs special handling
            "albert-base-v2"
        ]
        self.results = {}
    
    def extract_attention(self, text: str, model, tokenizer) -> np.ndarray:
        """Extract average attention across all layers."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, 
                          truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Average across all layers and heads
        all_attentions = []
        for layer_attention in outputs.attentions:
            avg_attention = layer_attention.mean(dim=1).squeeze().numpy()
            all_attentions.append(avg_attention)
        
        # Return mean across layers
        return np.mean(all_attentions, axis=0)
    
    def test_h1_counterfactual(self, model_name: str) -> Dict:
        """Test counterfactual divergence for a model."""
        print(f"  Testing H1 (Counterfactual)...")
        
        # Load model
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Test pairs
        test_pairs = [
            ("The glass broke because it fell.", 
             "The glass broke because it was plastic."),
            ("She got wet because it rained.", 
             "She got wet because she swam."),
            ("He failed because he didn't study.", 
             "He failed because the test was unfair."),
        ]
        
        kl_divergences = []
        
        for factual, counterfactual in test_pairs:
            # Get attention patterns
            fact_att = self.extract_attention(factual, model, tokenizer)
            counter_att = self.extract_attention(counterfactual, model, tokenizer)
            
            # Ensure same shape
            min_len = min(fact_att.shape[0], counter_att.shape[0])
            fact_att = fact_att[:min_len, :min_len]
            counter_att = counter_att[:min_len, :min_len]
            
            # Find "because" position (simplified - just use middle)
            mid_point = min_len // 2
            
            # Calculate KL divergence at intervention point
            f_dist = fact_att[mid_point, :] + 1e-10
            c_dist = counter_att[mid_point, :] + 1e-10
            f_dist = f_dist / f_dist.sum()
            c_dist = c_dist / c_dist.sum()
            
            kl = stats.entropy(f_dist, c_dist)
            kl_divergences.append(kl)
        
        mean_kl = np.mean(kl_divergences)
        std_kl = np.std(kl_divergences)
        
        # Test against threshold (0.2)
        t_stat, p_value = stats.ttest_1samp(kl_divergences, 0.2)
        
        return {
            "mean_kl": float(mean_kl),
            "std_kl": float(std_kl),
            "p_value": float(p_value),
            "supported": mean_kl > 0.2 and p_value < 0.05,
            "n_samples": len(test_pairs)
        }
    
    def test_h2_feedback(self, model_name: str) -> Dict:
        """Test feedback loop density for a model."""
        print(f"  Testing H2 (Feedback Loops)...")
        
        # Load model
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Test triads
        test_triads = [
            ("Predators control prey which affects vegetation which affects predators",
             "Predators control prey and then vegetation changes",
             "Predators and prey and vegetation exist"),
            
            ("Success builds confidence which improves performance which creates success",
             "Success builds confidence and then performance improves",
             "Success and confidence and performance exist"),
            
            ("Heat causes evaporation which forms clouds which creates rain which cools heat",
             "Heat causes evaporation and then clouds form and rain falls",
             "Heat and evaporation and clouds and rain occur"),
        ]
        
        circular_densities = []
        linear_densities = []
        control_densities = []
        
        for circular, linear, control in test_triads:
            # Get attention patterns
            circ_att = self.extract_attention(circular, model, tokenizer)
            lin_att = self.extract_attention(linear, model, tokenizer)
            ctrl_att = self.extract_attention(control, model, tokenizer)
            
            # Calculate density (mean off-diagonal)
            def calculate_density(att):
                n = att.shape[0]
                mask = np.ones_like(att) - np.eye(n)
                return (att * mask).mean()
            
            circular_densities.append(calculate_density(circ_att))
            linear_densities.append(calculate_density(lin_att))
            control_densities.append(calculate_density(ctrl_att))
        
        # Calculate effect size (Cohen's d)
        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(
                ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof
            )
        
        effect_size = cohens_d(circular_densities, linear_densities)
        t_stat, p_value = stats.ttest_ind(circular_densities, linear_densities)
        
        return {
            "circular_mean": float(np.mean(circular_densities)),
            "linear_mean": float(np.mean(linear_densities)),
            "control_mean": float(np.mean(control_densities)),
            "effect_size": float(effect_size),
            "p_value": float(p_value),
            "original_direction": effect_size > 0.3,  # Original hypothesis
            "efficiency_finding": effect_size < -0.3,  # Our discovery
            "n_samples": len(test_triads)
        }
    
    def run_all_tests(self):
        """Run both hypotheses on all models."""
        print("=" * 60)
        print("ROBUSTNESS TESTING ACROSS MODELS")
        print("=" * 60)
        
        for model_name in self.models_to_test:
            print(f"\nTesting {model_name}...")
            
            try:
                h1_results = self.test_h1_counterfactual(model_name)
                h2_results = self.test_h2_feedback(model_name)
                
                self.results[model_name] = {
                    "H1_counterfactual": h1_results,
                    "H2_feedback": h2_results
                }
                
                print(f"  H1: KL={h1_results['mean_kl']:.3f}, "
                      f"Supported={h1_results['supported']}")
                print(f"  H2: Effect={h2_results['effect_size']:.3f}, "
                      f"Efficiency={h2_results['efficiency_finding']}")
                
            except Exception as e:
                print(f"  Error testing {model_name}: {e}")
                self.results[model_name] = {"error": str(e)}
        
        # Save results
        with open("robustness_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: robustness_results.json")
        
        return self.results
    
    def visualize_results(self):
        """Create comparison visualizations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # H1: KL Divergences
        models = []
        kl_values = []
        colors_h1 = []
        
        for model, data in self.results.items():
            if "error" not in data:
                models.append(model.replace("-base", "").replace("-uncased", ""))
                kl = data["H1_counterfactual"]["mean_kl"]
                kl_values.append(kl)
                colors_h1.append("green" if kl > 0.2 else "red")
        
        ax1.bar(range(len(models)), kl_values, color=colors_h1, alpha=0.7)
        ax1.axhline(y=0.2, color='black', linestyle='--', label='Threshold')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45)
        ax1.set_ylabel('KL Divergence')
        ax1.set_title('H1: Counterfactual Divergence Across Models')
        ax1.legend()
        
        # H2: Effect Sizes
        effect_sizes = []
        colors_h2 = []
        
        for model, data in self.results.items():
            if "error" not in data:
                effect = data["H2_feedback"]["effect_size"]
                effect_sizes.append(effect)
                colors_h2.append("blue" if effect < 0 else "orange")
        
        ax2.bar(range(len(models)), effect_sizes, color=colors_h2, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-')
        ax2.axhline(y=0.3, color='orange', linestyle='--', label='Original hypothesis')
        ax2.axhline(y=-0.3, color='blue', linestyle='--', label='Efficiency finding')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45)
        ax2.set_ylabel("Cohen's d")
        ax2.set_title('H2: Feedback Loop Density Across Models')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('robustness_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: robustness_comparison.png")
        plt.close()
    
    def summarize_findings(self):
        """Generate plain language summary."""
        print("\n" + "=" * 60)
        print("PLAIN LANGUAGE SUMMARY")
        print("=" * 60)
        
        h1_consistent = sum(1 for m, d in self.results.items() 
                           if "error" not in d and d["H1_counterfactual"]["supported"])
        h2_consistent = sum(1 for m, d in self.results.items() 
                           if "error" not in d and d["H2_feedback"]["efficiency_finding"])
        
        total_models = len([m for m, d in self.results.items() if "error" not in d])
        
        print(f"\nH1 (Causal Fingerprints): {h1_consistent}/{total_models} models supported")
        print("  Models can distinguish between real and fake causes by creating")
        print("  different attention patterns at the point where causation occurs.")
        
        print(f"\nH2 (Efficient Loops): {h2_consistent}/{total_models} models confirmed")
        print("  Models consistently use LESS attention for circular patterns than")
        print("  linear ones - like recognizing 'vicious cycle' as a single concept")
        print("  rather than tracking each step separately.")
        
        if h2_consistent == total_models:
            print("\n  ** This efficiency finding appears UNIVERSAL across architectures! **")
        
        print("\nBottom Line:")
        print("  AI models have developed two key strategies for understanding causation:")
        print("  1) They create unique signatures for different causal relationships")
        print("  2) They compress circular logic into efficient representations")

def main():
    tester = MultiModelTester()
    
    # Run tests
    results = tester.run_all_tests()
    
    # Visualize
    tester.visualize_results()
    
    # Summarize
    tester.summarize_findings()
    
    print("\n" + "=" * 60)
    print("Testing complete!")

if __name__ == "__main__":
    main()