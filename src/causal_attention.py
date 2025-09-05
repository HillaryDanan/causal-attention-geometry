"""
Core module for causal attention geometry analysis.
Hillary Danan's research on geometric patterns in transformer attention.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import spacy
from scipy import stats
from dataclasses import dataclass
import json


@dataclass
class AttentionResult:
    """Container for attention analysis results."""
    layer_attentions: Dict[int, np.ndarray]
    divergence_scores: Optional[Dict[int, float]] = None
    density_metrics: Optional[Dict[int, float]] = None
    intervention_points: Optional[List[int]] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None


class CausalAttentionAnalyzer:
    """
    Analyzer for testing causal attention geometry hypotheses.
    Tests whether transformers show geometric patterns when processing causal relationships.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize with specified transformer model."""
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.model.eval()
        
    def extract_attention_patterns(self, text: str) -> AttentionResult:
        """
        Extract layer-wise attention matrices from text.
        
        Critical: Don't average across layers initially - keep layer information
        for hypothesis 3 (layer specificity).
        """
        # Tokenize and get model outputs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract attention from all layers
        attentions = outputs.attentions  # Tuple of layers
        
        # Convert to numpy and organize by layer
        layer_attentions = {}
        for layer_idx, layer_attention in enumerate(attentions):
            # Shape: [batch, heads, seq_len, seq_len]
            # Average over heads but keep layer separate
            avg_attention = layer_attention.mean(dim=1).squeeze().numpy()
            layer_attentions[layer_idx] = avg_attention
            
        return AttentionResult(layer_attentions=layer_attentions)
    
    def identify_intervention_points(self, text: str) -> List[int]:
        """
        Use dependency parsing to identify causal intervention points.
        
        Looking for:
        - mark, advcl dependencies
        - Causal verbs: cause, lead, result, make, affect
        - Causal markers: because, since, therefore, thus, hence
        """
        doc = self.nlp(text)
        intervention_points = []
        
        # Causal dependency patterns
        causal_deps = {'mark', 'advcl', 'ccomp'}
        causal_verbs = {'cause', 'causes', 'caused', 'lead', 'leads', 'led', 
                       'result', 'results', 'resulted', 'make', 'makes', 'made',
                       'affect', 'affects', 'affected', 'influence', 'influences'}
        causal_markers = {'because', 'since', 'therefore', 'thus', 'hence', 
                         'consequently', 'so', 'if', 'when'}
        
        for token in doc:
            # Check dependency relations
            if token.dep_ in causal_deps:
                intervention_points.append(token.i)
            # Check causal verbs
            if token.lemma_.lower() in causal_verbs:
                intervention_points.append(token.i)
            # Check causal markers
            if token.text.lower() in causal_markers:
                intervention_points.append(token.i)
                
        return sorted(list(set(intervention_points)))
    
    def test_counterfactual_divergence(self, factual: str, counterfactual: str) -> Dict:
        """
        Test Hypothesis 1: Attention patterns diverge at causal intervention points.
        
        Expected: KL divergence > 0.2 between factual/counterfactual pairs.
        """
        # Extract attention for both versions
        factual_result = self.extract_attention_patterns(factual)
        counter_result = self.extract_attention_patterns(counterfactual)
        
        # Identify intervention points
        intervention_points = self.identify_intervention_points(factual)
        
        # Calculate KL divergence at intervention points for each layer
        layer_divergences = {}
        
        for layer_idx in factual_result.layer_attentions:
            factual_att = factual_result.layer_attentions[layer_idx]
            counter_att = counter_result.layer_attentions[layer_idx]
            
            # Ensure same shape
            min_len = min(factual_att.shape[0], counter_att.shape[0])
            factual_att = factual_att[:min_len, :min_len]
            counter_att = counter_att[:min_len, :min_len]
            
            if intervention_points:
                # Calculate divergence at intervention points
                divergences = []
                for point in intervention_points:
                    if point < min_len:
                        # Get attention distributions at this point
                        f_dist = factual_att[point, :] + 1e-10  # Add small epsilon
                        c_dist = counter_att[point, :] + 1e-10
                        
                        # Normalize
                        f_dist = f_dist / f_dist.sum()
                        c_dist = c_dist / c_dist.sum()
                        
                        # KL divergence
                        kl_div = stats.entropy(f_dist, c_dist)
                        divergences.append(kl_div)
                
                layer_divergences[layer_idx] = np.mean(divergences) if divergences else 0.0
            else:
                layer_divergences[layer_idx] = 0.0
        
        # Calculate overall statistics
        all_divergences = list(layer_divergences.values())
        mean_divergence = np.mean(all_divergences)
        
        # Statistical test: one-sample t-test against null hypothesis (divergence = 0.2)
        t_stat, p_value = stats.ttest_1samp(all_divergences, 0.2)
        
        # Effect size (Cohen's d)
        effect_size = (mean_divergence - 0.2) / np.std(all_divergences) if np.std(all_divergences) > 0 else 0
        
        return {
            "hypothesis": "counterfactual_divergence",
            "mean_kl_divergence": float(mean_divergence),
            "layer_divergences": {k: float(v) for k, v in layer_divergences.items()},
            "intervention_points": intervention_points,
            "supported": mean_divergence > 0.2,
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "t_statistic": float(t_stat)
        }
    
    def calculate_attention_density(self, attention_matrix: np.ndarray) -> float:
        """
        Calculate attention density metric for feedback loops.
        
        Density = sum of off-diagonal attention weights (indicating connections)
        """
        if len(attention_matrix.shape) != 2:
            return 0.0
            
        # Create mask for off-diagonal elements
        mask = np.ones_like(attention_matrix) - np.eye(attention_matrix.shape[0])
        
        # Calculate density as mean of off-diagonal attention
        density = (attention_matrix * mask).mean()
        
        return float(density)
    
    def test_feedback_loop_density(self, circular_text: str, linear_text: str, 
                                  control_text: str) -> Dict:
        """
        Test Hypothesis 2: Circular causation shows denser attention than linear.
        
        Expected: Cohen's d > 0.3 between conditions.
        """
        # Extract attention for all conditions
        circular_result = self.extract_attention_patterns(circular_text)
        linear_result = self.extract_attention_patterns(linear_text)
        control_result = self.extract_attention_patterns(control_text)
        
        # Calculate density for each layer and condition
        circular_densities = []
        linear_densities = []
        control_densities = []
        
        for layer_idx in circular_result.layer_attentions:
            circ_density = self.calculate_attention_density(
                circular_result.layer_attentions[layer_idx]
            )
            lin_density = self.calculate_attention_density(
                linear_result.layer_attentions[layer_idx]
            )
            cont_density = self.calculate_attention_density(
                control_result.layer_attentions[layer_idx]
            )
            
            circular_densities.append(circ_density)
            linear_densities.append(lin_density)
            control_densities.append(cont_density)
        
        # Calculate effect sizes (Cohen's d)
        def cohens_d(x, y):
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(
                ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof
            )
        
        # Effect size: circular vs linear
        d_circ_lin = cohens_d(circular_densities, linear_densities)
        
        # Effect size: circular vs control
        d_circ_control = cohens_d(circular_densities, control_densities)
        
        # Statistical tests
        t_stat_cl, p_value_cl = stats.ttest_ind(circular_densities, linear_densities)
        t_stat_cc, p_value_cc = stats.ttest_ind(circular_densities, control_densities)
        
        return {
            "hypothesis": "feedback_loop_density",
            "mean_circular_density": float(np.mean(circular_densities)),
            "mean_linear_density": float(np.mean(linear_densities)),
            "mean_control_density": float(np.mean(control_densities)),
            "effect_size_circular_vs_linear": float(d_circ_lin),
            "effect_size_circular_vs_control": float(d_circ_control),
            "p_value_circular_vs_linear": float(p_value_cl),
            "p_value_circular_vs_control": float(p_value_cc),
            "supported": bool(d_circ_lin > 0.3),
            "layer_densities": {
                "circular": [float(d) for d in circular_densities],
                "linear": [float(d) for d in linear_densities],
                "control": [float(d) for d in control_densities]
            }
        }
    
    def test_layer_specificity(self, causal_texts: List[str], 
                              non_causal_texts: List[str]) -> Dict:
        """
        Test Hypothesis 3: Middle layers (5-8) show strongest causal effects.
        
        Following Tenney et al. (2019) layer-wise analysis approach.
        """
        # Extract attention for all texts
        causal_layer_patterns = {i: [] for i in range(12)}  # Assuming 12-layer model
        non_causal_layer_patterns = {i: [] for i in range(12)}
        
        for text in causal_texts:
            result = self.extract_attention_patterns(text)
            intervention_points = self.identify_intervention_points(text)
            
            for layer_idx, attention in result.layer_attentions.items():
                if intervention_points:
                    # Extract attention at intervention points
                    intervention_attention = []
                    for point in intervention_points:
                        if point < attention.shape[0]:
                            intervention_attention.append(attention[point, :].mean())
                    if intervention_attention:
                        causal_layer_patterns[layer_idx].append(np.mean(intervention_attention))
        
        for text in non_causal_texts:
            result = self.extract_attention_patterns(text)
            for layer_idx, attention in result.layer_attentions.items():
                # Use mean attention as baseline
                non_causal_layer_patterns[layer_idx].append(attention.mean())
        
        # Calculate effect sizes per layer
        layer_effect_sizes = {}
        layer_p_values = {}
        
        for layer_idx in range(12):
            if causal_layer_patterns[layer_idx] and non_causal_layer_patterns[layer_idx]:
                # Calculate Cohen's d
                causal_vals = np.array(causal_layer_patterns[layer_idx])
                non_causal_vals = np.array(non_causal_layer_patterns[layer_idx])
                
                # Effect size
                pooled_std = np.sqrt(
                    ((len(causal_vals)-1)*np.std(causal_vals)**2 + 
                     (len(non_causal_vals)-1)*np.std(non_causal_vals)**2) / 
                    (len(causal_vals) + len(non_causal_vals) - 2)
                )
                
                if pooled_std > 0:
                    d = (causal_vals.mean() - non_causal_vals.mean()) / pooled_std
                else:
                    d = 0.0
                    
                layer_effect_sizes[layer_idx] = float(d)
                
                # Statistical test
                t_stat, p_val = stats.ttest_ind(causal_vals, non_causal_vals)
                layer_p_values[layer_idx] = float(p_val)
            else:
                layer_effect_sizes[layer_idx] = 0.0
                layer_p_values[layer_idx] = 1.0
        
        # Check if middle layers (5-8) show strongest effects
        middle_layers = [5, 6, 7, 8]
        middle_effects = [layer_effect_sizes[i] for i in middle_layers if i in layer_effect_sizes]
        other_effects = [layer_effect_sizes[i] for i in layer_effect_sizes if i not in middle_layers]
        
        middle_mean = np.mean(middle_effects) if middle_effects else 0
        other_mean = np.mean(other_effects) if other_effects else 0
        
        # Apply Bonferroni correction for multiple comparisons
        bonferroni_alpha = 0.05 / 12  # 12 layers
        significant_layers = [i for i, p in layer_p_values.items() if p < bonferroni_alpha]
        
        return {
            "hypothesis": "layer_specificity",
            "layer_effect_sizes": layer_effect_sizes,
            "layer_p_values": layer_p_values,
            "middle_layers_mean_effect": float(middle_mean),
            "other_layers_mean_effect": float(other_mean),
            "supported": middle_mean > other_mean and any(i in significant_layers for i in middle_layers),
            "significant_layers": significant_layers,
            "bonferroni_corrected_alpha": float(bonferroni_alpha)
        }
    
    def generate_controls(self, causal_text: str) -> Dict[str, str]:
        """
        Generate control conditions for causal text.
        
        - Temporal: Replace "because" → "and then"
        - Scrambled: Preserve function words, scramble content
        - Conjunction: Replace causal markers with "and"
        """
        # Temporal control
        temporal = causal_text
        for marker in ['because', 'since', 'therefore', 'thus', 'hence']:
            temporal = temporal.replace(marker, 'and then')
        
        # Conjunction control
        conjunction = causal_text
        for marker in ['because', 'since', 'therefore', 'thus', 'hence', 'so']:
            conjunction = conjunction.replace(marker, 'and')
        
        # Scrambled control (preserve structure)
        doc = self.nlp(causal_text)
        content_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        function_words = [token.text for token in doc if token.pos_ not in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        
        # Shuffle content words
        import random
        random.shuffle(content_words)
        
        # Reconstruct with scrambled content
        scrambled_tokens = []
        content_idx = 0
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and content_idx < len(content_words):
                scrambled_tokens.append(content_words[content_idx])
                content_idx += 1
            else:
                scrambled_tokens.append(token.text)
        
        scrambled = ' '.join(scrambled_tokens)
        
        return {
            "temporal": temporal,
            "conjunction": conjunction,
            "scrambled": scrambled
        }
    
    def calculate_power_analysis(self, effect_size: float = 0.5, alpha: float = 0.05,
                                power: float = 0.8) -> int:
        """
        Calculate required sample size for desired statistical power.
        
        Using standard power analysis for t-tests.
        """
        from scipy.stats import norm
        
        # Z-scores for alpha and power
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) ** 2 * 2) / (effect_size ** 2)
        
        return int(np.ceil(n))