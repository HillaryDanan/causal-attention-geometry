"""
Extract attention patterns from transformer models
"""

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple

class AttentionExtractor:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        
    def extract_attention(self, text: str, layer: int = -1) -> np.ndarray:
        """Extract attention matrix for specific layer"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get attention from specified layer (default: last)
        attention = outputs.attentions[layer]
        
        # Average over heads: [batch, heads, seq, seq] -> [seq, seq]
        attention = attention.mean(dim=1).squeeze().numpy()
        
        return attention

    def measure_divergence(self, text1: str, text2: str, position: int) -> float:
        """Measure KL divergence at specific position"""
        attn1 = self.extract_attention(text1)
        attn2 = self.extract_attention(text2)
        
        # Get distributions at position
        dist1 = attn1[position] + 1e-10
        dist2 = attn2[position] + 1e-10
        
        # Normalize
        dist1 = dist1 / dist1.sum()
        dist2 = dist2 / dist2.sum()
        
        # KL divergence
        kl = np.sum(dist1 * np.log(dist1 / dist2))
        
        return kl
