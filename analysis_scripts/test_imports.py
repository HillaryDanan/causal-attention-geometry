#!/usr/bin/env python3
"""
Test imports and basic functionality for causal attention geometry.
Run this first to ensure all dependencies are working.
"""

import sys
from pathlib import Path

print("Testing imports and basic functionality...")
print("=" * 60)

# Test basic imports
try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoModel, AutoTokenizer
    print("✓ Transformers imported successfully")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")
    sys.exit(1)

try:
    import spacy
    print(f"✓ spaCy imported successfully (version: {spacy.__version__})")
except ImportError as e:
    print(f"✗ spaCy import failed: {e}")
    sys.exit(1)

try:
    import scipy.stats
    print("✓ SciPy stats imported successfully")
except ImportError as e:
    print(f"✗ SciPy import failed: {e}")
    sys.exit(1)

print("\nTesting spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The rain caused flooding.")
    print(f"✓ spaCy model loaded successfully")
    print(f"  Example parse: {[(token.text, token.dep_) for token in doc]}")
except Exception as e:
    print(f"✗ spaCy model failed: {e}")
    print("  Try: python3 -m spacy download en_core_web_sm")
    sys.exit(1)

print("\nTesting causal_attention module...")
# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from causal_attention import CausalAttentionAnalyzer
    print("✓ CausalAttentionAnalyzer imported successfully")
except ImportError as e:
    print(f"✗ CausalAttentionAnalyzer import failed: {e}")
    sys.exit(1)

print("\nTesting model initialization (this may take a moment)...")
try:
    analyzer = CausalAttentionAnalyzer(model_name="bert-base-uncased")
    print("✓ BERT model loaded successfully")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

print("\nTesting basic attention extraction...")
try:
    test_text = "The glass broke because it fell."
    result = analyzer.extract_attention_patterns(test_text)
    print(f"✓ Attention extraction successful")
    print(f"  Number of layers: {len(result.layer_attentions)}")
    print(f"  First layer shape: {result.layer_attentions[0].shape}")
except Exception as e:
    print(f"✗ Attention extraction failed: {e}")
    sys.exit(1)

print("\nTesting intervention point detection...")
try:
    intervention_points = analyzer.identify_intervention_points(test_text)
    print(f"✓ Intervention detection successful")
    print(f"  Detected points: {intervention_points}")
except Exception as e:
    print(f"✗ Intervention detection failed: {e}")
    sys.exit(1)

print("\nTesting JSON serialization...")
try:
    import json
    # Test numpy bool_ conversion
    test_data = {
        "numpy_bool": bool(np.bool_(True)),
        "numpy_float": float(np.float64(3.14)),
        "numpy_int": int(np.int64(42)),
        "regular_bool": True,
        "regular_float": 3.14,
        "regular_int": 42
    }
    json_str = json.dumps(test_data)
    print("✓ JSON serialization successful")
    print(f"  Test data serialized: {json_str[:50]}...")
except Exception as e:
    print(f"✗ JSON serialization failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("You can now run the full experiments:")
print("  python3 experiments/run_all.py")
print("Or use the clean script:")
print("  bash clean_and_run.sh")
print("=" * 60)