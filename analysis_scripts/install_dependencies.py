#!/usr/bin/env python3
"""
Install and verify all dependencies for causal attention geometry experiments.
"""

import subprocess
import sys
import importlib

def run_command(cmd):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_module(module_name):
    """Check if a Python module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install all required dependencies."""
    
    print("=" * 60)
    print("INSTALLING DEPENDENCIES FOR CAUSAL ATTENTION GEOMETRY")
    print("=" * 60)
    
    # List of required packages
    packages = [
        ("numpy", "numpy>=1.24.0"),
        ("scipy", "scipy>=1.10.0"),
        ("torch", "torch>=2.0.0"),
        ("transformers", "transformers>=4.30.0"),
        ("spacy", "spacy>=3.5.0"),
        ("tqdm", "tqdm>=4.65.0"),
        ("matplotlib", "matplotlib>=3.7.0"),
        ("pandas", "pandas>=2.0.0"),
        ("seaborn", "seaborn>=0.12.0")
    ]
    
    print("\nChecking Python version...")
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"⚠ Python {python_version.major}.{python_version.minor} detected. Python 3.8+ recommended.")
    
    print("\nInstalling/upgrading packages...")
    
    for module_name, package_spec in packages:
        if check_module(module_name):
            print(f"✓ {module_name} already installed")
        else:
            print(f"Installing {package_spec}...")
            success, stdout, stderr = run_command(f"pip install {package_spec}")
            if success:
                print(f"✓ {module_name} installed successfully")
            else:
                print(f"✗ Failed to install {module_name}")
                print(f"  Error: {stderr}")
                
    # Special handling for PyTorch with CPU-only option
    if not check_module("torch"):
        print("\nAttempting PyTorch CPU-only installation...")
        success, stdout, stderr = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        )
        if success:
            print("✓ PyTorch (CPU) installed successfully")
        else:
            print("✗ PyTorch installation failed")
            print("  Please install manually: https://pytorch.org/get-started/locally/")
    
    # Install spaCy model
    print("\nInstalling spaCy English model...")
    success, stdout, stderr = run_command("python3 -m spacy download en_core_web_sm")
    if success or "already installed" in stdout.lower() or "already installed" in stderr.lower():
        print("✓ spaCy English model ready")
    else:
        print("⚠ spaCy model installation may have issues")
        print("  Try manually: python3 -m spacy download en_core_web_sm")
    
    # Test critical imports
    print("\nVerifying critical imports...")
    
    critical_tests = [
        ("NumPy", "import numpy as np"),
        ("PyTorch", "import torch"),
        ("Transformers", "from transformers import AutoModel, AutoTokenizer"),
        ("spaCy", "import spacy"),
        ("SciPy", "from scipy import stats")
    ]
    
    all_passed = True
    for name, import_statement in critical_tests:
        try:
            exec(import_statement)
            print(f"✓ {name} imports correctly")
        except Exception as e:
            print(f"✗ {name} import failed: {e}")
            all_passed = False
    
    # Test spaCy model
    print("\nTesting spaCy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Test sentence.")
        print(f"✓ spaCy model loads correctly")
    except Exception as e:
        print(f"✗ spaCy model failed: {e}")
        all_passed = False
    
    # Test transformers model download
    print("\nTesting transformers model access...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✓ BERT tokenizer downloads correctly")
    except Exception as e:
        print(f"⚠ BERT download test failed: {e}")
        print("  This might work when running experiments")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("SUCCESS! All dependencies installed and verified.")
        print("\nYou can now run:")
        print("  python3 test_imports.py    # Test basic functionality")
        print("  python3 experiments/run_all.py  # Run all experiments")
    else:
        print("PARTIAL SUCCESS. Some issues detected.")
        print("\nTry:")
        print("  pip install -r requirements.txt")
        print("  python3 test_imports.py")
        print("\nIf issues persist, install packages individually.")
    
    print("=" * 60)

if __name__ == "__main__":
    install_dependencies()