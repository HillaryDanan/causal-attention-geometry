#!/usr/bin/env python3
"""
Run each hypothesis test individually with detailed output.
This helps isolate issues and see exactly what's happening.
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_single_test(test_name, script_path, args=None):
    """Run a single test and capture output."""
    print("\n" + "=" * 60)
    print(f"Running: {test_name}")
    print("=" * 60)
    
    cmd = ["python3", str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"\n⚠️ Test exited with code: {result.returncode}")
        else:
            print(f"\n✓ Test completed successfully")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"\n✗ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n✗ Error running test: {e}")
        return False

def check_results():
    """Check what result files were created."""
    results_dir = Path("results")
    if not results_dir.exists():
        print("\n⚠️ No results directory found")
        return {}
    
    print("\n" + "-" * 60)
    print("Checking result files:")
    print("-" * 60)
    
    results = {}
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                print(f"✓ {json_file.name}: Valid JSON")
                
                # Extract key metrics
                if "hypothesis" in data:
                    hyp_name = data["hypothesis"]
                    results[hyp_name] = {
                        "file": json_file.name,
                        "supported": data.get("hypothesis_supported", False)
                    }
                    
                    # Add specific metrics
                    if hyp_name == "counterfactual_divergence":
                        results[hyp_name]["kl_divergence"] = data.get("mean_kl_divergence", 0)
                        results[hyp_name]["p_value"] = data.get("p_value", 1)
                    elif hyp_name == "feedback_loop_density":
                        results[hyp_name]["effect_size"] = data.get("effect_sizes", {}).get("circular_vs_linear", 0)
                    elif hyp_name == "layer_specificity":
                        results[hyp_name]["middle_effect"] = data.get("middle_layers_mean_effect", 0)
                        
        except json.JSONDecodeError:
            print(f"✗ {json_file.name}: Invalid JSON")
        except Exception as e:
            print(f"✗ {json_file.name}: Error reading: {e}")
    
    return results

def main():
    print("=" * 70)
    print("INDIVIDUAL HYPOTHESIS TESTING")
    print("Running each test separately with detailed output")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check environment
    print("\nEnvironment check:")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {Path.cwd()}")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Define tests
    tests = [
        {
            "name": "Hypothesis 1: Counterfactual Divergence",
            "script": "experiments/test_counterfactual.py",
            "args": ["--n-samples=20"]  # Start with smaller N for debugging
        },
        {
            "name": "Hypothesis 2: Feedback Loop Density",
            "script": "experiments/test_feedback_loop.py",
            "args": ["--n-samples=20"]
        },
        {
            "name": "Hypothesis 3: Layer Specificity",
            "script": "experiments/test_layer_specificity.py",
            "args": ["--n-samples=20"]
        }
    ]
    
    # Run tests
    results = []
    for test in tests:
        success = run_single_test(
            test["name"],
            test["script"],
            test.get("args")
        )
        results.append((test["name"], success))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    # Check actual results
    result_data = check_results()
    
    if result_data:
        print("\n" + "-" * 60)
        print("Result Analysis:")
        print("-" * 60)
        
        for hyp_name, data in result_data.items():
            print(f"\n{hyp_name}:")
            print(f"  File: {data['file']}")
            print(f"  Supported: {data['supported']}")
            
            if "kl_divergence" in data:
                print(f"  KL Divergence: {data['kl_divergence']:.4f}")
                print(f"  p-value: {data['p_value']:.4f}")
            elif "effect_size" in data:
                print(f"  Effect Size: {data['effect_size']:.4f}")
            elif "middle_effect" in data:
                print(f"  Middle Layer Effect: {data['middle_effect']:.4f}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    failed_tests = [name for name, success in results if not success]
    if failed_tests:
        print("Failed tests need debugging:")
        for test in failed_tests:
            print(f"  - {test}")
        print("\nRun debug scripts:")
        print("  python3 debug_h1_counterfactual.py")
        print("  python3 debug_h2_feedback.py")
        print("  python3 debug_h3_layers.py")
    else:
        print("All tests ran successfully!")
        print("\nTo run with more samples:")
        print("  python3 experiments/run_all.py --n-samples=64")
    
    print("=" * 70)

if __name__ == "__main__":
    main()