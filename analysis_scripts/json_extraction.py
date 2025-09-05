# json_extraction.py - add print statement
import json

with open('robustness_results.json', 'r') as f:
    data = json.load(f)

summary = {}
for model, results in data.items():
    summary[model] = {
        'H1_kl': results['H1_counterfactual']['mean_kl'],
        'H1_supported': results['H1_counterfactual']['supported'],
        'H2_effect': results['H2_feedback']['effect_size'],
        'H2_confirmed': results['H2_feedback']['efficiency_confirmed']
    }

# ADD THIS TO SEE OUTPUT
print(json.dumps(summary, indent=2))

with open('summary_metrics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to summary_metrics.json")