import json
from collections import defaultdict
import numpy as np

# Load delivery.jsonl
tasks = []
with open('delivery.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(tasks)} tasks from delivery.jsonl")

# Collect all unique model names to identify Claude
model_names_set = set()
for task in tasks:
    for output in task.get('model_outputs', []):
        model_names_set.add(output.get('model_name'))

print(f"\nUnique model names found: {sorted(model_names_set)}")

# Calculate pass rates for each model (using logic from generate_full_dashboard.py)
model_weighted_grades = defaultdict(list)
model_unweighted_grades = defaultdict(list)
model_task_count = defaultdict(int)

for task in tasks:
    task_id = task.get('task_id')
    rubrics = task.get('rubrics', [])
    
    if not rubrics:
        continue
    
    # Exclude critical criteria from weight calculations
    non_critical_rubrics = [r for r in rubrics if r.get('critical_classification') != 'critical criteria']
    
    total_points = sum(r.get('max_points', 0) for r in non_critical_rubrics)
    rubric_lookup = {r['id']: r for r in rubrics}
    
    for output in task.get('model_outputs', []):
        model = output.get('model_name')
        grades = output.get('rubric_grades', [])
        
        if not grades:
            continue
        
        grade_map = {g['criteria_id']: g for g in grades}
        
        # Calculate weighted grade (points earned / total points) - excluding critical criteria
        earned = sum(rubric_lookup[rid].get('max_points', 0) for rid, g in grade_map.items() 
                    if g.get('is_satisfied') == 'True' 
                    and rid in rubric_lookup 
                    and rubric_lookup[rid].get('critical_classification') != 'critical criteria')
        
        # Calculate unweighted grade (rubrics passed / total rubrics) - excluding critical criteria
        passed = sum(1 for g in grades 
                    if g.get('is_satisfied') == 'True' 
                    and g.get('criteria_id') in rubric_lookup
                    and rubric_lookup[g.get('criteria_id')].get('critical_classification') != 'critical criteria')
        
        if total_points > 0 and len(non_critical_rubrics) > 0:
            model_weighted_grades[model].append(earned / total_points * 100)
            model_unweighted_grades[model].append(passed / len(non_critical_rubrics) * 100)
            model_task_count[model] += 1

# Display results for all models
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

for model in sorted(model_weighted_grades.keys()):
    weighted = model_weighted_grades[model]
    unweighted = model_unweighted_grades[model]
    
    if weighted:
        print(f"\n{model}:")
        print(f"  Tasks evaluated: {model_task_count[model]}")
        print(f"  Weighted pass rate (points-based):")
        print(f"    Mean: {np.mean(weighted):.2f}%")
        print(f"    Median: {np.median(weighted):.2f}%")
        print(f"    Std Dev: {np.std(weighted):.2f}%")
        print(f"  Unweighted pass rate (rubrics-based):")
        print(f"    Mean: {np.mean(unweighted):.2f}%")
        print(f"    Median: {np.median(unweighted):.2f}%")

# Focus on Claude specifically
print("\n" + "="*80)
print("CLAUDE-SPECIFIC RESULTS")
print("="*80)

claude_models = [m for m in model_names_set if 'claude' in m.lower() or 'Claude' in m]
if claude_models:
    for claude_model in sorted(claude_models):
        if claude_model in model_weighted_grades:
            weighted = model_weighted_grades[claude_model]
            print(f"\n{claude_model} Pass Rate:")
            print(f"  {np.mean(weighted):.2f}% (mean weighted)")
            print(f"  {np.median(weighted):.2f}% (median weighted)")
else:
    print("No Claude models found in the data.")
    print(f"Available models: {sorted(model_names_set)}")
