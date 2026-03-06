import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import csv

# EXACT style from visualize_data.py
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#c9d1d9'
plt.rcParams['ytick.color'] = '#c9d1d9'
plt.rcParams['text.color'] = '#c9d1d9'

# EXACT colors from visualize_data.py
COLORS = {
    'purple': '#8B5CF6',
    'lavender': '#A78BFA',
    'orange': '#F97316',
    'blue': '#3B82F6',
    'cyan': '#06B6D4',
    'green': '#10B981',
    'red': '#EF4444',
    'pink': '#EC4899',
    'yellow': '#FBBF24',
}

# Load OCR classification data
ocr_classification_data = {}
try:
    with open('ocr_classification_delivery_sender.json', 'r') as f:
        ocr_classification_data = json.load(f)
except FileNotFoundError:
    print("Warning: ocr_classification_delivery_sender.json not found")

# Process OCR disagreement data
ocr_summary = ocr_classification_data.get('summary', {})
ocr_rubrics = ocr_classification_data.get('ocr_rubrics', [])
non_ocr_rubrics = ocr_classification_data.get('non_ocr_rubrics', [])

# Count disagreement types
ocr_disagreements = Counter()
non_ocr_disagreements = Counter()

for r in ocr_rubrics:
    for dtype in r.get('disagreement_types', []):
        ocr_disagreements[dtype] += 1

for r in non_ocr_rubrics:
    for dtype in r.get('disagreement_types', []):
        non_ocr_disagreements[dtype] += 1

# Load grader results for agreement matrix
grader_data = defaultdict(lambda: defaultdict(dict))
with open('grader_results_18of20.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            task_id = d.get('task_id')
            model_name = d.get('model_name')
            grader = d.get('grader')
            comparisons = d.get('comparisons', [])
            grader_data[task_id][model_name][grader] = comparisons
        except json.JSONDecodeError:
            continue

# Calculate agreement pairs
human_gpt5_pairs = []
human_gemini_pairs = []
gpt5_gemini_pairs = []

for task_id, models in grader_data.items():
    for model_name, grader_results in models.items():
        gpt52_comps = grader_results.get('gpt52', [])
        gemini_comps = grader_results.get('gemini3pro', grader_results.get('gemini3flash', []))
        
        for comp in gpt52_comps:
            human_gpt5_pairs.append((comp.get('existing'), comp.get('grader')))
        
        for comp in gemini_comps:
            human_gemini_pairs.append((comp.get('existing'), comp.get('grader')))
        
        if gpt52_comps and gemini_comps:
            gpt52_by_id = {c['rubric_id']: c['grader'] for c in gpt52_comps}
            gemini_by_id = {c['rubric_id']: c['grader'] for c in gemini_comps}
            common_ids = set(gpt52_by_id.keys()) & set(gemini_by_id.keys())
            for rid in common_ids:
                gpt5_gemini_pairs.append((gpt52_by_id[rid], gemini_by_id[rid]))

def calc_agreement(pairs):
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not valid:
        return 0, 0
    agreed = sum(1 for a, b in valid if a == b)
    return (agreed / len(valid)) * 100, len(valid)

human_gpt5, _ = calc_agreement(human_gpt5_pairs)
human_gemini, _ = calc_agreement(human_gemini_pairs)
gpt5_gemini, _ = calc_agreement(gpt5_gemini_pairs)

# Load main task data
with open('delivery.jsonl', 'r') as f:
    tasks = [json.loads(line) for line in f]

# Load mime data
mime_data = {}
with open('DAMM_Tasks_2026_03_02.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        task_id = row.get('TASK_ID', '').strip()
        if task_id:
            input_mimes = row.get('ATTACHMENTS_S3_MIME_TYPE', '').strip()
            output_mimes = row.get('GOLDEN_MIME_TYPE', '').strip()
            input_list = [m.strip() for m in input_mimes.split('\n') if m.strip()]
            
            # Parse GOLDEN_MIME_TYPES as JSON if it looks like JSON, otherwise split by newlines
            output_list = []
            if output_mimes:
                try:
                    # Try parsing as JSON array
                    output_list = json.loads(output_mimes)
                except json.JSONDecodeError:
                    # Fall back to newline-separated
                    output_list = [m.strip() for m in output_mimes.split('\n') if m.strip()]
            
            mime_data[task_id] = {'input': input_list, 'output': output_list}

def mime_to_short(mime):
    mime_map = {
        'text/csv': 'CSV', 'application/json': 'JSON', 'application/pdf': 'PDF',
        'application/x-zip-compressed': 'X-ZIP-COMP', 'application/zip': 'ZIP',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
        'image/png': 'PNG', 'image/jpeg': 'JPEG',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX',
    }
    return mime_map.get(mime, mime.split('/')[-1].upper()[:10] if mime else 'Unknown')

# Extract stats
domains = Counter()
input_types = Counter()
output_types = Counter()
input_files_per_task = []
output_files_per_task = []
rubrics_per_task = []
critical_per_task = []
model_grades = defaultdict(list)
model_unweighted = defaultdict(list)

for task in tasks:
    task_id = task.get('task_id')
    domains[task.get('metadata', {}).get('domain', 'Unknown')] += 1
    
    if task_id in mime_data:
        input_list = mime_data[task_id]['input']
        input_files_per_task.append(len(input_list))
        # Count all input file types (no filtering)
        for mime in input_list:
            input_types[mime_to_short(mime)] += 1
        
        output_list = mime_data[task_id]['output']
        # Fallback: CSV GOLDEN_MIME_TYPE is often empty; use golden_solution attachments from task
        if not output_list:
            golden_urls = task.get('golden_solution', {}).get('solution_attachments', [])
            output_files_per_task.append(len(golden_urls))
            for _ in golden_urls:
                output_types['Unknown'] += 1
        else:
            output_files_per_task.append(len(output_list))
            for mime in output_list:
                output_types[mime_to_short(mime)] += 1
    
    rubrics = task.get('rubrics', [])
    rubrics_per_task.append(len(rubrics))
    critical_per_task.append(sum(1 for r in rubrics if r.get('critical_classification') == 'critical criteria'))
    
    # Exclude critical criteria from scoring
    non_critical_rubrics = [r for r in rubrics if r.get('critical_classification') != 'critical criteria']
    total_points = sum(r.get('max_points', 0) for r in non_critical_rubrics)
    rubric_lookup = {r['id']: r for r in rubrics}
    
    for output in task.get('model_outputs', []):
        model = output.get('model_name')
        # Skip GPT5 from grade calculations
        if model == 'GPT5':
            continue
        grades = output.get('rubric_grades', [])
        grade_map = {g['criteria_id']: g for g in grades}
        
        earned = sum(rubric_lookup[rid].get('max_points', 0) for rid, g in grade_map.items() 
                    if g.get('is_satisfied') == 'True' and rid in rubric_lookup
                    and rubric_lookup[rid].get('critical_classification') != 'critical criteria')
        passed = sum(1 for g in grades if g.get('is_satisfied') == 'True')
        
        if total_points > 0:
            model_grades[model].append(earned / total_points * 100)
            model_unweighted[model].append(passed / len(rubrics) * 100)

total_tasks = len(tasks)

# Create figure - EXACT same as visualize_data.py
fig = plt.figure(figsize=(24, 14))
fig.suptitle(f'{total_tasks} Task Delivery (3/02/2026)', fontsize=18, fontweight='bold', color='white', y=0.98)
gs = fig.add_gridspec(2, 8, hspace=0.4, wspace=0.35, top=0.92, bottom=0.08, left=0.04, right=0.98)

# ===== TOP ROW =====

# 1. Domain Distribution (lavender bars like original)
ax1 = fig.add_subplot(gs[0, 0:2])
domain_items = domains.most_common(10)
domain_names = [d[0][:20] + '...' if len(d[0]) > 20 else d[0] for d in domain_items]
domain_pcts = [d[1]/total_tasks*100 for d in domain_items]
y_pos = range(len(domain_names))
bars = ax1.barh(y_pos, domain_pcts, color=COLORS['lavender'], edgecolor='white', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(domain_names, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel('% of Tasks', fontsize=9)
for bar, pct in zip(bars, domain_pcts):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center', fontsize=7)
ax1.set_title('Domain Distribution', fontsize=11, fontweight='bold', pad=5)

# 2. Input File Type Distribution (orange top, blue rest)
ax2 = fig.add_subplot(gs[0, 2:4])
input_items = input_types.most_common(10)
type_names = [t[0] for t in input_items]
type_pcts = [t[1]/total_tasks*100 for t in input_items]
y_pos = range(len(type_names))
colors_list = [COLORS['orange'] if i == 0 else COLORS['blue'] for i in range(len(type_names))]
bars = ax2.barh(y_pos, type_pcts, color=colors_list, edgecolor='white', linewidth=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(type_names, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('% of Tasks', fontsize=9)
for bar, pct in zip(bars, type_pcts):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center', fontsize=7)
ax2.set_title('Input File Type Distribution', fontsize=11, fontweight='bold', pad=5)

# 3. Output File Type Distribution (orange bars)
ax3 = fig.add_subplot(gs[0, 4:6])
output_items = output_types.most_common(10)
type_names = [t[0] for t in output_items]
type_pcts = [t[1]/total_tasks*100 for t in output_items]
y_pos = range(len(type_names))
bars = ax3.barh(y_pos, type_pcts, color=COLORS['orange'], edgecolor='white', linewidth=0.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(type_names, fontsize=9)
ax3.invert_yaxis()
ax3.set_xlabel('% of Tasks', fontsize=9)
for bar, pct in zip(bars, type_pcts):
    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center', fontsize=7)
ax3.set_title('Output File Type Distribution', fontsize=11, fontweight='bold', pad=5)

# 4. Input Files per Task (orange bars)
ax4 = fig.add_subplot(gs[0, 6])
if input_files_per_task:
    file_counts = Counter(input_files_per_task)
    x_vals = sorted(file_counts.keys())
    y_vals = [file_counts[x]/total_tasks*100 for x in x_vals]
    bars = ax4.bar(x_vals, y_vals, color=COLORS['orange'], edgecolor='white', linewidth=0.5)
    ax4.set_xticks(x_vals)
    for bar, pct in zip(bars, y_vals):
        if pct > 3:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.0f}%', ha='center', fontsize=8)
ax4.set_xlabel('Number of Input Files', fontsize=9)
ax4.set_ylabel('% of Tasks', fontsize=9)
avg_input = np.mean(input_files_per_task) if input_files_per_task else 0
print(f"DEBUG: Input files - count: {len(input_files_per_task)}, avg: {avg_input:.2f}, values: {input_files_per_task[:5]}")
ax4.set_title(f'Input Files per Task\n(avg: {avg_input:.2f})', fontsize=11, fontweight='bold', pad=5)

# 5. Output Files per Task (orange bars)
ax5 = fig.add_subplot(gs[0, 7])
if output_files_per_task:
    file_counts = Counter(output_files_per_task)
    x_vals = sorted(file_counts.keys())
    y_vals = [file_counts[x]/total_tasks*100 for x in x_vals]
    bars = ax5.bar(x_vals, y_vals, color=COLORS['orange'], edgecolor='white', linewidth=0.5)
    ax5.set_xticks(x_vals)
    for bar, pct in zip(bars, y_vals):
        if pct > 3:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.0f}%', ha='center', fontsize=8)
ax5.set_xlabel('Number of Output Files', fontsize=9)
ax5.set_ylabel('% of Tasks', fontsize=9)
avg_output = np.mean(output_files_per_task) if output_files_per_task else 0
print(f"DEBUG: Output files - count: {len(output_files_per_task)}, avg: {avg_output:.2f}, values: {output_files_per_task[:5]}")
ax5.set_title(f'Output Files per Task\n(avg: {avg_output:.2f})', fontsize=11, fontweight='bold', pad=5)

# ===== BOTTOM ROW =====

# 6. Total Rubrics per Task (blue histogram)
ax6 = fig.add_subplot(gs[1, 0:2])
if rubrics_per_task:
    min_r, max_r = min(rubrics_per_task), max(rubrics_per_task)
    bins = range(min_r, max_r + 2)
    ax6.hist(rubrics_per_task, bins=bins, color=COLORS['blue'], edgecolor='white', linewidth=0.5, alpha=0.9, align='left')
ax6.set_xlabel('Number of Rubrics', fontsize=9)
ax6.set_ylabel('Number of Tasks', fontsize=9)
ax6.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
mean_val = np.mean(rubrics_per_task) if rubrics_per_task else 0
ax6.set_title(f'Total Rubrics per Task\n(avg: {mean_val:.1f})', fontsize=11, fontweight='bold', pad=5)

# 7. Critical Rubrics per Task (red histogram)
ax7 = fig.add_subplot(gs[1, 2:4])
if critical_per_task:
    min_c, max_c = min(critical_per_task), max(critical_per_task)
    bins = range(min_c, max_c + 2)
    ax7.hist(critical_per_task, bins=bins, color=COLORS['red'], edgecolor='white', linewidth=0.5, alpha=0.9, align='left')
ax7.set_xlabel('Number of Critical Rubrics', fontsize=9)
ax7.set_ylabel('Number of Tasks', fontsize=9)
ax7.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
mean_val = np.mean(critical_per_task) if critical_per_task else 0
ax7.set_title(f'Critical Rubrics per Task\n(avg: {mean_val:.1f})', fontsize=11, fontweight='bold', pad=5)

# 8. Weighted Grade (blue/orange box plots)
ax8 = fig.add_subplot(gs[1, 4:6])
model_names = list(model_grades.keys())
if model_names and any(model_grades.values()):
    box_data = [model_grades[m] for m in model_names]
    colors_list = [COLORS['blue'], COLORS['orange']][:len(model_names)]
    bp = ax8.boxplot(box_data, tick_labels=model_names, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    ax8.set_ylabel('Score (%)', fontsize=9)
    ax8.set_ylim(0, 110)
    for i, (data, color) in enumerate(zip(box_data, colors_list)):
        if data:
            mean = np.mean(data)
            median = np.median(data)
            # Place label inside the box at the median position
            ax8.text(i+1, median, f'μ={mean:.1f}%', ha='center', va='center', 
                    fontsize=10, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9, edgecolor='none'))
ax8.set_title('Weighted Grade\n(points earned / total points)', fontsize=10, fontweight='bold', pad=5)

# 9. Grader Agreement Matrix (green heatmap)
ax9 = fig.add_subplot(gs[1, 6:8])
graders = ['GPT', 'Gemini Pro', 'Human']
matrix = np.array([
    [100.0, gpt5_gemini, human_gpt5],
    [gpt5_gemini, 100.0, human_gemini],
    [human_gpt5, human_gemini, 100.0]
])
im = ax9.imshow(matrix, cmap='Greens', vmin=50, vmax=100)
ax9.set_xticks(range(len(graders)))
ax9.set_yticks(range(len(graders)))
ax9.set_xticklabels(graders, fontsize=9, rotation=45, ha='right')
ax9.set_yticklabels(graders, fontsize=9)
for i in range(len(graders)):
    for j in range(len(graders)):
        val = matrix[i, j]
        text = f'{val:.0f}%'
        color = 'white' if val < 85 else 'black'
        ax9.text(j, i, text, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
ax9.set_title('Grader Agreement Matrix (%)', fontsize=11, fontweight='bold', pad=5)

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'dashboard_{timestamp}.png'
plt.savefig(filename, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
print(f"✅ Saved {filename} with EXACT original theme")
