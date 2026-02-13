#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERTICAL Stacked Bar Chart with Correct Grouping
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Paths
CSV_PATH = '/rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Model Answer/Task Performance/error_analysis_summary.csv'
SCRIPT_DIR = '/rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Script'
OUTPUT_DIR = '/rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Chart/Error Analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# Load data
df = pd.read_csv(CSV_PATH)
error_cols = [
    'Invalid Answer', 'Applicability Error (AE)', 'Mixed Errors (CSE+AE)',
    'Column Selection Error (CSE)', 'Mixed Errors (CSE+STC)', 'Statistical Task Confusion (STC)',
    'Mixed Errors (STC+AE)', 'Mixed Errors (CSE+STC+AE)'
]

# Define Groups
groups_order = ['llama2_13b', 'llama2_7b', 'llama3_8b', 'llama3_8b_instruct', 'deepseek']

def group_model(model):
    if 'llama2_13b' in model: return 'llama2_13b'
    if 'llama2_7b' in model: return 'llama2_7b'
    if 'llama3_8b_instruct' in model: return 'llama3_8b_instruct'  # Check first!
    if 'llama3_8b' in model: return 'llama3_8b'
    if 'llamadeepseek' in model: return 'llamadeepseek'
    return 'Other'

# Add Group column
df['Group'] = df['Model'].apply(group_model)

# Sort DataFrame by Group order
df['Group_Cat'] = pd.Categorical(df['Group'], categories=groups_order, ordered=True)
df_sorted = df.sort_values(['Group_Cat', 'Model']).reset_index(drop=True)

# Prepare plot data
df_errors = df_sorted[error_cols].fillna(0).copy()
df_errors['Model'] = df_sorted['Model']
df_errors['Group'] = df_sorted['Group']

# Plot
fig, ax = plt.subplots(figsize=(16, 10))
x_pos = np.arange(len(df_errors))
bottom = np.zeros(len(df_errors))

# Colors
colors = sns.color_palette("Set2", len(error_cols))
color_map = dict(zip(error_cols, colors))

# Stacked bars
for i, error_type in enumerate(error_cols):
    values = df_errors[error_type].values
    ax.bar(x_pos, values, bottom=bottom, label=error_type, 
           color=color_map[error_type], alpha=0.85, edgecolor='white', linewidth=1.2)
    bottom += values

# Group Separators (BETWEEN groups)
# Calculate boundaries: indices where group changes
group_boundaries = []
current_group = df_errors['Group'].iloc[0]
for i in range(len(df_errors) - 1):
    next_group = df_errors['Group'].iloc[i+1]
    if next_group != current_group:
        group_boundaries.append(i + 0.5)  # Line between bar i and i+1
        current_group = next_group

# Draw separators
for boundary in group_boundaries:
    ax.axvline(boundary, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)

# Group Labels (optional: centered text above groups)
for group in groups_order:
    models_in_group = df_errors[df_errors['Group'] == group]
    if not models_in_group.empty:
        group_indices = x_pos[df_errors['Group'] == group]
        center = np.mean(group_indices)
        ax.text(center, 1.02, group.replace('_', ' ').title(), 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333')

# Customize axes
ax.set_xticks(x_pos)
# Shorten model names for x-axis
short_labels = [m.replace('llama2_', '').replace('llama3_', '').replace('llamadeepseek', 'deepseek').replace('instruct_', 'inst_') for m in df_errors['Model']]
ax.set_xticklabels(short_labels, fontsize=10, rotation=45, ha='right')

ax.set_ylabel('Error Rate', fontsize=16, fontweight='bold')
ax.set_title('LLM Error Analysis in StatQA', fontsize=20, fontweight='bold', pad=40) # More pad for group labels
ax.set_ylim(0, 1.05)
ax.set_xlim(-0.5, len(df_errors) - 0.5)

# Legend
handles, labels = ax.get_legend_handles_labels()
clean_labels = [l.replace('Mixed Errors', 'Mixed').replace('(', '').replace(')', '') for l in labels]
ax.legend(handles=handles, labels=clean_labels, bbox_to_anchor=(1.01, 1), 
          loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)

# Value Labels (>10%)
for i in range(len(df_errors)):
    bottom_i = 0
    for error_type in error_cols:
        val = df_errors.iloc[i][error_type]
        if val > 0.10:
            ax.text(i, bottom_i + val/2, f'{val:.2f}', ha='center', va='center', 
                   fontweight='bold', fontsize=8, color='black')
        bottom_i += val

output_path = os.path.join(OUTPUT_DIR, 'User_Error_Analysis_Bar_Chart.png')
plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Chart saved: {output_path}")

# Summary table
print("\n📊 Group Averages:")
summary = df.groupby('Group')[error_cols].mean().round(3)
print(summary.to_string())
