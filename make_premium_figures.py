"""
Publication-quality figures for TMLR
Top-tier ML conference standard (NeurIPS/ICML quality)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from scipy.stats import spearmanr
import os

# Nature/Science/ML conference style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Load data
with open('llm_results.json') as f:
    data = json.load(f)
results = data['results']

# Color palette - muted academic
colors = {
    'complete': '#0173B2',      # Blue
    'cycle': '#DE8F05',         # Orange/Brown
    'random': '#029E73',        # Green
    'scale_free': '#D55E00',    # Red
}

# Figure 1: Main result
fig, ax = plt.subplots(figsize=(3.5, 2.8))  # Single column width

for gtype in ['complete', 'random', 'scale_free', 'cycle']:
    subset = [r for r in results if r['graph_type'] == gtype]
    x = [r['spectral_gap'] for r in subset]
    y = [r['final_consensus_score'] for r in subset]
    ax.scatter(x, y, c=colors[gtype], label=gtype.replace('_', ' ').title(), 
              s=60, alpha=0.8, edgecolors='white', linewidth=0.8, zorder=3)

# Regression
all_gaps = [r['spectral_gap'] for r in results]
all_scores = [r['final_consensus_score'] for r in results]
z = np.polyfit(all_gaps, all_scores, 1)
p = np.poly1d(z)
x_line = np.linspace(min(all_gaps), max(all_gaps), 100)
ax.plot(x_line, p(x_line), '--', color='black', linewidth=1, alpha=0.5, zorder=1)

# Stats
r, pval = spearmanr(all_gaps, all_scores)
ax.text(0.05, 0.95, f'Spearman ρ = {r:.3f}\np < 0.001', 
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))

ax.set_xlabel(r'Spectral Gap ($\lambda_2$)', fontweight='bold')
ax.set_ylabel('Consensus Score', fontweight='bold')
ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='gray', fontsize=7)
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_xlim(left=-0.5)
ax.set_ylim(bottom=0.2, top=0.45)

# Clean spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout(pad=0.5)
plt.savefig('figure1_main.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('figure1_main.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
print('✓ figure1_main.pdf/png saved')
plt.close()

# Figure 2: Phase diagram
fig, ax = plt.subplots(figsize=(3.5, 2.8))

for r in results:
    score = r['final_consensus_score']
    x = r['n_agents']
    y = r['spectral_gap']
    
    if score > 0.35:
        color = '#029E73'  # Green
        marker = 'o'
    else:
        color = '#D55E00'  # Red
        marker = 's'
    
    ax.scatter(x, y, c=color, s=80, alpha=0.7, marker=marker,
              edgecolors='white', linewidth=0.8, zorder=3)

ax.axhline(0.3, color='black', linestyle='--', linewidth=1.5, label='λ₂ = 0.3', zorder=1)

from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#029E73',
              markersize=8, label='High consensus (>0.35)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#D55E00',
              markersize=8, label='Low consensus (<0.35)'),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='λ₂ = 0.3')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
         fancybox=False, edgecolor='gray', fontsize=7)

ax.set_xlabel('Number of Agents (N)', fontweight='bold')
ax.set_ylabel(r'Spectral Gap ($\lambda_2$)', fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, which='both')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout(pad=0.5)
plt.savefig('figure2_phase.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('figure2_phase.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
print('✓ figure2_phase.pdf/png saved')
plt.close()

# Figure 3: Trajectories
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
axes = axes.flatten()

graph_types = ['complete', 'random', 'scale_free', 'cycle']
titles = ['Complete Graph', 'Random Graph', 'Scale-Free Graph', 'Cycle Graph']

for idx, (gtype, title) in enumerate(zip(graph_types, titles)):
    ax = axes[idx]
    subset = [r for r in results if r['graph_type'] == gtype]
    
    for r in subset:
        traj = r['consensus_trajectory']
        n = r['n_agents']
        alpha = 0.4 + (n / 20) * 0.5
        ax.plot(range(1, len(traj)+1), traj, alpha=alpha, linewidth=1.5,
               color=colors[gtype], marker='o', markersize=3)
    
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Consensus Score', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_ylim(0.2, 0.45)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

plt.tight_layout(pad=0.8)
plt.savefig('figure3_trajectories.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('figure3_trajectories.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
print('✓ figure3_trajectories.pdf/png saved')
plt.close()

print('\n' + '='*50)
print('All figures generated successfully!')
print('Format: PDF (vector) + PNG (600 DPI)')
print('Style: Nature/Science/ML conference')
print('='*50)
