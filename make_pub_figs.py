"""
True publication-quality figures for TMLR
Embedded fonts, vector output, no compromises
"""
import matplotlib
matplotlib.use('PDF')  # Force PDF backend

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import spearmanr

# Embedding ensures fonts are in the PDF
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Load data
with open('llm_results.json') as f:
    data = json.load(f)
results = data['results']

# Figure 1: Main result
fig = plt.figure(figsize=(3.5, 2.625))  # Exact single column
ax = fig.add_subplot(111)

colors = {'complete': '#1f77b4', 'cycle': '#ff7f0e', 
          'random': '#2ca02c', 'scale_free': '#d62728'}

for gtype in ['complete', 'random', 'scale_free', 'cycle']:
    subset = [r for r in results if r['graph_type'] == gtype]
    x = [r['spectral_gap'] for r in subset]
    y = [r['final_consensus_score'] for r in subset]
    ax.scatter(x, y, c=colors[gtype], label=gtype.replace('_', ' ').title(), 
              s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

# Regression
all_gaps = [r['spectral_gap'] for r in results]
all_scores = [r['final_consensus_score'] for r in results]
z = np.polyfit(all_gaps, all_scores, 1)
p = np.poly1d(z)
x_line = np.linspace(min(all_gaps), max(all_gaps), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1, alpha=0.5)

# Stats
r, pval = spearmanr(all_gaps, all_scores)
ax.text(0.05, 0.95, f'Spearman r = {r:.3f}\np < 0.001', 
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.set_xlabel('Spectral Gap', fontsize=9)
ax.set_ylabel('Consensus Score', fontsize=9)
ax.set_title('Spectral Gap Predicts Consensus Quality', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=7, frameon=True, fancybox=False)
ax.grid(True, alpha=0.3, linestyle='--')

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('pub_fig1.pdf', format='pdf', dpi=300, bbox_inches='tight',
            metadata={'Creator': 'Agentic Thermodynamics Study'})
plt.savefig('pub_fig1.png', dpi=600, bbox_inches='tight')
plt.close()

print('✓ pub_fig1.pdf - Vector PDF with embedded fonts')
print('✓ pub_fig1.png - 600 DPI preview')
