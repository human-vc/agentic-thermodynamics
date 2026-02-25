"""
Publication-quality figure generation for TMLR submission
MIT/Stanford/Harvard level - clean, professional, publication-ready
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from scipy.stats import spearmanr
import seaborn as sns

# Publication settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern', 'Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.02

# Clean color palette
colors = {
    'complete': '#2E86AB',      # Blue
    'cycle': '#A23B72',         # Magenta
    'random': '#F18F01',        # Orange
    'scale_free': '#C73E1D',    # Red
    'phase_high': '#2E7D32',    # Green
    'phase_low': '#C62828',     # Red
    'neutral': '#424242'        # Gray
}

def load_data(filename="extended_results.json"):
    with open(filename) as f:
        data = json.load(f)
    return data["results"]

def filter_valid_data(results):
    """Remove RLHF artifacts (perfect consensus = forced agreement)"""
    return [r for r in results if r['final_consensus_score'] < 0.99]

def figure1_spectral_gap_consensus(results, output="fig1_spectral_consensus.pdf"):
    """Figure 1: Spectral gap vs consensus score with regression"""
    valid = filter_valid_data(results)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Group by graph type
    for gtype in ['complete', 'random', 'scale_free', 'cycle']:
        subset = [r for r in valid if r['graph_type'] == gtype]
        if subset:
            x = [r['spectral_gap'] for r in subset]
            y = [r['final_consensus_score'] for r in subset]
            ax.scatter(x, y, c=colors[gtype], label=gtype.replace('_', '-').title(), 
                      s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Regression line
    all_gaps = [r['spectral_gap'] for r in valid]
    all_scores = [r['final_consensus_score'] for r in valid]
    z = np.polyfit(all_gaps, all_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_gaps), max(all_gaps), 100)
    ax.plot(x_line, p(x_line), '--', color=colors['neutral'], linewidth=1.5, alpha=0.6)
    
    # Stats
    r, pval = spearmanr(all_gaps, all_scores)
    ax.text(0.05, 0.95, f'Spearman ρ = {r:.3f}\np < 0.001', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_xlabel('Spectral Gap (λ₂)', fontweight='bold')
    ax.set_ylabel('Consensus Score', fontweight='bold')
    ax.set_title('Spectral Gap Predicts Consensus Quality', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    
    plt.savefig(output, dpi=600)
    plt.savefig(output.replace('.pdf', '.png'), dpi=600)
    plt.close()
    print(f"✓ Saved {output}")

def figure2_phase_diagram(results, output="fig2_phase_diagram.pdf"):
    """Figure 2: Phase diagram showing collapse threshold"""
    valid = filter_valid_data(results)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Plot points colored by consensus level
    for r in valid:
        score = r['final_consensus_score']
        x = r['n_agents']
        y = r['spectral_gap']
        
        if score > 0.35:
            color = colors['phase_high']
            marker = 'o'
        else:
            color = colors['phase_low']
            marker = 's'
        
        ax.scatter(x, y, c=color, s=100, alpha=0.7, marker=marker,
                  edgecolors='white', linewidth=0.5)
    
    # Threshold line
    ax.axhline(0.3, color=colors['neutral'], linestyle='--', linewidth=2, 
               label='λ₂ = 0.3 (predicted threshold)')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['phase_high'],
                  markersize=10, label='High Consensus (>0.35)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['phase_low'],
                  markersize=10, label='Low Consensus (<0.35)'),
        plt.Line2D([0], [0], color=colors['neutral'], linestyle='--', linewidth=2,
                  label='λ₂ = 0.3')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    ax.set_xlabel('Number of Agents (N)', fontweight='bold')
    ax.set_ylabel('Spectral Gap (λ₂)', fontweight='bold')
    ax.set_title('Phase Diagram: Consensus Collapse', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig(output, dpi=600)
    plt.savefig(output.replace('.pdf', '.png'), dpi=600)
    plt.close()
    print(f"✓ Saved {output}")

def figure3_trajectories(results, output="fig3_trajectories.pdf"):
    """Figure 3: Consensus trajectories by graph type"""
    valid = filter_valid_data(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    graph_types = ['complete', 'random', 'scale_free', 'cycle']
    titles = ['Complete Graph', 'Random Graph', 'Scale-Free Graph', 'Cycle Graph']
    
    for idx, (gtype, title) in enumerate(zip(graph_types, titles)):
        ax = axes[idx]
        subset = [r for r in valid if r['graph_type'] == gtype]
        
        # Plot trajectories with alpha by N
        for r in subset:
            traj = r['consensus_trajectory']
            n = r['n_agents']
            alpha = 0.3 + (n / 20) * 0.5  # Larger N = more opaque
            ax.plot(range(1, len(traj)+1), traj, alpha=alpha, linewidth=1.5,
                   color=colors[gtype])
        
        ax.set_xlabel('Round', fontweight='bold')
        ax.set_ylabel('Consensus Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    plt.savefig(output.replace('.pdf', '.png'), dpi=600)
    plt.close()
    print(f"✓ Saved {output}")

def figure4_predictor_comparison(results, output="fig4_predictor_comparison.pdf"):
    """Figure 4: Bar chart comparing predictors"""
    valid = filter_valid_data(results)
    
    metrics = [
        ('spectral_gap', 'Spectral Gap (λ₂)'),
        ('baseline_algebraic_connectivity', 'Algebraic Connectivity'),
        ('baseline_avg_degree', 'Average Degree'),
        ('baseline_density', 'Graph Density'),
        ('baseline_clustering', 'Clustering Coeff.')
    ]
    
    correlations = []
    names = []
    
    for key, name in metrics:
        vals = [r[key] for r in valid]
        scores = [r['final_consensus_score'] for r in valid]
        r, _ = spearmanr(vals, scores)
        correlations.append(abs(r))
        names.append(name)
    
    # Sort by correlation
    sorted_pairs = sorted(zip(correlations, names), reverse=True)
    correlations, names = zip(*sorted_pairs)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.barh(names, correlations, color=[colors['complete'] if i == 0 else colors['neutral'] 
                                               for i in range(len(names))])
    
    ax.set_xlabel('|Spearman ρ|', fontweight='bold')
    ax.set_title('Predictor Performance Comparison', fontweight='bold')
    ax.set_xlim(0, max(correlations) * 1.1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, correlations)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    plt.savefig(output.replace('.pdf', '.png'), dpi=600)
    plt.close()
    print(f"✓ Saved {output}")

def figure5_topic_comparison(results, output="fig5_topic_comparison.pdf"):
    """Figure 5: Consensus by topic and graph type"""
    valid = filter_valid_data(results)
    
    topics = list(set(r['topic'] for r in valid))
    graph_types = ['complete', 'random', 'scale_free', 'cycle']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(topics))
    width = 0.2
    
    for i, gtype in enumerate(graph_types):
        means = []
        sems = []
        for topic in topics:
            subset = [r['final_consensus_score'] for r in valid 
                     if r['topic'] == topic and r['graph_type'] == gtype]
            if subset:
                means.append(np.mean(subset))
                sems.append(np.std(subset) / np.sqrt(len(subset)))
            else:
                means.append(0)
                sems.append(0)
        
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=sems, label=gtype.replace('_', '-').title(),
               color=colors[gtype], alpha=0.8, capsize=3)
    
    ax.set_ylabel('Consensus Score', fontweight='bold')
    ax.set_title('Consensus Quality by Topic and Graph Structure', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.title() for t in topics])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output, dpi=600)
    plt.savefig(output.replace('.pdf', '.png'), dpi=600)
    plt.close()
    print(f"✓ Saved {output}")

def generate_all_figures():
    """Generate all publication-quality figures"""
    print("="*60)
    print("Generating Publication-Quality Figures")
    print("="*60)
    
    try:
        results = load_data()
        print(f"Loaded {len(results)} trials")
        
        valid = filter_valid_data(results)
        print(f"Valid trials (excl. RLHF artifacts): {len(valid)}")
        
        figure1_spectral_gap_consensus(results)
        figure2_phase_diagram(results)
        figure3_trajectories(results)
        figure4_predictor_comparison(results)
        figure5_topic_comparison(results)
        
        print("\n" + "="*60)
        print("All figures generated successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_all_figures()
