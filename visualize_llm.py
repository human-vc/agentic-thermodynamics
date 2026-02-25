import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import rcParams

rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13

def load_results(filename="llm_results.json"):
    with open(filename) as f:
        data = json.load(f)
    return data["results"], data["analysis"]

def plot_spectral_gap_vs_consensus(results, output="figure1_spectral_consensus.pdf"):
    spectral_gaps = [r["spectral_gap"] for r in results]
    consensus_scores = [r["final_consensus_score"] for r in results]
    graph_types = [r["graph_type"] for r in results]
    n_agents = [r["n_agents"] for r in results]
    
    colors_map = {"complete": "#2ecc71", "cycle": "#e74c3c", "random": "#3498db", "scale_free": "#9b59b6"}
    colors = [colors_map.get(g, "#95a5a6") for g in graph_types]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for gtype in set(graph_types):
        mask = [gt == gtype for gt in graph_types]
        x = [spectral_gaps[i] for i, m in enumerate(mask) if m]
        y = [consensus_scores[i] for i, m in enumerate(mask) if m]
        ax.scatter(x, y, c=colors_map.get(gtype, "#95a5a6"), label=gtype.capitalize(), s=100, alpha=0.7)
    
    z = np.polyfit(spectral_gaps, consensus_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(spectral_gaps), max(spectral_gaps), 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.5, label=f'Linear fit (slope={z[0]:.4f})')
    
    ax.set_xlabel('Spectral Gap (λ₂)')
    ax.set_ylabel('Final Consensus Score')
    ax.set_title('Spectral Gap Predicts Consensus Quality in LLM Swarms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.savefig(output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {output}")

def plot_trajectories_by_graph_type(results, output="figure2_trajectories.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    graph_types = list(set(r["graph_type"] for r in results))
    
    for idx, gtype in enumerate(graph_types):
        ax = axes[idx]
        subset = [r for r in results if r["graph_type"] == gtype]
        
        for r in subset:
            traj = r["consensus_trajectory"]
            n = r["n_agents"]
            ax.plot(range(1, len(traj)+1), traj, alpha=0.7, linewidth=2, label=f'N={n}')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Consensus Score')
        ax.set_title(f'{gtype.capitalize()} Graph')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.savefig(output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {output}")

def plot_agent_scaling(results, output="figure3_agent_scaling.pdf"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    graph_types = list(set(r["graph_type"] for r in results))
    colors_map = {"complete": "#2ecc71", "cycle": "#e74c3c", "random": "#3498db", "scale_free": "#9b59b6"}
    
    for gtype in graph_types:
        subset = [r for r in results if r["graph_type"] == gtype]
        n_vals = sorted(set(r["n_agents"] for r in subset))
        avg_scores = []
        avg_gaps = []
        
        for n in n_vals:
            n_subset = [r for r in subset if r["n_agents"] == n]
            avg_scores.append(np.mean([r["final_consensus_score"] for r in n_subset]))
            avg_gaps.append(np.mean([r["spectral_gap"] for r in n_subset]))
        
        ax.plot(n_vals, avg_scores, 'o-', color=colors_map.get(gtype, "#95a5a6"), 
                linewidth=2, markersize=8, label=f'{gtype.capitalize()} (consensus)')
    
    ax.set_xlabel('Number of Agents (N)')
    ax.set_ylabel('Final Consensus Score')
    ax.set_title('Consensus Quality vs. Swarm Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.savefig(output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {output}")

def plot_phase_diagram(results, output="figure4_phase_diagram.pdf"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for r in results:
        x = r["n_agents"]
        y = r["spectral_gap"]
        color = "#2ecc71" if r["final_consensus_score"] > 0.35 else "#e74c3c"
        ax.scatter(x, y, c=color, s=200, alpha=0.7, edgecolors='black')
    
    ax.axhline(0.3, color='black', linestyle='--', linewidth=2, label='Predicted Threshold (λ₂ ≈ 0.3)')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Higher Consensus (>0.35)'),
                       Patch(facecolor='#e74c3c', edgecolor='black', label='Lower Consensus (<0.35)'),
                       plt.Line2D([0], [0], color='black', linestyle='--', label='Threshold')]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('Number of Agents (N)')
    ax.set_ylabel('Spectral Gap (λ₂)')
    ax.set_title('Phase Diagram: Consensus Collapse')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.savefig(output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {output}")

if __name__ == "__main__":
    results, analysis = load_results()
    
    plot_spectral_gap_vs_consensus(results)
    plot_trajectories_by_graph_type(results)
    plot_agent_scaling(results)
    plot_phase_diagram(results)
    
    print("\nAll figures generated successfully.")
    print(f"Analysis: {analysis['predictive_accuracy']*100:.1f}% accuracy, Spearman r={analysis['spearman_r']:.3f}")
