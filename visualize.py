import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import rcParams

rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

def load_results(filename="results.json"):
    with open(filename) as f:
        data = json.load(f)
    return data["results"], data["analysis"]

def plot_spectral_gap_vs_convergence(results, analysis, output="figure1_spectral_gap.pdf"):
    spectral_gaps = [r["spectral_gap"] for r in results]
    fiedler_values = [r["fiedler_value"] for r in results]
    converged = [r["converged"] for r in results]
    n_agents = [r["n_agents"] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    colors = ['#2ecc71' if c else '#e74c3c' for c in converged]
    ax.scatter(spectral_gaps, fiedler_values, c=colors, alpha=0.6, s=50)
    threshold = analysis["optimal_threshold"]
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
    ax.set_xlabel('Spectral Gap (λ₂)')
    ax.set_ylabel('Fiedler Value')
    ax.set_title('Consensus Outcome vs. Spectral Properties')
    ax.legend()
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Converged'),
                       Patch(facecolor='#e74c3c', label='Collapsed'),
                       plt.Line2D([0], [0], color='black', linestyle='--', label=f'Threshold = {threshold:.3f}')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax = axes[1]
    for converged_flag in [True, False]:
        mask = [c == converged_flag for c in converged]
        n_agents_subset = [n for n, m in zip(n_agents, mask) if m]
        spectral_subset = [s for s, m in zip(spectral_gaps, mask) if m]
        label = 'Converged' if converged_flag else 'Collapsed'
        color = '#2ecc71' if converged_flag else '#e74c3c'
        ax.scatter(n_agents_subset, spectral_subset, c=color, alpha=0.6, s=50, label=label)
    
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Number of Agents (N)')
    ax.set_ylabel('Spectral Gap (λ₂)')
    ax.set_title('Phase Transition in Agent Count')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.savefig(output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {output}")

def plot_trajectories(results, output="figure2_trajectories.pdf"):
    converged_runs = [r for r in results if r["converged"]][:5]
    collapsed_runs = [r for r in results if not r["converged"]][:5]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    for run in converged_runs:
        traj = run["consensus_trajectory"]
        ax.plot(traj, alpha=0.7, linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Consensus Metric')
    ax.set_title('Converged Swarms (Exponential Decay)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for run in collapsed_runs:
        traj = run["consensus_trajectory"]
        ax.plot(traj, alpha=0.7, linewidth=2, color='#e74c3c')
    ax.set_xlabel('Round')
    ax.set_ylabel('Consensus Metric')
    ax.set_title('Collapsed Swarms (No Convergence)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.savefig(output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {output}")

def plot_phase_diagram(results, output="figure3_phase_diagram.pdf"):
    graph_types = list(set(r["graph_type"] for r in results))
    
    fig, axes = plt.subplots(1, len(graph_types), figsize=(5*len(graph_types), 4))
    if len(graph_types) == 1:
        axes = [axes]
    
    for ax, gtype in zip(axes, graph_types):
        subset = [r for r in results if r["graph_type"] == gtype]
        n_vals = sorted(list(set(r["n_agents"] for r in subset)))
        
        conv_rates = []
        spectral_gaps = []
        for n in n_vals:
            n_subset = [r for r in subset if r["n_agents"] == n]
            conv_rate = np.mean([r["converged"] for r in n_subset])
            avg_spectral = np.mean([r["spectral_gap"] for r in n_subset])
            conv_rates.append(conv_rate)
            spectral_gaps.append(avg_spectral)
        
        ax2 = ax.twinx()
        ax.plot(n_vals, conv_rates, 'b-o', linewidth=2, markersize=8, label='Convergence Rate')
        ax2.plot(n_vals, spectral_gaps, 'r--s', linewidth=2, markersize=8, label='Spectral Gap')
        
        ax.set_xlabel('Number of Agents (N)')
        ax.set_ylabel('Convergence Rate', color='b')
        ax2.set_ylabel('Spectral Gap (λ₂)', color='r')
        ax.set_title(f'{gtype.capitalize()} Graph')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.savefig(output.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved {output}")

def generate_summary_table(results, analysis):
    graph_types = list(set(r["graph_type"] for r in results))
    
    print("\n" + "="*80)
    print("SUMMARY: Spectral Predictors of Consensus Collapse")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total trials: {analysis['total_samples']}")
    print(f"  Overall convergence rate: {analysis['convergence_rate']*100:.1f}%")
    print(f"  Spectral gap threshold: {analysis['optimal_threshold']:.3f}")
    print(f"  Predictive accuracy: {analysis['predictive_accuracy']*100:.1f}%")
    print(f"  Spearman correlation: r={analysis['spearman_r']:.3f}, p={analysis['spearman_p']:.2e}")
    
    print(f"\nBy Graph Type:")
    for gtype in graph_types:
        subset = [r for r in results if r["graph_type"] == gtype]
        conv_rate = np.mean([r["converged"] for r in subset])
        avg_spectral = np.mean([r["spectral_gap"] for r in subset])
        print(f"  {gtype:12s}: {conv_rate*100:5.1f}% converged, avg λ₂={avg_spectral:.3f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    results, analysis = load_results()
    
    plot_spectral_gap_vs_convergence(results, analysis)
    plot_trajectories(results)
    plot_phase_diagram(results)
    generate_summary_table(results, analysis)
    
    print("\nAll figures generated successfully.")
