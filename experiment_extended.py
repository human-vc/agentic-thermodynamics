import numpy as np
import networkx as nx
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

@dataclass
class SwarmConfig:
    n_agents: int
    n_rounds: int
    graph_type: str
    opinion_dim: int = 10
    consensus_threshold: float = 0.5
    noise_std: float = 0.02
    seed: int = 42

class LLMAgent:
    def __init__(self, agent_id: int, opinion_dim: int, bias: Optional[np.ndarray] = None):
        self.id = agent_id
        self.opinion = np.random.randn(opinion_dim)
        self.opinion /= np.linalg.norm(self.opinion) + 1e-8
        self.bias = bias if bias is not None else np.random.randn(opinion_dim) * 0.1
        self.history = [self.opinion.copy()]
    
    def update_opinion(self, neighbor_opinions: List[np.ndarray], weights: List[float], noise_std: float):
        if not neighbor_opinions:
            return
        weighted_sum = sum(w * op for w, op in zip(weights, neighbor_opinions))
        weighted_sum /= sum(weights) + 1e-8
        noise = np.random.randn(len(self.opinion)) * noise_std
        self.opinion = weighted_sum + self.bias + noise
        self.opinion /= np.linalg.norm(self.opinion) + 1e-8
        self.history.append(self.opinion.copy())

class ConsensusSwarm:
    def __init__(self, config: SwarmConfig):
        self.config = config
        np.random.seed(config.seed)
        self.agents = [LLMAgent(i, config.opinion_dim) for i in range(config.n_agents)]
        self.graph = self._build_graph()
        self.laplacian = nx.laplacian_matrix(self.graph).todense()
        self.eigenvalues = self._compute_spectrum()
        self.consensus_metric_history = []
        self.spectral_gap = self.eigenvalues[1] if len(self.eigenvalues) > 1 else 0
        
    def _build_graph(self) -> nx.Graph:
        n = self.config.n_agents
        if self.config.graph_type == "complete":
            G = nx.complete_graph(n)
        elif self.config.graph_type == "cycle":
            G = nx.cycle_graph(n)
        elif self.config.graph_type == "star":
            G = nx.star_graph(n-1)
        elif self.config.graph_type == "random":
            p = min(1.0, 4.0 / n)
            G = nx.erdos_renyi_graph(n, p)
        elif self.config.graph_type == "scale_free":
            G = nx.barabasi_albert_graph(n, max(1, n // 10))
        else:
            G = nx.complete_graph(n)
        return G
    
    def _compute_spectrum(self) -> np.ndarray:
        eigenvals = np.linalg.eigvalsh(self.laplacian)
        return np.sort(eigenvals)
    
    def _compute_consensus_metric(self) -> float:
        opinions = np.array([a.opinion for a in self.agents])
        centroid = np.mean(opinions, axis=0)
        distances = np.linalg.norm(opinions - centroid, axis=1)
        return np.mean(distances)
    
    def run_consensus_dynamics(self) -> dict:
        for round_idx in range(self.config.n_rounds):
            consensus_metric = self._compute_consensus_metric()
            self.consensus_metric_history.append(consensus_metric)
            
            if consensus_metric < self.config.consensus_threshold:
                break
            
            for agent in self.agents:
                neighbors = list(self.graph.neighbors(agent.id))
                if neighbors:
                    neighbor_opinions = [self.agents[n].opinion for n in neighbors]
                    weights = [1.0] * len(neighbors)
                    agent.update_opinion(neighbor_opinions, weights, self.config.noise_std)
        
        final_metric = self._compute_consensus_metric()
        converged = final_metric < self.config.consensus_threshold
        
        return {
            "n_agents": self.config.n_agents,
            "graph_type": self.config.graph_type,
            "spectral_gap": float(self.spectral_gap),
            "fiedler_value": float(self.eigenvalues[1]) if len(self.eigenvalues) > 1 else 0,
            "final_consensus_metric": float(final_metric),
            "converged": bool(converged),
            "rounds_to_converge": len(self.consensus_metric_history) if converged else self.config.n_rounds,
            "consensus_trajectory": [float(x) for x in self.consensus_metric_history],
            "algebraic_connectivity": float(nx.algebraic_connectivity(self.graph)),
            "graph_density": float(nx.density(self.graph)),
            "max_degree": int(max(dict(self.graph.degree()).values())),
            "min_degree": int(min(dict(self.graph.degree()).values()))
        }

def run_scaling_experiment(max_agents: int = 50, n_trials: int = 5) -> List[dict]:
    results = []
    graph_types = ["complete", "random", "scale_free", "cycle"]
    
    for n_agents in tqdm(range(3, max_agents + 1, 2), desc="Agent count"):
        for graph_type in graph_types:
            for trial in range(n_trials):
                config = SwarmConfig(
                    n_agents=n_agents,
                    n_rounds=100,
                    graph_type=graph_type,
                    seed=trial * 1000 + n_agents
                )
                swarm = ConsensusSwarm(config)
                result = swarm.run_consensus_dynamics()
                result["trial"] = trial
                results.append(result)
    
    return results

def run_critical_threshold_experiment(n_agents: int = 20, n_graphs: int = 100) -> List[dict]:
    results = []
    
    for trial in tqdm(range(n_graphs), desc="Graph samples"):
        p = np.random.uniform(0.05, 0.8)
        G = nx.erdos_renyi_graph(n_agents, p)
        
        if not nx.is_connected(G):
            continue
        
        config = SwarmConfig(
            n_agents=n_agents,
            n_rounds=100,
            graph_type="random",
            seed=trial
        )
        
        np.random.seed(config.seed)
        agents = [LLMAgent(i, config.opinion_dim) for i in range(config.n_agents)]
        laplacian = nx.laplacian_matrix(G).todense()
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
        
        consensus_metric_history = []
        for round_idx in range(config.n_rounds):
            opinions = np.array([a.opinion for a in agents])
            centroid = np.mean(opinions, axis=0)
            distances = np.linalg.norm(opinions - centroid, axis=1)
            consensus_metric = np.mean(distances)
            consensus_metric_history.append(consensus_metric)
            
            if consensus_metric < config.consensus_threshold:
                break
            
            for agent in agents:
                neighbors = list(G.neighbors(agent.id))
                if neighbors:
                    neighbor_opinions = [agents[n].opinion for n in neighbors]
                    weights = [1.0] * len(neighbors)
                    weighted_sum = sum(w * op for w, op in zip(weights, neighbor_opinions))
                    weighted_sum /= sum(weights) + 1e-8
                    noise = np.random.randn(len(agent.opinion)) * config.noise_std
                    agent.opinion = weighted_sum + agent.bias + noise
                    agent.opinion /= np.linalg.norm(agent.opinion) + 1e-8
        
        final_metric = np.mean([np.linalg.norm(a.opinion - np.mean([ag.opinion for ag in agents], axis=0)) for a in agents])
        converged = final_metric < config.consensus_threshold
        
        results.append({
            "n_agents": n_agents,
            "graph_type": "random",
            "spectral_gap": float(spectral_gap),
            "fiedler_value": float(eigenvalues[1]) if len(eigenvalues) > 1 else 0,
            "edge_probability": float(p),
            "final_consensus_metric": float(final_metric),
            "converged": bool(converged),
            "graph_density": float(nx.density(G)),
            "algebraic_connectivity": float(nx.algebraic_connectivity(G))
        })
    
    return results

def analyze_spectral_predictor(results: List[dict]) -> dict:
    spectral_gaps = []
    converged = []
    n_agents_list = []
    
    for r in results:
        spectral_gaps.append(r["spectral_gap"])
        converged.append(1 if r["converged"] else 0)
        n_agents_list.append(r["n_agents"])
    
    spectral_gaps = np.array(spectral_gaps)
    converged = np.array(converged)
    n_agents_list = np.array(n_agents_list)
    
    threshold_candidates = np.linspace(0.01, max(spectral_gaps), 100)
    best_accuracy = 0
    best_threshold = 0
    
    for thresh in threshold_candidates:
        predicted = (spectral_gaps >= thresh).astype(int)
        accuracy = np.mean(predicted == converged)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    
    from scipy.stats import pearsonr, spearmanr
    
    if len(set(converged)) > 1 and len(set(spectral_gaps)) > 1:
        corr_pearson, p_pearson = pearsonr(spectral_gaps, converged)
        corr_spearman, p_spearman = spearmanr(spectral_gaps, converged)
    else:
        corr_pearson, p_pearson = 0.0, 1.0
        corr_spearman, p_spearman = 0.0, 1.0
    
    fiedler_values = np.array([r["fiedler_value"] for r in results])
    if len(set(fiedler_values)) > 1 and len(set(converged)) > 1:
        fiedler_corr, fiedler_p = spearmanr(fiedler_values, converged)
    else:
        fiedler_corr, fiedler_p = 0.0, 1.0
    
    return {
        "optimal_threshold": float(best_threshold),
        "predictive_accuracy": float(best_accuracy),
        "pearson_r": float(corr_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(corr_spearman),
        "spearman_p": float(p_spearman),
        "fiedler_spearman_r": float(fiedler_corr),
        "fiedler_spearman_p": float(fiedler_p),
        "total_samples": len(results),
        "convergence_rate": float(np.mean(converged)),
        "mean_spectral_gap": float(np.mean(spectral_gaps)),
        "std_spectral_gap": float(np.std(spectral_gaps))
    }

def generate_detailed_report(results: List[dict], analysis: dict):
    print("\n" + "="*80)
    print("DETAILED EXPERIMENTAL REPORT")
    print("="*80)
    
    print("\n[1] SPECTRAL PREDICTOR VALIDATION")
    print(f"    Optimal threshold: λ₂ ≥ {analysis['optimal_threshold']:.4f}")
    print(f"    Classification accuracy: {analysis['predictive_accuracy']*100:.2f}%")
    print(f"    Spearman correlation: r = {analysis['spearman_r']:.4f} (p = {analysis['spearman_p']:.2e})")
    print(f"    Fiedler value correlation: r = {analysis['fiedler_spearman_r']:.4f}")
    
    print("\n[2] PHASE TRANSITION ANALYSIS")
    graph_types = list(set(r["graph_type"] for r in results))
    for gtype in sorted(graph_types):
        subset = [r for r in results if r["graph_type"] == gtype]
        n_vals = sorted(set(r["n_agents"] for r in subset))
        
        print(f"\n    {gtype.upper()} GRAPHS:")
        for n in n_vals[:5]:
            n_subset = [r for r in subset if r["n_agents"] == n]
            conv_rate = np.mean([r["converged"] for r in n_subset])
            avg_gap = np.mean([r["spectral_gap"] for r in n_subset])
            print(f"      N={n:2d}: {conv_rate*100:5.1f}% converged, λ₂={avg_gap:.3f}")
    
    print("\n[3] COLLAPSE MECHANISM")
    collapsed = [r for r in results if not r["converged"]]
    converged = [r for r in results if r["converged"]]
    
    if collapsed:
        avg_gap_collapsed = np.mean([r["spectral_gap"] for r in collapsed])
        avg_gap_converged = np.mean([r["spectral_gap"] for r in converged])
        print(f"    Avg spectral gap (collapsed):  {avg_gap_collapsed:.3f}")
        print(f"    Avg spectral gap (converged):  {avg_gap_converged:.3f}")
        print(f"    Ratio: {avg_gap_converged / max(avg_gap_collapsed, 0.001):.2f}x")
    
    print("\n[4] SUMMARY STATISTICS")
    print(f"    Total trials: {analysis['total_samples']}")
    print(f"    Overall convergence rate: {analysis['convergence_rate']*100:.1f}%")
    print(f"    Spectral gap range: {analysis['mean_spectral_gap']:.3f} ± {analysis['std_spectral_gap']:.3f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("Experiment 1: Scaling with agent count")
    results1 = run_scaling_experiment(max_agents=30, n_trials=3)
    
    print("\nExperiment 2: Critical threshold detection")
    results2 = run_critical_threshold_experiment(n_agents=20, n_graphs=100)
    
    all_results = results1 + results2
    
    analysis = analyze_spectral_predictor(all_results)
    generate_detailed_report(all_results, analysis)
    
    with open("results_extended.json", "w") as f:
        json.dump({"results": all_results, "analysis": analysis}, f, indent=2)
    
    print("\nResults saved to results_extended.json")
