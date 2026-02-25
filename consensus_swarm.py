import numpy as np
import networkx as nx
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

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
            "algebraic_connectivity": float(nx.algebraic_connectivity(self.graph))
        }

def run_scaling_experiment(max_agents: int = 50, n_trials: int = 5) -> List[dict]:
    results = []
    graph_types = ["complete", "random", "scale_free"]
    
    for n_agents in range(3, max_agents + 1, 2):
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
    
    threshold_candidates = np.linspace(0.01, 2.0, 100)
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
    
    return {
        "optimal_threshold": float(best_threshold),
        "predictive_accuracy": float(best_accuracy),
        "pearson_r": float(corr_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(corr_spearman),
        "spearman_p": float(p_spearman),
        "total_samples": len(results),
        "convergence_rate": float(np.mean(converged))
    }

if __name__ == "__main__":
    print("Running scaling experiment...")
    results = run_scaling_experiment(max_agents=30, n_trials=3)
    
    print(f"Completed {len(results)} trials")
    
    analysis = analyze_spectral_predictor(results)
    print("\nSpectral Predictor Analysis:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
    
    with open("results.json", "w") as f:
        json.dump({"results": results, "analysis": analysis}, f, indent=2)
    
    print("\nResults saved to results.json")
    
    collapsed = sum(1 for r in results if not r["converged"])
    print(f"\nConsensus collapsed in {collapsed}/{len(results)} trials")
    print(f"Collapse rate: {collapsed/len(results)*100:.1f}%")
