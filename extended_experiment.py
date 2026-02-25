import os
import asyncio
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import aiohttp

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 150
    api_key: Optional[str] = None
    rate_limit_delay: float = 0.05

@dataclass
class SwarmConfig:
    n_agents: int
    n_rounds: int
    graph_type: str
    consensus_threshold: float = 0.6
    seed: int = 42
    topic: str = "climate"
    persona_type: str = "mixed"

TOPICS = {
    "climate": "What is the most effective policy to address climate change?",
    "healthcare": "Should healthcare be primarily public or private?",
    "education": "What is the best approach to improve K-12 education?",
    "tech": "How should AI be regulated?",
    "economy": "What is the best strategy to reduce income inequality?"
}

PERSONA_SETS = {
    "mixed": [
        "You are a data-driven analyst who values evidence.",
        "You are an optimistic innovator focused on solutions.",
        "You are a skeptical critic who questions assumptions.",
        "You are a pragmatic policymaker focused on feasibility.",
        "You are a community advocate prioritizing people."
    ],
    "homogeneous": [
        "You are a pragmatic analyst focused on evidence-based solutions."
    ] * 5,
    "polarized": [
        "You are a progressive advocate for systemic change.",
        "You are a conservative defender of traditional values.",
        "You are a progressive advocate for equity.",
        "You are a conservative proponent of free markets.",
        "You are a moderate seeking compromise."
    ]
}

class RealLLMAgent:
    def __init__(self, agent_id: int, config: LLMConfig, persona: str):
        self.id = agent_id
        self.config = config
        self.persona = persona
        self.opinion_history = []
        self.current_opinion = None
        
    async def generate_opinion(self, question: str, context: str = "") -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key or os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        system_msg = f"{self.persona} Answer in 1-2 sentences."
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\n\nOther opinions:\n{context}"
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        opinion = data["choices"][0]["message"]["content"].strip()
                        self.current_opinion = opinion
                        self.opinion_history.append(opinion)
                        return opinion
                    else:
                        return self._fallback()
        except:
            return self._fallback()
        finally:
            await asyncio.sleep(self.config.rate_limit_delay)
    
    def _fallback(self):
        return "Based on available information, further analysis is needed."

class ConsensusSwarm:
    def __init__(self, config: SwarmConfig, llm_config: LLMConfig):
        self.config = config
        np.random.seed(config.seed)
        self.llm_config = llm_config
        
        personas = PERSONA_SETS[config.persona_type]
        self.agents = [RealLLMAgent(i, llm_config, personas[i % len(personas)]) 
                       for i in range(config.n_agents)]
        self.graph = self._build_graph()
        self.laplacian = nx.laplacian_matrix(self.graph).todense()
        eigenvals = np.linalg.eigvalsh(self.laplacian)
        self.eigenvalues = np.sort(eigenvals)
        self.spectral_gap = float(self.eigenvalues[1]) if len(self.eigenvalues) > 1 else 0
        self.consensus_scores = []
        
        self.baseline_metrics = self._compute_baseline_metrics()
        
    def _build_graph(self) -> nx.Graph:
        n = self.config.n_agents
        gt = self.config.graph_type
        
        if gt == "complete":
            return nx.complete_graph(n)
        elif gt == "cycle":
            return nx.cycle_graph(n)
        elif gt == "star":
            return nx.star_graph(n-1)
        elif gt == "random":
            p = min(1.0, 4.0 / n)
            G = nx.erdos_renyi_graph(n, p)
            while not nx.is_connected(G):
                G = nx.erdos_renyi_graph(n, p)
            return G
        elif gt == "scale_free":
            return nx.barabasi_albert_graph(n, max(1, n // 10))
        return nx.complete_graph(n)
    
    def _compute_baseline_metrics(self) -> Dict:
        G = self.graph
        return {
            "avg_degree": float(np.mean(list(dict(G.degree()).values()))),
            "max_degree": int(max(dict(G.degree()).values())),
            "min_degree": int(min(dict(G.degree()).values())),
            "density": float(nx.density(G)),
            "clustering": float(nx.average_clustering(G)),
            "diameter": int(nx.diameter(G)) if nx.is_connected(G) else -1,
            "algebraic_connectivity": float(nx.algebraic_connectivity(G))
        }
    
    def _compute_consensus_score(self, opinions: List[str]) -> float:
        if len(opinions) < 2:
            return 1.0
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            vectors = vectorizer.fit_transform(opinions)
            sim_matrix = cosine_similarity(vectors)
            
            upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            return float(np.mean(upper_triangle))
        except:
            return 0.5
    
    async def run_consensus_round(self, question: str, round_num: int) -> Dict:
        tasks = []
        
        for agent in self.agents:
            neighbors = list(self.graph.neighbors(agent.id))
            context = ""
            
            if neighbors and round_num > 0:
                neighbor_ops = [f"Agent {n}: {self.agents[n].current_opinion}" 
                               for n in neighbors if self.agents[n].current_opinion]
                if neighbor_ops:
                    context = "\n".join(neighbor_ops[:3])
            
            tasks.append(agent.generate_opinion(question, context))
        
        opinions = await asyncio.gather(*tasks)
        consensus_score = self._compute_consensus_score(opinions)
        self.consensus_scores.append(consensus_score)
        
        return {"opinions": opinions, "consensus_score": consensus_score}
    
    async def run_full_consensus(self, question: str) -> Dict:
        for round_num in range(self.config.n_rounds):
            await self.run_consensus_round(question, round_num)
            
            if self.consensus_scores[-1] >= self.config.consensus_threshold:
                break
        
        final_score = self.consensus_scores[-1]
        converged = final_score >= self.config.consensus_threshold
        
        result = {
            "n_agents": self.config.n_agents,
            "graph_type": self.config.graph_type,
            "topic": self.config.topic,
            "persona_type": self.config.persona_type,
            "spectral_gap": self.spectral_gap,
            "fiedler_value": float(self.eigenvalues[1]) if len(self.eigenvalues) > 1 else 0,
            "final_consensus_score": final_score,
            "converged": converged,
            "rounds_to_converge": len(self.consensus_scores),
            "consensus_trajectory": self.consensus_scores,
            **{f"baseline_{k}": v for k, v in self.baseline_metrics.items()}
        }
        
        return result

async def run_extended_experiment(api_key: Optional[str] = None):
    topics = ["climate", "healthcare", "tech"]
    persona_types = ["mixed", "homogeneous", "polarized"]
    graph_types = ["complete", "cycle", "random", "scale_free"]
    agent_counts = [5, 10, 15]
    
    all_results = []
    
    total_trials = len(topics) * len(persona_types) * len(graph_types) * len(agent_counts)
    print(f"Running {total_trials} trials...")
    
    trial_num = 0
    for topic in topics:
        for persona_type in persona_types:
            for graph_type in graph_types:
                for n_agents in agent_counts:
                    trial_num += 1
                    print(f"\n[{trial_num}/{total_trials}] {topic}, {persona_type}, {graph_type}, N={n_agents}")
                    
                    config = SwarmConfig(
                        n_agents=n_agents,
                        n_rounds=8,
                        graph_type=graph_type,
                        consensus_threshold=0.6,
                        seed=trial_num * 100,
                        topic=topic,
                        persona_type=persona_type
                    )
                    
                    llm_config = LLMConfig(api_key=api_key)
                    swarm = ConsensusSwarm(config, llm_config)
                    
                    try:
                        question = TOPICS[topic]
                        result = await swarm.run_full_consensus(question)
                        all_results.append(result)
                        print(f"  -> Score: {result['final_consensus_score']:.3f}, Spectral: {result['spectral_gap']:.3f}")
                    except Exception as e:
                        print(f"  Error: {e}")
    
    return all_results

def compare_predictors(results: List[Dict]) -> Dict:
    from scipy.stats import spearmanr
    
    metrics_to_test = {
        "spectral_gap": "Spectral Gap (λ₂)",
        "baseline_avg_degree": "Average Degree",
        "baseline_density": "Graph Density",
        "baseline_clustering": "Clustering Coefficient",
        "baseline_algebraic_connectivity": "Algebraic Connectivity"
    }
    
    consensus_scores = [r["final_consensus_score"] for r in results]
    
    comparisons = {}
    for metric_key, metric_name in metrics_to_test.items():
        values = [r[metric_key] for r in results]
        if len(set(values)) > 1 and len(set(consensus_scores)) > 1:
            corr, pval = spearmanr(values, consensus_scores)
            comparisons[metric_key] = {
                "name": metric_name,
                "spearman_r": float(corr),
                "spearman_p": float(pval)
            }
    
    return comparisons

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("No API key found. Set OPENAI_API_KEY environment variable.")
        return
    
    print("Running extended experiment with multiple topics and persona distributions...")
    print("Estimated cost: $5-8\n")
    
    results = await run_extended_experiment(api_key)
    
    print("\n" + "="*60)
    print("COMPARING BASELINE PREDICTORS")
    print("="*60)
    
    comparisons = compare_predictors(results)
    for metric, stats in sorted(comparisons.items(), key=lambda x: -abs(x[1]["spearman_r"])):
        print(f"{stats['name']:25s}: r = {stats['spearman_r']:6.3f} (p = {stats['spearman_p']:.2e})")
    
    with open("extended_results.json", "w") as f:
        json.dump({"results": results, "comparisons": comparisons}, f, indent=2)
    
    print("\nResults saved to extended_results.json")

if __name__ == "__main__":
    asyncio.run(main())
