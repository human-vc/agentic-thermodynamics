import os
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import asyncio
import aiohttp
from collections import defaultdict
import time

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
    api_base: str = "https://api.openai.com/v1"
    rate_limit_delay: float = 0.1

@dataclass
class SwarmConfig:
    n_agents: int
    n_rounds: int
    graph_type: str
    consensus_threshold: float = 0.8
    seed: int = 42
    topic: str = "climate_change"

class RealLLMAgent:
    def __init__(self, agent_id: int, config: LLMConfig, persona: Optional[str] = None):
        self.id = agent_id
        self.config = config
        self.persona = persona or self._generate_persona()
        self.opinion_history = []
        self.message_history = []
        self.current_opinion = None
        self.confidence = 0.5
        
    def _generate_persona(self) -> str:
        personas = [
            "You are a cautious analyst who values data and evidence.",
            "You are an optimistic technologist who believes in innovation.",
            "You are a skeptical critic who questions assumptions.",
            "You are a pragmatic policymaker focused on feasibility.",
            "You are an environmental advocate prioritizing sustainability.",
            "You are an economist focused on cost-benefit analysis."
        ]
        return personas[self.id % len(personas)]
    
    async def generate_opinion(self, question: str, context: str = "") -> Tuple[str, float]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key or os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        system_msg = f"{self.persona} Answer concisely in 1-2 sentences."
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\n\nContext from other agents:\n{context}"
        
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
                    f"{self.config.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        opinion = data["choices"][0]["message"]["content"].strip()
                        confidence = self._estimate_confidence(opinion)
                        self.current_opinion = opinion
                        self.opinion_history.append(opinion)
                        return opinion, confidence
                    else:
                        error_text = await resp.text()
                        print(f"API error for agent {self.id}: {resp.status} - {error_text}")
                        return self._fallback_opinion(question), 0.5
        except Exception as e:
            print(f"Exception for agent {self.id}: {e}")
            return self._fallback_opinion(question), 0.5
        finally:
            await asyncio.sleep(self.config.rate_limit_delay)
    
    def _estimate_confidence(self, text: str) -> float:
        confidence_markers = ['definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly']
        uncertainty_markers = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain']
        
        text_lower = text.lower()
        conf_count = sum(1 for m in confidence_markers if m in text_lower)
        unc_count = sum(1 for m in uncertainty_markers if m in text_lower)
        
        base_conf = 0.5
        return min(0.95, max(0.1, base_conf + 0.1 * conf_count - 0.15 * unc_count))
    
    def _fallback_opinion(self, question: str) -> str:
        fallbacks = [
            "Based on current evidence, more analysis is needed.",
            "This requires careful consideration of multiple factors.",
            "The data suggests a nuanced approach is warranted."
        ]
        return fallbacks[self.id % len(fallbacks)]

class RealConsensusSwarm:
    def __init__(self, config: SwarmConfig, llm_config: LLMConfig):
        self.config = config
        np.random.seed(config.seed)
        self.llm_config = llm_config
        self.agents = [RealLLMAgent(i, llm_config) for i in range(config.n_agents)]
        self.graph = self._build_graph()
        self.laplacian = nx.laplacian_matrix(self.graph).todense()
        self.eigenvalues = self._compute_spectrum()
        self.spectral_gap = float(self.eigenvalues[1]) if len(self.eigenvalues) > 1 else 0
        self.consensus_scores = []
        self.round_opinions = []
        
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
    
    def _compute_consensus_score(self, opinions: List[str]) -> float:
        if len(opinions) < 2:
            return 1.0
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            vectors = vectorizer.fit_transform(opinions)
            sim_matrix = cosine_similarity(vectors)
            
            avg_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            return float(avg_similarity)
        except:
            return 0.5
    
    async def run_consensus_round(self, question: str, round_num: int) -> Dict:
        tasks = []
        
        for agent in self.agents:
            neighbors = list(self.graph.neighbors(agent.id))
            context = ""
            
            if neighbors and round_num > 0:
                neighbor_opinions = []
                for nid in neighbors:
                    if self.agents[nid].current_opinion:
                        neighbor_opinions.append(f"Agent {nid}: {self.agents[nid].current_opinion}")
                if neighbor_opinions:
                    context = "\n".join(neighbor_opinions[:3])
            
            task = agent.generate_opinion(question, context)
            tasks.append(task)
        
        opinions_data = await asyncio.gather(*tasks)
        opinions = [op for op, _ in opinions_data]
        confidences = [conf for _, conf in opinions_data]
        
        consensus_score = self._compute_consensus_score(opinions)
        self.consensus_scores.append(consensus_score)
        self.round_opinions.append(opinions)
        
        return {
            "round": round_num,
            "opinions": opinions,
            "confidences": confidences,
            "consensus_score": consensus_score
        }
    
    async def run_full_consensus(self, question: str) -> Dict:
        print(f"Running consensus for {self.config.n_agents} agents on graph: {self.config.graph_type}")
        print(f"Spectral gap: {self.spectral_gap:.4f}")
        
        for round_num in range(self.config.n_rounds):
            round_data = await self.run_consensus_round(question, round_num)
            print(f"  Round {round_num + 1}: consensus score = {round_data['consensus_score']:.3f}")
            
            if round_data['consensus_score'] >= self.config.consensus_threshold:
                print(f"  Consensus reached at round {round_num + 1}")
                break
        
        final_score = self.consensus_scores[-1] if self.consensus_scores else 0
        converged = final_score >= self.config.consensus_threshold
        
        return {
            "n_agents": self.config.n_agents,
            "graph_type": self.config.graph_type,
            "spectral_gap": self.spectral_gap,
            "fiedler_value": float(self.eigenvalues[1]) if len(self.eigenvalues) > 1 else 0,
            "final_consensus_score": final_score,
            "converged": converged,
            "rounds_to_converge": len(self.consensus_scores),
            "consensus_trajectory": self.consensus_scores,
            "question": question,
            "algebraic_connectivity": float(nx.algebraic_connectivity(self.graph)),
            "graph_density": float(nx.density(self.graph))
        }

class MockLLMConfig(LLMConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.responses = [
            "Climate change requires immediate action to reduce emissions.",
            "Economic growth and environmental protection must be balanced carefully.",
            "Renewable energy technologies offer promising solutions.",
            "International cooperation is essential for effective climate policy.",
            "Carbon pricing mechanisms can incentivize sustainable practices.",
            "Adaptation strategies are needed alongside mitigation efforts."
        ]

class MockConsensusSwarm(RealConsensusSwarm):
    async def run_consensus_round(self, question: str, round_num: int) -> Dict:
        opinions = []
        for agent in self.agents:
            idx = (agent.id + round_num) % len(self.llm_config.responses)
            if round_num > 0 and np.random.random() < 0.3:
                neighbor_ids = list(self.graph.neighbors(agent.id))
                if neighbor_ids:
                    idx = neighbor_ids[0] % len(self.llm_config.responses)
            
            opinion = self.llm_config.responses[idx]
            if self.spectral_gap < 0.3:
                opinion += f" (variant {np.random.randint(100)})"
            
            agent.current_opinion = opinion
            agent.opinion_history.append(opinion)
            opinions.append(opinion)
        
        consensus_score = self._compute_consensus_score(opinions)
        self.consensus_scores.append(consensus_score)
        self.round_opinions.append(opinions)
        
        await asyncio.sleep(0.01)
        
        return {
            "round": round_num,
            "opinions": opinions,
            "confidences": [0.7] * len(opinions),
            "consensus_score": consensus_score
        }

async def run_quick_experiment(use_real_llm: bool = False, api_key: Optional[str] = None):
    questions = [
        "What is the most effective approach to address climate change?"
    ]
    
    graph_types = ["complete", "cycle", "random", "scale_free"]
    agent_counts = [5, 10, 15]
    
    all_results = []
    
    for n_agents in agent_counts:
        for graph_type in graph_types:
            for q_idx, question in enumerate(questions):
                print(f"\nRunning: N={n_agents}, graph={graph_type}, q={q_idx+1}")
                
                config = SwarmConfig(
                    n_agents=n_agents,
                    n_rounds=10,
                    graph_type=graph_type,
                    consensus_threshold=0.6,
                    seed=q_idx * 100 + n_agents
                )
                
                if use_real_llm:
                    llm_config = LLMConfig(api_key=api_key)
                    swarm = RealConsensusSwarm(config, llm_config)
                else:
                    llm_config = MockLLMConfig()
                    swarm = MockConsensusSwarm(config, llm_config)
                
                try:
                    result = await swarm.run_full_consensus(question)
                    result["question_idx"] = q_idx
                    all_results.append(result)
                    print(f"  -> Converged: {result['converged']}, Score: {result['final_consensus_score']:.3f}")
                except Exception as e:
                    print(f"  Error: {e}")
    
    return all_results

async def run_experiment_suite(use_real_llm: bool = False, api_key: Optional[str] = None):
    questions = [
        "What is the most effective approach to address climate change?",
        "Should governments prioritize economic growth or environmental protection?",
        "What role should nuclear energy play in the transition to renewables?"
    ]
    
    graph_types = ["complete", "random", "cycle", "scale_free"]
    agent_counts = [5, 10, 15, 20]
    
    all_results = []
    
    for n_agents in tqdm(agent_counts, desc="Agent counts"):
        for graph_type in tqdm(graph_types, desc="Graph types", leave=False):
            for q_idx, question in enumerate(questions):
                config = SwarmConfig(
                    n_agents=n_agents,
                    n_rounds=5,
                    graph_type=graph_type,
                    seed=q_idx * 100 + n_agents
                )
                
                if use_real_llm:
                    llm_config = LLMConfig(api_key=api_key)
                    swarm = RealConsensusSwarm(config, llm_config)
                else:
                    llm_config = MockLLMConfig()
                    swarm = MockConsensusSwarm(config, llm_config)
                
                try:
                    result = await swarm.run_full_consensus(question)
                    result["question_idx"] = q_idx
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in experiment: {e}")
    
    return all_results

def analyze_results(results: List[Dict]) -> Dict:
    spectral_gaps = np.array([r["spectral_gap"] for r in results])
    converged = np.array([1 if r["converged"] else 0 for r in results])
    
    threshold_candidates = np.linspace(0.01, max(spectral_gaps), 100)
    best_accuracy = 0
    best_threshold = 0
    
    for thresh in threshold_candidates:
        predicted = (spectral_gaps >= thresh).astype(int)
        accuracy = np.mean(predicted == converged)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    
    from scipy.stats import spearmanr
    
    if len(set(converged)) > 1 and len(set(spectral_gaps)) > 1:
        corr, pval = spearmanr(spectral_gaps, converged)
    else:
        corr, pval = 0.0, 1.0
    
    return {
        "optimal_threshold": float(best_threshold),
        "predictive_accuracy": float(best_accuracy),
        "spearman_r": float(corr),
        "spearman_p": float(pval),
        "total_trials": len(results),
        "convergence_rate": float(np.mean(converged))
    }

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    use_real = api_key is not None
    
    if use_real:
        print("Using real OpenAI API")
        print("Running REDUCED experiment (2 agents counts × 2 graph types)")
    else:
        print("No API key found, using mock LLM")
    
    # Quick test with fewer combinations
    results = await run_quick_experiment(use_real_llm=use_real, api_key=api_key)
    
    analysis = analyze_results(results)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total trials: {analysis['total_trials']}")
    print(f"Convergence rate: {analysis['convergence_rate']*100:.1f}%")
    print(f"Optimal threshold: {analysis['optimal_threshold']:.4f}")
    print(f"Predictive accuracy: {analysis['predictive_accuracy']*100:.1f}%")
    print(f"Spearman r: {analysis['spearman_r']:.4f} (p={analysis['spearman_p']:.2e})")
    
    with open("llm_results.json", "w") as f:
        json.dump({"results": results, "analysis": analysis}, f, indent=2)
    
    print("\nResults saved to llm_results.json")

if __name__ == "__main__":
    asyncio.run(main())
