"""
Debug: Check for state leakage between trials
"""
import os
import asyncio
import numpy as np
import networkx as nx
import json
import aiohttp
from dataclasses import dataclass
from typing import Optional

# Copy the exact classes from extended_experiment.py
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
}

TOPICS = {
    "climate": "What is the most effective policy to address climate change?",
    "healthcare": "Should healthcare be primarily public or private?",
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
        
    def _build_graph(self) -> nx.Graph:
        n = self.config.n_agents
        gt = self.config.graph_type
        
        if gt == "complete":
            return nx.complete_graph(n)
        elif gt == "cycle":
            return nx.cycle_graph(n)
        elif gt == "random":
            p = min(1.0, 4.0 / n)
            G = nx.erdos_renyi_graph(n, p)
            while not nx.is_connected(G):
                G = nx.erdos_renyi_graph(n, p)
            return G
        elif gt == "scale_free":
            return nx.barabasi_albert_graph(n, max(1, n // 10))
        return nx.complete_graph(n)
    
    def _compute_consensus_score(self, opinions):
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
        except Exception as e:
            print(f"Consensus calculation error: {e}")
            return 0.5
    
    async def run_consensus_round(self, question: str, round_num: int):
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
    
    async def run_full_consensus(self, question: str):
        for round_num in range(self.config.n_rounds):
            await self.run_consensus_round(question, round_num)
            
            if self.consensus_scores[-1] >= self.config.consensus_threshold:
                break
        
        final_score = self.consensus_scores[-1]
        converged = final_score >= self.config.consensus_threshold
        
        return {
            "n_agents": self.config.n_agents,
            "graph_type": self.config.graph_type,
            "topic": self.config.topic,
            "persona_type": self.config.persona_type,
            "spectral_gap": self.spectral_gap,
            "final_consensus_score": final_score,
            "converged": converged,
            "rounds_to_converge": len(self.consensus_scores),
            "consensus_trajectory": self.consensus_scores,
        }

async def debug_trials():
    """Run 2 trials back-to-back and check for state leakage"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("="*70)
    print("TRIAL 1: climate, mixed, complete, N=5")
    print("="*70)
    
    config1 = SwarmConfig(
        n_agents=5, n_rounds=3, graph_type="complete",
        consensus_threshold=0.6, seed=100,
        topic="climate", persona_type="mixed"
    )
    llm_config1 = LLMConfig(api_key=api_key)
    swarm1 = ConsensusSwarm(config1, llm_config1)
    
    result1 = await swarm1.run_full_consensus(TOPICS["climate"])
    print(f"Trajectory: {result1['consensus_trajectory']}")
    print(f"Final: {result1['final_consensus_score']:.3f}")
    
    print("\n" + "="*70)
    print("TRIAL 2: healthcare, homogeneous, cycle, N=5")
    print("="*70)
    
    config2 = SwarmConfig(
        n_agents=5, n_rounds=3, graph_type="cycle",
        consensus_threshold=0.6, seed=200,
        topic="healthcare", persona_type="homogeneous"
    )
    llm_config2 = LLMConfig(api_key=api_key)
    swarm2 = ConsensusSwarm(config2, llm_config2)
    
    result2 = await swarm2.run_full_consensus(TOPICS["healthcare"])
    print(f"Trajectory: {result2['consensus_trajectory']}")
    print(f"Final: {result2['final_consensus_score']:.3f}")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if result2['final_consensus_score'] >= 0.99:
        print("⚠️  Trial 2 has perfect consensus - state may be leaking!")
    else:
        print("✓ Trial 2 shows normal consensus - no state leakage detected")

if __name__ == "__main__":
    asyncio.run(debug_trials())
