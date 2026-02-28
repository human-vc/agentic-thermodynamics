"""
144-trial replication with EMBEDDINGS
Pre-registered success criteria: ρ > 0.3 AND p < 0.05
Logs BOTH TF-IDF and embedding similarity for head-to-head comparison
"""
import os
import asyncio
import numpy as np
import networkx as nx
import json
import aiohttp
from dataclasses import dataclass
from typing import List, Dict, Optional

# Pre-registered success criteria
SUCCESS_RHO = 0.3
SUCCESS_P = 0.05
FAILURE_RHO = 0.15
FAILURE_P = 0.1

@dataclass
class TrialConfig:
    topic: str
    persona_type: str
    graph_type: str
    n_agents: int
    seed: int

TOPICS = {
    "climate": "What is the most effective policy to address climate change?",
    "healthcare": "Should healthcare be primarily public or private?",
    "education": "Should K-12 education focus more on standardized testing or project-based learning?",
    "economy": "Is globalization more beneficial or harmful for domestic workers?"
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

async def generate_opinion(persona: str, question: str, context: str, api_key: str) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_msg = f"{persona} Answer in 1-2 sentences."
    user_msg = f"Question: {question}"
    if context:
        user_msg += f"\n\nOther opinions:\n{context}"
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                    elif resp.status == 429:
                        wait = (2 ** attempt) + np.random.random()
                        await asyncio.sleep(wait)
                        continue
        except:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    
    return None

async def get_embedding(text: str, api_key: str) -> Optional[List[float]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["data"][0]["embedding"]
    except:
        pass
    
    return None

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def compute_tfidf_consensus(opinions):
    if len(opinions) < 2:
        return 1.0
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        vectors = vectorizer.fit_transform(opinions)
        sim_matrix = cosine_similarity(vectors)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        return float(np.mean(upper))
    except:
        return 0.5

def compute_embedding_consensus(embeddings):
    if len(embeddings) < 2:
        return 1.0
    
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
    
    return float(np.mean(similarities))

def build_graph(graph_type, n):
    if graph_type == "complete":
        G = nx.complete_graph(n)
    elif graph_type == "cycle":
        G = nx.cycle_graph(n)
    elif graph_type == "random":
        p = min(1.0, 4.0 / n)
        G = nx.erdos_renyi_graph(n, p)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, p)
    elif graph_type == "scale_free":
        G = nx.barabasi_albert_graph(n, max(1, n//10))
    else:
        G = nx.complete_graph(n)
    
    L = nx.laplacian_matrix(G).todense()
    eigenvals = np.sort(np.linalg.eigvalsh(L))
    spectral_gap = float(eigenvals[1]) if len(eigenvals) > 1 else 0
    
    return G, spectral_gap

async def run_trial(config, api_key, trial_id):
    G, spectral_gap = build_graph(config.graph_type, config.n_agents)
    
    question = TOPICS[config.topic]
    personas = PERSONA_SETS[config.persona_type]
    
    agent_opinions = [None] * config.n_agents
    
    for round_num in range(5):
        new_opinions = []
        
        for agent_id in range(config.n_agents):
            neighbors = list(G.neighbors(agent_id))
            context = ""
            
            if neighbors and round_num > 0 and any(agent_opinions[n] for n in neighbors):
                neighbor_ops = [f"Agent {n}: {agent_opinions[n][:60]}..." 
                               for n in neighbors if agent_opinions[n]]
                if neighbor_ops:
                    context = "\n".join(neighbor_ops[:3])
            
            persona = personas[agent_id % len(personas)]
            opinion = await generate_opinion(persona, question, context, api_key)
            
            if opinion is None:
                return None
            
            new_opinions.append(opinion)
            await asyncio.sleep(1.0)
        
        agent_opinions = new_opinions
    
    tfidf_consensus = compute_tfidf_consensus(agent_opinions)
    
    embeddings = []
    for opinion in agent_opinions:
        emb = await get_embedding(opinion, api_key)
        if emb:
            embeddings.append(emb)
        else:
            return None
        await asyncio.sleep(0.2)
    
    embedding_consensus = compute_embedding_consensus(embeddings)
    
    return {
        "trial_id": trial_id,
        "topic": config.topic,
        "persona_type": config.persona_type,
        "graph_type": config.graph_type,
        "n_agents": config.n_agents,
        "spectral_gap": spectral_gap,
        "tfidf_consensus": tfidf_consensus,
        "embedding_consensus": embedding_consensus
    }

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key")
        return
    
    print("="*70)
    print("144-TRIAL EMBEDDING REPLICATION")
    print("="*70)
    print(f"Pre-registered success criteria:")
    print(f"  SUCCESS: ρ > {SUCCESS_RHO} AND p < {SUCCESS_P}")
    print(f"  FAILURE: ρ < {FAILURE_RHO} OR p > {FAILURE_P}")
    print("="*70)
    print()
    
    topics = ["climate", "healthcare", "education", "economy"]
    persona_types = ["mixed", "homogeneous", "polarized"]
    graph_types = ["complete", "cycle", "random", "scale_free"]
    agent_counts = [5, 10, 15]
    
    configs = []
    trial_num = 0
    for topic in topics:
        for persona in persona_types:
            for graph in graph_types:
                for n in agent_counts:
                    trial_num += 1
                    configs.append((trial_num, TrialConfig(topic, persona, graph, n, trial_num * 100)))
    
    print(f"Total trials: {len(configs)}")
    print(f"Estimated cost: ${len(configs) * 0.20:.2f}")
    print(f"Estimated time: 14-18 hours")
    print()
    
    results = []
    failed = 0
    
    for i, (trial_num, config) in enumerate(configs, 1):
        trial_id = f"T{trial_num:03d}_{config.topic}_{config.persona_type}_{config.graph_type}_N{config.n_agents}"
        print(f"[{i}/{len(configs)}] {trial_id}")
        
        result = await run_trial(config, api_key, trial_id)
        
        if result:
            results.append(result)
            print(f"  ✓ TF-IDF: {result['tfidf_consensus']:.3f}, Emb: {result['embedding_consensus']:.3f}")
        else:
            failed += 1
            print(f"  ❌ Failed")
    
    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"Valid: {len(results)} / {len(configs)}")
    
    if len(results) < 50:
        print("Not enough data")
        return
    
    from scipy.stats import spearmanr
    
    gaps = [r['spectral_gap'] for r in results]
    tfidf_scores = [r['tfidf_consensus'] for r in results]
    emb_scores = [r['embedding_consensus'] for r in results]
    
    tfidf_corr, tfidf_p = spearmanr(gaps, tfidf_scores)
    emb_corr, emb_p = spearmanr(gaps, emb_scores)
    
    print(f"\nTF-IDF: ρ = {tfidf_corr:.3f}, p = {tfidf_p:.4f}")
    print(f"Embedding: ρ = {emb_corr:.3f}, p = {emb_p:.4f}")
    
    print()
    print("="*70)
    print("EVALUATION")
    print("="*70)
    
    if emb_corr > SUCCESS_RHO and emb_p < SUCCESS_P:
        print("✅ SUCCESS: Spectral predictor works")
    elif emb_corr < FAILURE_RHO or emb_p > FAILURE_P:
        print("❌ FAILURE: Hypothesis rejected - pivot to methodological paper")
    else:
        print("⚠️ AMBIGUOUS: Gray zone")
    
    with open('embedding_replication_144.json', 'w') as f:
        json.dump({
            'results': results,
            'tfidf': {'rho': tfidf_corr, 'p': tfidf_p},
            'embedding': {'rho': emb_corr, 'p': emb_p}
        }, f, indent=2)
    
    print("\n✓ Saved to embedding_replication_144.json")

if __name__ == "__main__":
    asyncio.run(main())
