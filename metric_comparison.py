"""
Quick test: Embedding similarity vs TF-IDF
20 trials to check if metric was masking signal
"""
import os
import asyncio
import numpy as np
import networkx as nx
import json
import aiohttp

# Test configs - diverse sample
configs = [
    ("climate", "mixed", "complete", 5),
    ("climate", "mixed", "cycle", 5),
    ("healthcare", "polarized", "complete", 10),
    ("healthcare", "polarized", "cycle", 10),
    ("education", "homogeneous", "random", 15),
    ("education", "homogeneous", "scale_free", 15),
]

# Run 3-4 trials per config = ~20 total

PERSONAS = {
    "mixed": [
        "You are a data-driven analyst who values evidence.",
        "You are an optimistic innovator focused on solutions.",
        "You are a skeptical critic who questions assumptions.",
        "You are a pragmatic policymaker focused on feasibility.",
        "You are a community advocate prioritizing people."
    ],
    "polarized": [
        "You are a progressive advocate for systemic change.",
        "You are a conservative defender of traditional values.",
        "You are a progressive advocate for equity.",
        "You are a conservative proponent of free markets.",
        "You are a moderate seeking compromise."
    ],
    "homogeneous": [
        "You are a pragmatic analyst focused on evidence-based solutions."
    ] * 5
}

TOPICS = {
    "climate": "What is the most effective policy to address climate change?",
    "healthcare": "Should healthcare be primarily public or private?",
    "education": "Should K-12 education focus more on standardized testing or project-based learning?"
}

async def generate_opinion(persona, question, context, api_key):
    """Generate single opinion"""
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
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
    except:
        pass
    return None

async def get_embedding(text, api_key):
    """Get OpenAI embedding"""
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
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
    return None

def cosine_similarity(v1, v2):
    """Compute cosine similarity"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def run_trial(topic, persona_type, graph_type, n_agents, api_key):
    """Run single trial with both metrics"""
    
    # Build graph
    if graph_type == "complete":
        G = nx.complete_graph(n_agents)
    elif graph_type == "cycle":
        G = nx.cycle_graph(n_agents)
    elif graph_type == "random":
        p = min(1.0, 4.0 / n_agents)
        G = nx.erdos_renyi_graph(n_agents, p)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n_agents, p)
    elif graph_type == "scale_free":
        G = nx.barabasi_albert_graph(n_agents, max(1, n_agents // 10))
    
    # Spectral gap
    L = nx.laplacian_matrix(G).todense()
    eigenvals = np.sort(np.linalg.eigvalsh(L))
    spectral_gap = float(eigenvals[1]) if len(eigenvals) > 1 else 0
    
    question = TOPICS[topic]
    personas = PERSONAS[persona_type]
    
    # Run 3 rounds
    agent_opinions = [None] * n_agents
    
    for round_num in range(3):
        new_opinions = []
        
        for agent_id in range(n_agents):
            neighbors = list(G.neighbors(agent_id))
            context = ""
            
            if neighbors and round_num > 0 and any(agent_opinions[n] for n in neighbors):
                neighbor_ops = [f"Agent {n}: {agent_opinions[n][:60]}" 
                               for n in neighbors if agent_opinions[n]]
                if neighbor_ops:
                    context = "\n".join(neighbor_ops[:2])
            
            persona = personas[agent_id % len(personas)]
            opinion = await generate_opinion(persona, question, context, api_key)
            
            if opinion:
                new_opinions.append(opinion)
            else:
                return None  # Trial failed
            
            await asyncio.sleep(1.0)  # Rate limit
        
        agent_opinions = new_opinions
    
    # Compute TF-IDF consensus
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as tfidf_sim
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        vectors = vectorizer.fit_transform(agent_opinions)
        sim_matrix = tfidf_sim(vectors)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        tfidf_consensus = float(np.mean(upper))
    except:
        tfidf_consensus = 0.5
    
    # Compute embedding consensus
    embeddings = []
    for opinion in agent_opinions:
        emb = await get_embedding(opinion, api_key)
        if emb:
            embeddings.append(emb)
        await asyncio.sleep(0.1)
    
    if len(embeddings) >= 2:
        emb_sims = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                emb_sims.append(cosine_similarity(embeddings[i], embeddings[j]))
        embedding_consensus = float(np.mean(emb_sims))
    else:
        embedding_consensus = 0.5
    
    return {
        "topic": topic,
        "persona_type": persona_type,
        "graph_type": graph_type,
        "n_agents": n_agents,
        "spectral_gap": spectral_gap,
        "tfidf_consensus": tfidf_consensus,
        "embedding_consensus": embedding_consensus
    }

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key")
        return
    
    print("Running 20-trial metric comparison...")
    print("TF-IDF vs OpenAI text-embedding-3-small")
    print(f"Configs: {len(configs)} x 3-4 trials each")
    print()
    
    results = []
    trial_num = 0
    
    for topic, persona, graph, n in configs:
        for i in range(4):  # 4 trials per config
            trial_num += 1
            print(f"[{trial_num}/24] {topic}, {persona}, {graph}, N={n}")
            
            result = await run_trial(topic, persona, graph, n, api_key)
            
            if result:
                results.append(result)
                print(f"  ✓ TF-IDF: {result['tfidf_consensus']:.3f}, Embedding: {result['embedding_consensus']:.3f}, λ₂={result['spectral_gap']:.2f}")
            else:
                print(f"  ✗ Trial failed")
    
    print()
    print("="*60)
    print("METRIC COMPARISON RESULTS")
    print("="*60)
    
    if len(results) < 10:
        print(f"Not enough data ({len(results)} trials)")
        return
    
    # Correlations
    gaps = [r['spectral_gap'] for r in results]
    tfidf_scores = [r['tfidf_consensus'] for r in results]
    emb_scores = [r['embedding_consensus'] for r in results]
    
    from scipy.stats import spearmanr
    
    tfidf_corr, tfidf_p = spearmanr(gaps, tfidf_scores)
    emb_corr, emb_p = spearmanr(gaps, emb_scores)
    
    print(f"TF-IDF correlation: ρ = {tfidf_corr:.3f}, p = {tfidf_p:.3f}")
    print(f"Embedding correlation: ρ = {emb_corr:.3f}, p = {emb_p:.3f}")
    print()
    
    # Score ranges
    print(f"TF-IDF range: {min(tfidf_scores):.3f} - {max(tfidf_scores):.3f}")
    print(f"Embedding range: {min(emb_scores):.3f} - {max(emb_scores):.3f}")
    print()
    
    # By graph type
    by_graph = {}
    for r in results:
        g = r['graph_type']
        if g not in by_graph:
            by_graph[g] = {'tfidf': [], 'emb': []}
        by_graph[g]['tfidf'].append(r['tfidf_consensus'])
        by_graph[g]['emb'].append(r['embedding_consensus'])
    
    print("By Graph Type:")
    for g in ['complete', 'random', 'scale_free', 'cycle']:
        if g in by_graph:
            tfidf_mean = np.mean(by_graph[g]['tfidf'])
            emb_mean = np.mean(by_graph[g]['emb'])
            print(f"  {g:15s}: TF-IDF={tfidf_mean:.3f}, Embedding={emb_mean:.3f}")
    
    # Save
    with open('metric_comparison.json', 'w') as f:
        json.dump({
            'results': results,
            'tfidf_correlation': {'rho': tfidf_corr, 'p': tfidf_p},
            'embedding_correlation': {'rho': emb_corr, 'p': emb_p}
        }, f, indent=2)
    
    print()
    print("✓ Saved to metric_comparison.json")
    print(f"Estimated cost: ~${len(results) * 0.15:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
