"""
Quick spectral predictor validation with GPT-3.5-turbo
"""
import os
import asyncio
import numpy as np
import networkx as nx
import json
import aiohttp

async def quick_llm_experiment():
    """Run 12 trials with GPT-3.5-turbo to validate spectral predictor"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key found")
        return
    
    configs = [
        (5, "complete"), (10, "complete"), (15, "complete"),
        (5, "cycle"), (10, "cycle"), (15, "cycle"),
        (5, "random"), (10, "random"), (15, "random"),
        (5, "scale_free"), (10, "scale_free"), (15, "scale_free")
    ]
    
    question = "What is the most effective policy to address climate change?"
    
    personas = [
        "You are a progressive environmental activist.",
        "You are a conservative business owner focused on profits.",
        "You are a pragmatic scientist who values data.",
        "You are a skeptical journalist who questions everything.",
        "You are a centrist politician seeking compromise."
    ]
    
    results = []
    
    for trial_num, (n_agents, graph_type) in enumerate(configs, 1):
        print(f"\n[{trial_num}/12] {graph_type}, N={n_agents}")
        
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
        else:  # scale_free
            G = nx.barabasi_albert_graph(n_agents, max(1, n_agents // 10))
        
        # Calculate spectral gap
        L = nx.laplacian_matrix(G).todense()
        eigenvals = np.sort(np.linalg.eigvalsh(L))
        spectral_gap = float(eigenvals[1]) if len(eigenvals) > 1 else 0
        
        # Run 8 rounds
        agent_opinions = [None] * n_agents
        trajectory = []
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        for round_num in range(8):
            new_opinions = []
            
            for agent_id in range(n_agents):
                neighbors = list(G.neighbors(agent_id))
                context = ""
                
                if neighbors and round_num > 0 and any(agent_opinions[n] for n in neighbors):
                    neighbor_ops = [f"Agent {n}: {agent_opinions[n]}" 
                                   for n in neighbors if agent_opinions[n]]
                    if neighbor_ops:
                        context = "\n".join(neighbor_ops[:2])
                
                persona = personas[agent_id % len(personas)]
                
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": f"{persona} Answer in 1-2 sentences."},
                        {"role": "user", "content": f"Question: {question}" + (f"\n\nOther opinions:\n{context}" if context else "")}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150
                }
                
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
                                opinion = data["choices"][0]["message"]["content"].strip()
                                new_opinions.append(opinion)
                            else:
                                new_opinions.append("")
                except Exception as e:
                    new_opinions.append("")
                
                await asyncio.sleep(0.05)
            
            agent_opinions = new_opinions
            
            # Calculate consensus
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
                vectors = vectorizer.fit_transform(agent_opinions)
                sim_matrix = cosine_similarity(vectors)
                upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                consensus = float(np.mean(upper_triangle))
            except:
                consensus = 0.5
            
            trajectory.append(consensus)
            print(f"  Round {round_num+1}: {consensus:.3f}")
        
        final_score = trajectory[-1]
        
        results.append({
            "n_agents": n_agents,
            "graph_type": graph_type,
            "spectral_gap": spectral_gap,
            "final_consensus_score": final_score,
            "consensus_trajectory": trajectory
        })
        
        print(f"  -> Final: {final_score:.3f}, λ₂={spectral_gap:.3f}")
    
    # Calculate predictive accuracy
    print("\n" + "="*60)
    print("SPECTRAL PREDICTOR ANALYSIS")
    print("="*60)
    
    from scipy.stats import spearmanr
    
    gaps = [r["spectral_gap"] for r in results]
    scores = [r["final_consensus_score"] for r in results]
    
    # Ranking accuracy
    gap_ranks = np.argsort(np.argsort(gaps))
    score_ranks = np.argsort(np.argsort(scores))
    correct = sum(g == s for g, s in zip(gap_ranks, score_ranks))
    accuracy = correct / len(results)
    
    r, p = spearmanr(gaps, scores)
    
    print(f"Spearman correlation: r = {r:.3f} (p = {p:.2e})")
    print(f"Ranking accuracy: {accuracy*100:.1f}% ({correct}/{len(results)})")
    
    print("\nBy graph type:")
    by_type = {}
    for r in results:
        gt = r["graph_type"]
        if gt not in by_type:
            by_type[gt] = []
        by_type[gt].append(r["final_consensus_score"])
    
    for gt, sc in sorted(by_type.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(sc)/len(sc)
        print(f"  {gt:15s}: {avg:.3f}")
    
    with open("gpt35_results.json", "w") as f:
        json.dump({"results": results, "accuracy": accuracy, "spearman_r": r}, f, indent=2)
    
    print("\n✓ Results saved to gpt35_results.json")

if __name__ == "__main__":
    asyncio.run(quick_llm_experiment())
