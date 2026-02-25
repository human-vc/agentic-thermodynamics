"""
Debug script: Audit raw LLM responses round-by-round
Check for caching, deduplication, and actual convergence
"""
import os
import numpy as np
import networkx as nx
import asyncio
import aiohttp
import json

async def debug_single_trial():
    """Run one trial and print raw responses from each round"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key found")
        return
    
    # Config
    n_agents = 5
    graph_type = "complete"
    n_rounds = 5
    temperature = 0.7  # EXPLICIT
    
    # Build graph
    G = nx.complete_graph(n_agents)
    L = nx.laplacian_matrix(G).todense()
    eigenvals = np.sort(np.linalg.eigvalsh(L))
    spectral_gap = float(eigenvals[1])
    
    print(f"\n{'='*70}")
    print(f"DEBUG TRIAL: {graph_type}, N={n_agents}, λ₂={spectral_gap:.3f}")
    print(f"Temperature: {temperature}")
    print(f"{'='*70}\n")
    
    # Personas
    personas = [
        "You are a progressive environmental activist who wants radical change.",
        "You are a conservative business owner who prioritizes economic growth.",
        "You are a pragmatic scientist who only trusts peer-reviewed data.",
        "You are a skeptical journalist who questions all claims.",
        "You are a centrist politician seeking compromise solutions."
    ]
    
    question = "What is the most effective policy to address climate change?"
    
    # Agent state
    agent_opinions = [None] * n_agents
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    all_round_data = []
    
    for round_num in range(n_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}")
        print(f"{'='*70}")
        
        round_opinions = []
        
        for agent_id in range(n_agents):
            # Build context from neighbors
            neighbors = list(G.neighbors(agent_id))
            context = ""
            
            if neighbors and round_num > 0 and any(agent_opinions[n] for n in neighbors):
                neighbor_ops = []
                for nid in neighbors:
                    if agent_opinions[nid]:
                        neighbor_ops.append(f"Agent {nid}: {agent_opinions[nid][:80]}...")
                if neighbor_ops:
                    context = "\n".join(neighbor_ops[:2])
            
            persona = personas[agent_id % len(personas)]
            
            # Build prompt
            system_msg = f"{persona} Answer concisely in 1-2 sentences. Be specific and opinionated."
            user_msg = f"Question: {question}"
            if context:
                user_msg += f"\n\nYou heard these opinions from other agents:\n{context}\n\nConsider these views but maintain your own perspective."
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": temperature,
                "max_tokens": 150
            }
            
            print(f"\n--- Agent {agent_id} ---")
            print(f"Persona: {persona[:60]}...")
            if context:
                print(f"Context received: {context[:100]}...")
            else:
                print(f"Context: None (first round)")
            
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
                            round_opinions.append(opinion)
                            print(f"Response: {opinion}")
                        else:
                            error = await resp.text()
                            print(f"ERROR: {resp.status} - {error}")
                            round_opinions.append("")
            except Exception as e:
                print(f"EXCEPTION: {e}")
                round_opinions.append("")
            
            await asyncio.sleep(0.1)
        
        # Calculate consensus
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            vectors = vectorizer.fit_transform(round_opinions)
            sim_matrix = cosine_similarity(vectors)
            
            # Print similarity matrix
            print(f"\n--- Similarity Matrix ---")
            print(np.round(sim_matrix, 3))
            
            upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            consensus = float(np.mean(upper_triangle))
            print(f"\nConsensus score: {consensus:.3f}")
            
        except Exception as e:
            print(f"Consensus calculation error: {e}")
            consensus = 0.5
        
        agent_opinions = round_opinions
        all_round_data.append({
            "round": round_num + 1,
            "opinions": round_opinions,
            "consensus": consensus
        })
    
    # Final analysis
    print(f"\n{'='*70}")
    print("SUMMARY: Round 1 vs Round 5")
    print(f"{'='*70}")
    
    if len(all_round_data) >= 1:
        print("\n--- ROUND 1 OPINIONS ---")
        for i, op in enumerate(all_round_data[0]["opinions"]):
            print(f"Agent {i}: {op}")
    
    if len(all_round_data) >= 5:
        print("\n--- ROUND 5 OPINIONS ---")
        for i, op in enumerate(all_round_data[4]["opinions"]):
            print(f"Agent {i}: {op}")
        
        # Check if they're identical
        round1_ops = all_round_data[0]["opinions"]
        round5_ops = all_round_data[4]["opinions"]
        
        identical = sum(1 for a, b in zip(round1_ops, round5_ops) if a == b)
        print(f"\nIdentical opinions (Round 1 vs 5): {identical}/{n_agents}")
        
        if identical == n_agents:
            print("⚠️ WARNING: All opinions identical - possible caching or forced convergence")
        elif identical > 0:
            print("~ Some opinions unchanged")
        else:
            print("✓ All opinions changed between rounds")
    
    # Save debug data
    with open("debug_trial.json", "w") as f:
        json.dump({
            "config": {
                "n_agents": n_agents,
                "graph_type": graph_type,
                "spectral_gap": spectral_gap,
                "temperature": temperature,
                "model": "gpt-4o-mini"
            },
            "rounds": all_round_data
        }, f, indent=2)
    
    print(f"\n✓ Debug data saved to debug_trial.json")

if __name__ == "__main__":
    asyncio.run(debug_single_trial())
