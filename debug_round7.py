"""
Debug: Check what happens in round 7 when consensus jumps to 1.0
"""
import os
import asyncio
import numpy as np
import networkx as nx
import aiohttp

PERSONAS = [
    "You are a data-driven analyst who values evidence.",
    "You are an optimistic innovator focused on solutions.",
    "You are a skeptical critic who questions assumptions.",
    "You are a pragmatic policymaker focused on feasibility.",
    "You are a community advocate prioritizing people."
]

QUESTION = "What is the most effective policy to address climate change?"

async def debug_round_7():
    """Replicate the first trial and see what happens at round 7"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    n_agents = 5
    graph = nx.complete_graph(n_agents)
    agent_opinions = [None] * n_agents
    
    print("Replicating: climate, mixed, complete, N=5")
    print(f"Question: {QUESTION}")
    print()
    
    for round_num in range(8):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1}")
        print(f"{'='*60}")
        
        round_opinions = []
        errors = 0
        
        for agent_id in range(n_agents):
            # Build context
            neighbors = list(graph.neighbors(agent_id))
            context = ""
            
            if neighbors and round_num > 0 and any(agent_opinions[n] for n in neighbors):
                neighbor_ops = []
                for nid in neighbors:
                    if agent_opinions[nid]:
                        neighbor_ops.append(f"Agent {nid}: {agent_opinions[nid][:60]}...")
                if neighbor_ops:
                    context = "\n".join(neighbor_ops[:2])
            
            persona = PERSONAS[agent_id % len(PERSONAS)]
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": f"{persona} Answer in 1-2 sentences."},
                    {"role": "user", "content": f"Question: {QUESTION}" + (f"\n\nOther opinions:\n{context}" if context else "")}
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
                            round_opinions.append(opinion)
                            print(f"Agent {agent_id}: {opinion[:80]}...")
                        else:
                            error_text = await resp.text()
                            print(f"Agent {agent_id}: ERROR {resp.status} - {error_text[:50]}")
                            round_opinions.append("")
                            errors += 1
            except Exception as e:
                print(f"Agent {agent_id}: EXCEPTION - {str(e)[:50]}")
                round_opinions.append("")
                errors += 1
            
            await asyncio.sleep(0.05)
        
        # Calculate consensus
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            vectors = vectorizer.fit_transform(round_opinions)
            sim_matrix = cosine_similarity(vectors)
            
            upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            consensus = float(np.mean(upper_triangle))
            
        except Exception as e:
            print(f"Consensus error: {e}")
            consensus = 0.5
        
        agent_opinions = round_opinions
        
        print(f"\nConsensus: {consensus:.3f}, Errors: {errors}")
        
        if errors > 0:
            print("⚠️  ERRORS DETECTED - this may cause artificial consensus")

if __name__ == "__main__":
    asyncio.run(debug_round_7())
