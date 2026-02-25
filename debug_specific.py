"""
Debug: Investigate the 1.0 consensus case
healthcare, mixed, scale_free, N=10
"""
import os
import asyncio
import numpy as np
import networkx as nx
import aiohttp

# Exact setup
PERSONA_SETS = {
    "mixed": [
        "You are a data-driven analyst who values evidence.",
        "You are an optimistic innovator focused on solutions.",
        "You are a skeptical critic who questions assumptions.",
        "You are a pragmatic policymaker focused on feasibility.",
        "You are a community advocate prioritizing people."
    ],
}

TOPICS = {
    "healthcare": "Should healthcare be primarily public or private?",
}

async def debug_specific_case():
    api_key = os.getenv("OPENAI_API_KEY")
    
    n_agents = 10
    graph_type = "scale_free"
    topic = "healthcare"
    persona_type = "mixed"
    
    # Build scale-free graph
    G = nx.barabasi_albert_graph(n_agents, max(1, n_agents // 10))
    
    print(f"Graph: {graph_type}, N={n_agents}")
    print(f"Edges: {list(G.edges())}")
    print(f"Degrees: {dict(G.degree())}")
    print()
    
    # Get personas
    personas = PERSONA_SETS[persona_type]
    
    print("Agent personas:")
    for i in range(n_agents):
        persona = personas[i % len(personas)]
        print(f"  Agent {i}: {persona[:50]}...")
    print()
    
    # Generate first-round opinions
    question = TOPICS[topic]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    opinions = []
    
    print(f"Question: {question}")
    print()
    
    for i in range(n_agents):
        persona = personas[i % len(personas)]
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": f"{persona} Answer in 1-2 sentences."},
                {"role": "user", "content": f"Question: {question}"}
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
                        opinion = data["choices"][0]["message"]["content"].strip()
                        opinions.append(opinion)
                        print(f"Agent {i}: {opinion}")
                    else:
                        print(f"Agent {i}: ERROR {resp.status}")
                        opinions.append("")
        except Exception as e:
            print(f"Agent {i}: EXCEPTION {e}")
            opinions.append("")
        
        await asyncio.sleep(0.05)
    
    # Calculate consensus
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n" + "="*60)
    print("CONSENSUS ANALYSIS")
    print("="*60)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    vectors = vectorizer.fit_transform(opinions)
    sim_matrix = cosine_similarity(vectors)
    
    print("\nSimilarity matrix (first 5x5):")
    print(np.round(sim_matrix[:5, :5], 3))
    
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    consensus = float(np.mean(upper_triangle))
    
    print(f"\nConsensus score: {consensus:.3f}")
    print(f"Min similarity: {np.min(upper_triangle):.3f}")
    print(f"Max similarity: {np.max(upper_triangle):.3f}")
    
    unique_opinions = len(set(opinions))
    print(f"\nUnique opinions: {unique_opinions}/{len(opinions)}")
    
    if consensus >= 0.99:
        print("\n⚠️  PERFECT CONSENSUS DETECTED")
        print("Sample of opinions:")
        for i, op in enumerate(opinions[:3]):
            print(f"  {i}: {op}")

if __name__ == "__main__":
    asyncio.run(debug_specific_case())
