"""
Debug: Check if opinions are actually being generated correctly
"""
import os
import asyncio
import numpy as np
import networkx as nx
import json
import aiohttp

async def debug_opinion_generation():
    """Test if opinions are actually diverse in extended_experiment setup"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key found")
        return
    
    # Exact setup from extended_experiment.py
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
    
    # Test healthcare, homogeneous (the problematic case)
    topic = "healthcare"
    persona_type = "homogeneous"
    n_agents = 5
    
    question = TOPICS[topic]
    personas = PERSONA_SETS[persona_type]
    
    print(f"Testing: {topic}, {persona_type}, N={n_agents}")
    print(f"Question: {question}")
    print(f"Number of unique personas: {len(set(personas))}")
    print(f"Persona list: {personas}")
    print()
    
    # Generate opinions
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    opinions = []
    
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
        
        print(f"Agent {i}: {persona[:60]}...")
        
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
                        print(f"  -> {opinion}")
                    else:
                        print(f"  ERROR: {resp.status}")
                        opinions.append("")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            opinions.append("")
        
        await asyncio.sleep(0.1)
    
    # Calculate consensus
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n" + "="*60)
    print("CONSENSUS ANALYSIS")
    print("="*60)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    vectors = vectorizer.fit_transform(opinions)
    sim_matrix = cosine_similarity(vectors)
    
    print("\nSimilarity matrix:")
    print(np.round(sim_matrix, 3))
    
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    consensus = float(np.mean(upper_triangle))
    
    print(f"\nConsensus score: {consensus:.3f}")
    
    # Check if all opinions are identical
    unique_opinions = len(set(opinions))
    print(f"Unique opinions: {unique_opinions}/{len(opinions)}")
    
    if unique_opinions == 1:
        print("⚠️  ALL OPINIONS ARE IDENTICAL!")
    elif consensus > 0.9:
        print("⚠️  High consensus - opinions very similar")
    else:
        print("✓ Opinions are diverse")

if __name__ == "__main__":
    asyncio.run(debug_opinion_generation())
