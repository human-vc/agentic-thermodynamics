"""
Quick test with GPT-3.5-turbo to check if older models show diversity
"""
import os
import asyncio
import aiohttp

async def test_model_diversity(model="gpt-3.5-turbo", n_agents=5):
    """Test if a model produces diverse responses with different personas"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key found")
        return
    
    personas = [
        "You are a progressive environmental activist.",
        "You are a conservative business owner focused on profits.",
        "You are a pragmatic scientist who values data.",
        "You are a skeptical journalist who questions everything.",
        "You are a centrist politician seeking compromise."
    ]
    
    question = "What is the most effective policy to address climate change?"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    opinions = []
    
    print(f"Testing {model} with {n_agents} diverse personas...\n")
    
    for i, persona in enumerate(personas[:n_agents]):
        payload = {
            "model": model,
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
                        print(f"Agent {i+1} ({persona[:40]}...):")
                        print(f"  -> {opinion}\n")
                    else:
                        print(f"Error: {resp.status}")
                        opinions.append("")
        except Exception as e:
            print(f"Exception: {e}")
            opinions.append("")
        
        await asyncio.sleep(0.1)
    
    # Calculate consensus
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        vectors = vectorizer.fit_transform(opinions)
        sim_matrix = cosine_similarity(vectors)
        
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        consensus = float(np.mean(upper_triangle))
        
        print(f"\n{'='*60}")
        print(f"CONSENSUS SCORE: {consensus:.3f}")
        print(f"{'='*60}")
        
        if consensus > 0.9:
            print("⚠️  HIGH consensus - model produces nearly identical responses")
            print("   (RLHF alignment likely forcing uniformity)")
        elif consensus < 0.5:
            print("✓ LOW consensus - model respects persona diversity")
            print("   (Good for spectral predictor experiment)")
        else:
            print("~ MODERATE consensus - some convergence but still diverse")
            
    except Exception as e:
        print(f"Could not calculate consensus: {e}")
    
    return opinions

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-3.5-turbo"
    asyncio.run(test_model_diversity(model))
