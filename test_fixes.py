"""
Quick test of fixed code before overnight run
"""
import os
import asyncio
import extended_experiment as ee

async def test_fixes():
    """Run 2 trials to verify fixes work"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("Testing fixed code (2 trials)...")
    print(f"Temperature: 1.0")
    print(f"Rate limit delay: 1.0s")
    print(f"Fallback removed: Yes")
    print()
    
    results = []
    configs = [
        ("climate", "mixed", "complete", 5),
        ("healthcare", "polarized", "cycle", 5)
    ]
    
    for i, (topic, persona, graph, n) in enumerate(configs, 1):
        trial_id = f"test_{topic}_{persona}_{graph}_N{n}"
        print(f"\n[{i}/2] {trial_id}")
        
        config = ee.SwarmConfig(
            n_agents=n, n_rounds=3, graph_type=graph,
            consensus_threshold=0.6, seed=i*100,
            topic=topic, persona_type=persona
        )
        llm_config = ee.LLMConfig(api_key=api_key)
        swarm = ee.ConsensusSwarm(config, llm_config)
        
        result = await swarm.run_full_consensus(ee.TOPICS[topic], trial_id)
        
        if result is None:
            print("  ❌ Trial failed (API errors)")
        else:
            results.append(result)
            print(f"  ✓ Score: {result['final_consensus_score']:.3f}")
            print(f"    Trajectory: {[round(x, 3) for x in result['consensus_trajectory']]}")
            print(f"    Has errors: {result.get('has_errors', False)}")
    
    print(f"\n{'='*60}")
    print(f"Test complete: {len(results)}/2 trials successful")
    print(f"{'='*60}")
    
    if len(results) == 2:
        print("✓ Both trials successful - ready for overnight run")
        return True
    elif len(results) >= 1:
        print("~ Partial success - may need to adjust rate limits")
        return True
    else:
        print("✗ All trials failed - check API key and rate limits")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixes())
    exit(0 if success else 1)
