#!/usr/bin/env python3
"""
Overnight experiment runner
"""
import os
import asyncio
import json
import sys
sys.path.insert(0, '/Users/jacobcrainic/projects/agentic-thermodynamics')

import extended_experiment as ee

async def main():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: No API key found")
        return
    
    print("="*60)
    print("144-TRIAL OVERNIGHT EXPERIMENT")
    print("="*60)
    print("Temperature: 1.0")
    print("Rate limit delay: 1.0s")
    print("Failed trials excluded from analysis")
    print("="*60)
    print()
    
    results = await ee.run_extended_experiment(api_key, quick_test=False)
    
    # Save results
    output_file = '/Users/jacobcrainic/projects/agentic-thermodynamics/overnight_results.json'
    with open(output_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    print()
    print("="*60)
    print("OVERNIGHT RUN COMPLETE")
    print("="*60)
    print(f"Valid trials: {len(results)}")
    
    if len(results) >= 50:
        comparisons = ee.compare_predictors(results)
        print()
        print("Predictor comparison:")
        for metric, stats in sorted(comparisons.items(), key=lambda x: -abs(x[1]['spearman_r'])):
            print(f"  {stats['name']:25s}: r = {stats['spearman_r']:6.3f}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
