# Agentic Thermodynamics: Spectral Predictors of Consensus Collapse

Validating the hypothesis that spectral graph properties predict consensus collapse in multi-agent LLM swarms.

## Key Finding

Spectral gap (λ₂ of graph Laplacian) predicts consensus collapse with **92.4% accuracy**.

## Quick Start

### Simulated Agents (Free)
```bash
python3 experiment_extended.py
```

### Real LLM Agents (Requires OpenAI API Key)
```bash
export OPENAI_API_KEY="sk-..."
python3 real_llm_experiment.py
```

Or use the helper script:
```bash
./run_real_experiment.sh sk-...
```

**Cost**: ~$1-2 per full run (240 API calls)

## Results Summary (Simulated)

```
Total trials: 251
Overall convergence rate: 84.5%
Spectral gap threshold: 0.303
Predictive accuracy: 92.4%
Spearman correlation: r=0.576, p=1.52e-23

By Graph Type:
  complete    : 100.0% converged, avg λ₂=16.000
  random      :  92.9% converged, avg λ₂=0.856
  scale_free  :  76.2% converged, avg λ₂=0.522
  cycle       :  Collapse at N≥11 when λ₂<0.3

Collapsed vs Converged:
  Avg spectral gap (collapsed):  0.132
  Avg spectral gap (converged):  5.400
  Ratio: 40.98x
```

## Files

- `consensus_swarm.py` - Base simulation
- `experiment_extended.py` - Extended experiments with phase transitions
- `real_llm_experiment.py` - Real LLM agent experiments
- `visualize.py` - Figure generation
- `results*.json` - Experimental data
- `figure*.pdf/png` - Generated figures

## Literature Gap

| Paper | What They Do | Gap You Fill |
|-------|-------------|--------------|
| Yazici et al. (ICASSP 2026) | DeGroot framework, observes exponential decay, notes eigenvalue correlation | No predictive theory for collapse |
| Kaushal & Singh (2025) | Shows empirically that "more voices hurt" | No theoretical explanation |
| **This work** | Spectral predictor with phase transition | Early warning system for consensus collapse |
