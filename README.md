# Agentic Thermodynamics: Spectral Predictors of Consensus Collapse

Toy experiment validating the hypothesis that spectral graph properties predict consensus collapse in multi-agent LLM swarms.

## Key Finding

Spectral gap (λ₂ of graph Laplacian) predicts consensus collapse with **91.3% accuracy**.

## Results Summary

```
Total trials: 126
Overall convergence rate: 89.7%
Spectral gap threshold: 0.191
Predictive accuracy: 91.3%
Spearman correlation: r=0.449, p=1.29e-07

By Graph Type:
  random      :  92.9% converged, avg λ₂=0.856
  scale_free  :  76.2% converged, avg λ₂=0.522
  complete    : 100.0% converged, avg λ₂=16.000
```

## Files

- `consensus_swarm.py` - Main simulation code
- `visualize.py` - Figure generation
- `results.json` - Experimental results
- `figure*.pdf/png` - Generated figures

## Running

```bash
python3 consensus_swarm.py
python3 visualize.py
```

## Literature Gap

- Yazici et al. (ICASSP 2026): Uses DeGroot framework, observes exponential decay, no predictive theory
- Kaushal & Singh (2025): Shows "more voices hurt" empirically, no theoretical explanation
- **This work**: Spectral predictor of consensus collapse with phase transition
