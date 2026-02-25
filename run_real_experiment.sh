#!/bin/bash

export OPENAI_API_KEY="$1"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Usage: ./run_real_experiment.sh <OPENAI_API_KEY>"
    echo "Or: export OPENAI_API_KEY=sk-... && python3 real_llm_experiment.py"
    exit 1
fi

echo "Running Agentic Thermodynamics experiment with real LLMs..."
echo "This will cost approximately $1-2 in API credits"
echo ""

python3 real_llm_experiment.py 2>&1 | tee experiment_log.txt

echo ""
echo "Done! Check:"
echo "  - llm_results.json (raw data)"
echo "  - experiment_log.txt (console output)"
