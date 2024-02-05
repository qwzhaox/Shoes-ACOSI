#!/bin/bash

# Path to the JSON file containing the score types and scores
JSON_FILE="charts/scores.json"

# Python script to be executed
PYTHON_SCRIPT="scripts/eval/create_charts.py"

# Function to run the Python script with given score_type and score
run_python_script() {
    local score_type="$1"
    local score="$2"
    echo "Running with score_type: $score_type and score: $score"
    python $PYTHON_SCRIPT -st "$score_type" -s "$score" -i
}

# Iterate over each score_type
while IFS= read -r score_type; do
    # Use jq to extract scores as a raw string separated by newline, then iterate
    jq -r --arg st "$score_type" '.[$st][]' $JSON_FILE | while IFS= read -r score; do
        run_python_script "$score_type" "$score"
    done
done < <(jq -r 'keys[]' $JSON_FILE)