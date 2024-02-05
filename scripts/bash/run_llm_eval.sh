#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir="$1"

# Define paths
pkl_path="../EECS595Project/data/model_output/$model_dir/"
dataset_path="../EECS595Project/data/"
#output_base_path="../EECS595Project/data/eval_output/$model_dir/"
output_base_path="data/llm_output/$model_dir/"

# Find all .pkl files
pkl_files=$(find "$pkl_path" -wholename "*/$model_dir/*.pkl")

# Loop through each .pkl file
for pkl_file in $pkl_files; do
    # Extract model, task, and dataset from the file path
    IFS='/' read -r -a path_array <<< "$pkl_file"
    model=${path_array[5]}
    task=${path_array[6]}
    dataset=${path_array[7]}

    # Skip if task is "acos-extend"
    if [ "$task" == "acos-extend" ] && ( [ "$dataset" == "rest" ] || [ "$dataset" == "laptop" ] ); then
        continue
    fi


    # Construct dataset file path
    dataset_file="$dataset_path"
    if [ "$task" == "acos-extract" ]; then
        dataset_file+="acos/"
    elif [[ "$task" == "acosi-extract"  || "$task" == "acos-extend" ]]; then
        dataset_file+="acosi/"
    else
        echo "Invalid task: $task"
        continue
    fi
    dataset_file+="$dataset/toy.txt"
    echo "dataset file: $dataset_file"

    # Construct output file path
    output_path="$output_base_path$model/$task/$dataset"
    output_file="$output_path/score.json"
    echo "output file: $output_file"

    # Create directories and empty score.json file
    mkdir -p "$output_path" && touch "$output_file"

    # Run evaluation script
    if [ "$task" == "acos-extend" ]; then
        python3 scripts/eval/evaluate.py --dataset_file="$dataset_file" --pkl_file="$pkl_file" --output_file="$output_file" --task="acos-extend" -llm
    elif [ "$task" == "acos-extract" ]; then
        python3 scripts/eval/evaluate.py --dataset_file="$dataset_file" --pkl_file="$pkl_file" --output_file="$output_file" --task="acos-extract" -llm
    else
        python3 scripts/eval/evaluate.py --dataset_file="$dataset_file" --pkl_file="$pkl_file" --output_file="$output_file" --task="acosi-extract" -llm
    fi
done
