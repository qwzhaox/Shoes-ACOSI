#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir="$1"

# Define paths
pkl_path="../LLM-acosi/model_output/$model_dir/"
dataset_path="../LLM-acosi/data/"
#output_base_path="../LLM-acosi/eval_output/$model_dir/"
output_base_path="eval_output/$model_dir/"

# Find all .pkl files
model_output_files=$(find "$pkl_path" -wholename "*/$model_dir/*.pkl")

# Loop through each .pkl file
for model_output_file in $model_output_files; do
    echo "model output file: $model_output_file"

    # Extract model, task, and dataset from the file path
    IFS='/' read -r -a path_array <<< "$model_output_file"
    model=${path_array[4]}
    task=${path_array[5]}
    dataset=${path_array[6]}

    # Skip if task is "acos-extend"
    if [ "$task" == "acos-extend" ] && ( [ "$dataset" == "rest" ] || [ "$dataset" == "laptop" ] ); then
        continue
    fi


    # Construct dataset file path
    dataset_file="$dataset_path"
    if [ "$task" == "acos-extract" ]; then
        dataset_file+="acos_dataset/"
    elif [[ "$task" == "acosi-extract"  || "$task" == "acos-extend" ]]; then
        dataset_file+="acosi_dataset/"
    else
        echo "Invalid task: $task"
        continue
    fi
    dataset_file+="$dataset/test.txt"
    category_dict="data/llm_dataset/${dataset}_category_dict.json"
    echo "dataset file: $dataset_file"

    # Construct output file path
    output_path="$output_base_path$model/$task/$dataset"
    output_file="$output_path/score.json"
    echo "output file: $output_file"

    most_recent_old=""

    # Create directories and empty score.json file
    if [ -f "$output_file" ]; then
        back_int=0
        while [ -f "$output_file.$back_int.bak" ]; do
            back_int=$((back_int + 1))
        done
        mv "$output_file" "$output_file.$back_int.bak"
        most_recent_old="$output_file.$back_int.bak"
    fi
    mkdir -p "$output_path" && touch "$output_file"

    # Run evaluation script
    if [ "$task" == "acos-extend" ]; then
        python3 scripts/eval/evaluate.py --dataset_file="$dataset_file" --model_output_file="$model_output_file" --category_file="$category_dict" --output_file="$output_file" --task="acos-extend" -llm
    elif [ "$task" == "acos-extract" ]; then
        python3 scripts/eval/evaluate.py --dataset_file="$dataset_file" --model_output_file="$model_output_file" --category_file="$category_dict" --output_file="$output_file" --task="acos-extract" -llm
    else
        python3 scripts/eval/evaluate.py --dataset_file="$dataset_file" --model_output_file="$model_output_file" --category_file="$category_dict" --output_file="$output_file" --task="acosi-extract" -llm
    fi
done