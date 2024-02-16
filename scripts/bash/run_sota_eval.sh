#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir="$1"

input_directory="model_output/$model_dir"
output_directory="eval_output/$model_dir"

# Ensure input directory exists
if [ ! -d "$input_directory" ]; then
    echo "Input directory does not exist"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$output_directory"

# Loop through each filepath in the input directory
find "$input_directory" -type f | while read filepath; do
    filename=$(basename "$filepath")
    output_filepath=scores.json
    category_dict=category_dict.json
    task=""
    dataset=""
    dataset_file=""

    if [[ $filename == *"rest"* ]]; then
        output_filepath="rest/${output_filepath}"
        category_dict="rest_${category_dict}"
        dataset=rest
    elif [[ $filename == *"laptop"* ]]; then
        output_filepath="laptop/${output_filepath}"
        category_dict="laptop_${category_dict}"
        dataset=laptop
    elif [[ $filename == *"shoes"* ]]; then
        output_filepath="shoes/${output_filepath}"
        category_dict="shoes_${category_dict}"
        dataset=shoes
    else
        exit 1
    fi

    if [[ $filepath == *"acosi"* ]]; then
        dataset_file="data/acosi_dataset/shoes/test.txt"
        task="acosi-extract"
    elif [[ $filepath == *"acos"* ]]; then
        task="acos-extract"
        dataset_file="data/acos_dataset/${dataset}/test.txt"
    else
        exit 1
    fi
    
    output_filepath="${task}/${output_filepath}"

    if [[ $model_dir == "mvp" ]]; then
        if [[ $filename == *"unified"* ]]; then
            output_filepath="unified/${output_filepath}"
        else
            output_filepath="main/${output_filepath}"
        fi
    fi
    output_filepath="${output_directory}/${output_filepath}"

    # Create output directory
    mkdir -p "$(dirname "$output_filepath")"

    python3 scripts/eval/evaluate.py --dataset_file="$dataset_file" --model_output_file="$filepath" --output_file="$output_filepath" --task="$task" --category_file="data/${model_dir}_dataset/${category_dict}" -${model_dir}
done
