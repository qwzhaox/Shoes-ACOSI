#!/bin/bash

# Check if the correct number of arguments is given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Assign the first and second command line arguments to variables
input_directory="$1"
output_directory="$2"

# Ensure both directory paths end with a '/'
[[ "$input_directory" != */ ]] && input_directory="$input_directory/"
[[ "$output_directory" != */ ]] && output_directory="$output_directory/"

# Ensure output directory exists
mkdir -p "$output_directory"

# Loop through each file in the input directory
for file in "$input_directory"/*
do
    # Check if it's a file and not a directory
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        output_filename=scores.json
        category_dict=category_dict.json
        dataset=""

        if [[ $filename == *"rest"* ]]; then
            output_filename="rest_${output_filename}"
            category_dict="rest_${category_dict}"
            dataset=rest
        elif [[ $filename == *"laptop"* ]]; then
            output_filename="laptop_${output_filename}"
            category_dict="laptop_${category_dict}"
            dataset=laptop
        elif [[ $filename == *"shoes"* ]]; then
            output_filename="shoes_${output_filename}"
            category_dict="shoes_${category_dict}"
            dataset=shoes
        else
            exit 1
        fi

        if [[ $filename == *"unified"* ]]; then
            output_filename="unified_${output_filename}"
        fi

        # Process the file (e.g., copy to output directory)
        python3 scripts/eval/evaluate.py --dataset_file=data/acos_dataset/$dataset/test.txt --pkl_file="$file" --output_file="${output_directory}${output_filename}" --task=acos-extract --category_file="data/mvp_dataset/${category_dict}" -mvp
    fi
done
