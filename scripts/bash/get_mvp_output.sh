#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <shoes_task>"
    exit 1
fi

copy_files() {
    local task=$1
    local shoes_dataset=$2

    if [[ "$task" != "acos" && "$task" != "acosi" ]]; then
        echo "Invalid task: $task. Please specify 'acos' or 'acosi'."
        return 1
    fi

    for file in ../multi-view-prompting-acosi/outputs/${task}/*/top5_seed*_post_data1.0/result_cd_${task}_*_path5_beam1.pickle; do
        # Extract the dataset and seed values from the file path
        dataset=$(echo $file | sed -E "s/.*${task}\/(.*)\/top5_seed.*_post_data.*/\1/")
        seed=$(echo $file | sed -E 's/.*top5_seed([0-9]+)_post_data.*/\1/')

        if [ $shoes_dataset = "original_shoes" ] && [ $task = "acos" ] && [ $dataset = "shoes" ]; then
            continue
        fi

        # Construct the destination directory and ensure it exists
        dest_dir="model_output/mvp-seed-${seed}/${task}-extract/${shoes_dataset}"
        mkdir -p "$dest_dir"

        # Copy the file to the destination directory
        cp "$file" "${dest_dir}/result_cd_${task}_${dataset}_path5_beam1.pickle"
    done

    # Iterate over all matching files for Unified
    for file in ../multi-view-prompting-acosi/outputs/unified/top5_seed*/result_cd_${task}_*_path5_beam1.pickle; do
        # Extract the dataset and seed values from the file path
        dataset=$(echo $file | sed -E "s/.*unified\/top5_seed([0-9]+)\/result_cd_${task}_(.*)_path5_beam1.pickle/\2/")
        seed=$(echo $file | sed -E "s/.*top5_seed([0-9]+)\/result_cd_${task}.*_path5_beam1.pickle/\1/")

        if [ $shoes_dataset = "original_shoes" ] && [ $task = "acos" ] && [ $dataset = "shoes" ]; then
            continue
        fi

        # Construct the destination directory and ensure it exists
        dest_dir="model_output/mvp-seed-${seed}/${task}-extract/${shoes_dataset}"
        mkdir -p "$dest_dir"

        # Copy the file to the destination directory
        cp "$file" "${dest_dir}/unified_result_cd_${task}_${dataset}_path5_beam1.pickle"
    done
}

shoes_task="$1"
shoes_dataset=""

if [ $shoes_task = "acos" ]; then
    shoes_dataset="shoes_acos"
elif [ $shoes_task = "acosi" ]; then
    shoes_dataset="original_shoes"
    copy_files "acosi" $shoes_dataset
else
    echo "dataset_task not found"
    exit 1
fi

copy_files "acos" $shoes_dataset