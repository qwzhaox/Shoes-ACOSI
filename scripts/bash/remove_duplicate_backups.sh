#!/bin/bash

# Define paths
model_dir="$1"
output_base_path="eval_output/$model_dir/"

# Find all .json files
score_files=$(find "$output_base_path" -wholename "*/score.json")

# Loop through each .json file
for score_file in $score_files; do
    # Find all backup files
    backup_files=$(find "$output_base_path" -wholename "$score_file.*.bak")

    # Loop through each backup file
    for backup_file in $backup_files; do
        # Compare the backup file with the main score file
        if cmp -s "$score_file" "$backup_file"; then
            # Remove the backup file
            rm "$backup_file"
        fi
    done
done