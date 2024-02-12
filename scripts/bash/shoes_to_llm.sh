#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <task>"
    exit 1
fi

task=$1

if [ $task = "acos" ]; then
    echo "acos"
elif [ $task = "acosi" ]; then
    echo "acosi"
else
    echo "task not found"
    exit 1
fi

llm_task_dir="../LLM-acosi/data/${task}_dataset/shoes"
task_dir="data/mvp_dataset/${task}_dataset/shoes"

if [ ! -d $llm_task_dir ]; then
    mkdir -p $llm_task_dir
fi

cp $task_dir/train.txt $llm_task_dir/train.txt
cp $task_dir/test.txt $llm_task_dir/test.txt
cp $task_dir/dev.txt $llm_task_dir/dev.txt