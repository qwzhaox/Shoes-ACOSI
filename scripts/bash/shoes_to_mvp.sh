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

mvp_task_dir="../multi-view-prompting-acosi/data/$task/shoes"
task_dir="data/mvp_dataset/${task}_dataset/shoes"

if [ ! -d $mvp_task_dir ]; then
    mkdir -p $mvp_task_dir
fi

cp $task_dir/train.txt $mvp_task_dir/train.txt
cp $task_dir/test.txt $mvp_task_dir/test.txt
cp $task_dir/dev.txt $mvp_task_dir/dev.txt