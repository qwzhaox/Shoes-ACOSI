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

gen_scl_nat_task_dir="../GEN_SCL_NAT_acosi/data/${task}_shoes_data"
task_dir="data/gen_scl_nat_dataset/${task}_dataset/shoes"

if [ ! -d $gen_scl_nat_task_dir ]; then
    mkdir -p $gen_scl_nat_task_dir
fi

cp $task_dir/train.txt $gen_scl_nat_task_dir/train.txt
cp $task_dir/test.txt $gen_scl_nat_task_dir/test.txt
cp $task_dir/dev.txt $gen_scl_nat_task_dir/dev.txt