#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir="$1"

for selection_method in random tf-idf; do
    for k in 5 10; do
        bash scripts/bash/run_llm_eval.sh $model_dir-$selection_method-$k
        bash scripts/bash/run_llm_eval.sh $model_dir-$selection_method-$k-906
        for annot_source in mvp gen-scl-nat; do
            if [ $annot_source = "mvp" ]; then
                for seed in 5 10 15 20 25; do
                    bash scripts/bash/run_llm_eval.sh $model_dir-$selection_method-$k-$annot_source-seed-$seed
                done
            else
                bash scripts/bash/run_llm_eval.sh $model_dir-$selection_method-$k-$annot_source
            fi
        done
    done
done

bash scripts/bash/remove_duplicate_backups.sh

