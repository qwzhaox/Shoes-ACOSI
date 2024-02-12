#!/bin/bash

python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/llm_dataset/acosi_dataset/shoes/train.txt --test data/llm_dataset/acosi_dataset/shoes/test.txt --val data/llm_dataset/acosi_dataset/shoes/dev.txt -llm -acosi
python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/llm_dataset/acos_dataset/shoes/train.txt --test data/llm_dataset/acos_dataset/shoes/test.txt --val data/llm_dataset/acos_dataset/shoes/dev.txt -llm -acos