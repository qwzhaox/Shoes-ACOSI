#!/bin/bash

python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/main_dataset/train.txt --test data/main_dataset/test.txt --val data/main_dataset/dev.txt