#!/bin/bash

python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/mvp_dataset/train.txt --test data/mvp_dataset/test.txt --val data/mvp_dataset/dev.txt -mvp "$@"