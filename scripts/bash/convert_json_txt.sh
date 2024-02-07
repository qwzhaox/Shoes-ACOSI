#!/bin/bash



python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/acosi_dataset/shoes/train.txt --test data/acosi_dataset/shoes/test.txt --val data/acosi_dataset/shoes/dev.txt "$@"