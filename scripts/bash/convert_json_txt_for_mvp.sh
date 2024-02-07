#!/bin/bash

python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/mvp_dataset/acosi_dataset/shoes/train.txt --test data/mvp_dataset/acosi_dataset/shoes/test.txt --val data/mvp_dataset/acosi_dataset/shoes/dev.txt -mvp -acosi
python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/mvp_dataset/acos_dataset/shoes/train.txt --test data/mvp_dataset/acos_dataset/shoes/test.txt --val data/mvp_dataset/acos_dataset/shoes/dev.txt -mvp -acos