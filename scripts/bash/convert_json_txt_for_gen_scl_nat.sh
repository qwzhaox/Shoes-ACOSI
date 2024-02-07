#!/bin/bash

python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/gen_scl_nat_dataset/acosi_dataset/shoes/train.txt --test data/gen_scl_nat_dataset/acosi_dataset/shoes/test.txt --val data/gen_scl_nat_dataset/acosi_dataset/shoes/dev.txt -gen_scl_nat -acosi
python3 scripts/dataprocess/convert_json_txt.py --input_file data/splits.json --train data/gen_scl_nat_dataset/acos_dataset/shoes/train.txt --test data/gen_scl_nat_dataset/acos_dataset/shoes/test.txt --val data/gen_scl_nat_dataset/acos_dataset/shoes/dev.txt -gen_scl_nat -acos