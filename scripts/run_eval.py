import os
import pathlib


pkl_path = pathlib.Path("../EECS595Project/data/model_output/meta-llama/")
pkl_files = list(pkl_path.rglob("*.pkl"))

dataset_path = "../EECS595Project/data/"
output_path = "../EECS595Project/data/eval_output/meta-llama/"

for pkl_file in pkl_files:
    pkl_file = str(pkl_file)
    fn_list = pkl_file.split('/')

    model = fn_list[5]
    task = fn_list[6]
    dataset = fn_list[7]

    if task == "acos-extend":
        continue

    dataset_file = dataset_path
    if task == "acos-extract":
        dataset_file += "acos/"
    elif task == "acosi-extract":
        dataset_file += "acosi/"
    dataset_file += dataset
    dataset_file += "/toy.txt"
    print("dataset file: " + dataset_file)
    
    output_path = output_path + model + "/" + task + "/" + dataset
    output_file = output_path + "/score.json"
    print("output file: " + output_file)

    os.system("mkdir -p " + output_path + " && touch score.json")
    os.system("python3 scripts/evaluate.py --dataset_file=" + dataset_file + " --pkl_file=" + pkl_file + " --output_file=" + output_file + " -llm")
   
    dataset_path = "../EECS595Project/data/"
    output_path = "../EECS595Project/data/eval_output/meta-llama/"
