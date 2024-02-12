import json
import argparse
from pathlib import Path
from sys import path
path.insert(1, './scripts/dataprocess/')
from get_categories import get_categories

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir", type=str, default="data/acosi_dataset/shoes/", help="dataset directory"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="data/llm_dataset/shoes_category_dict.json",
    help="output directory",
)

args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
train_file = dataset_dir / "train.txt"
test_file = dataset_dir / "test.txt"
dev_file = dataset_dir / "dev.txt"

categories = get_categories()
llm_category_dict = {}

for category in categories:
    llm_version = (
        category.replace("#", " ").replace("\\_", "_").lower()
    )
    llm_category_dict[llm_version] = category

with open(args.output_file, "w") as file:
    json.dump(llm_category_dict, file, indent=4)
