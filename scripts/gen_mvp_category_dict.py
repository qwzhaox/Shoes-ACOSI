import json
import argparse
from ast import literal_eval
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir", type=str, default="data/main_dataset/", help="dataset directory"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="data/mvp_dataset/category_dict.json",
    help="output directory",
)

args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
train_file = dataset_dir / "train.txt"
test_file = dataset_dir / "test.txt"
dev_file = dataset_dir / "dev.txt"


def get_acosi_categories():
    unique_categories = set()

    def add_categories(file_path):
        with open(file_path, "r") as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line by '####' to separate the review text from annotations
                parts = line.split("####")
                if len(parts) == 2:
                    # Extract the annotations part and split it by ','
                    annotations = literal_eval(parts[1].strip())
                    # Iterate through each annotation and extract the category
                    for annotation in annotations:
                        category = annotation[1].strip()
                        unique_categories.add(category)

    add_categories(train_file)
    add_categories(test_file)
    add_categories(dev_file)
    # Convert the set of unique categories to a sorted list
    category_list = sorted(list(unique_categories))
    return category_list


acosi_categories = get_acosi_categories()
mvp_category_dict = {}

for category in acosi_categories:
    mvp_version = (
        category.replace("#", " ").replace("/", "_").replace("\\_", "_").lower()
    )
    mvp_category_dict[mvp_version] = category

with open(args.output_file, "w") as file:
    json.dump(mvp_category_dict, file, indent=4)
