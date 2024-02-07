import json
import argparse
from ast import literal_eval
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir", type=str, default="data/acosi_dataset/shoes/", help="dataset directory"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="data/shoes_category_list.json",
    help="output directory",
)

args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
train_file = dataset_dir / "train.txt"
test_file = dataset_dir / "test.txt"
dev_file = dataset_dir / "dev.txt"


def get_categories():
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
                        category = annotation[1].strip().lower()
                        unique_categories.add(category)

    add_categories(train_file)
    add_categories(test_file)
    add_categories(dev_file)
    # Convert the set of unique categories to a sorted list
    categories = sorted(list(unique_categories))
    return categories


categories = get_categories()

with open(args.output_file, "w") as file:
    json.dump(categories, file, indent=4)
