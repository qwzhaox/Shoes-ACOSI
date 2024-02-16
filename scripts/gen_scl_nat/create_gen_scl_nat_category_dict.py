import json
import argparse
from pathlib import Path
from sys import path
path.insert(1, '.')
from scripts.dataprocess.get_categories import get_categories

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir", type=str, default="data/acosi_dataset/shoes/", help="dataset directory"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="data/gen_scl_nat_dataset/shoes_category_dict.json",
    help="output directory",
)

args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
train_file = dataset_dir / "train.txt"
test_file = dataset_dir / "test.txt"
dev_file = dataset_dir / "dev.txt"

cat_dict_filepath = Path(args.output_file)
cat_map_filename = cat_dict_filepath.stem[:cat_dict_filepath.stem.find('_dict')] if cat_dict_filepath.stem.endswith('_dict') else cat_dict_filepath.stem
mappings_filepath = cat_dict_filepath.parent / f"{cat_map_filename}_mappings.json"

categories = get_categories()
gen_scl_nat_category_dict = {}
gen_scl_nat_category_mappings = {}

parent_mapping_key = f"{dataset_dir.name}_parent_mapping"
full_mapping_key = f"{dataset_dir.name}_full_mapping"
gen_scl_nat_category_mappings[parent_mapping_key] = []
gen_scl_nat_category_mappings[full_mapping_key] = []

parent_mapping = set()
full_mapping = set()

for category in categories:
    cat_parts = category.split("#")
    gen_scl_nat_parent = f"the {cat_parts[0]}"
    gen_scl_nat_parent = gen_scl_nat_parent.replace("contextofuse", "context of use")
    gen_scl_nat_parent = gen_scl_nat_parent.replace("general", "shoes overall")
    gen_scl_nat_parent = gen_scl_nat_parent.replace("misc", "miscellaneous overall")

    parent_mapping.add((gen_scl_nat_parent.lower(), cat_parts[0].upper()))

    if len(cat_parts) > 1:
        if cat_parts[1] == "general":
            gen_scl_nat_full = f"the {cat_parts[0]} overall"
        elif cat_parts[1] == "misc":
            gen_scl_nat_full = f"the miscellaneous {cat_parts[0]}"
        else:
            gen_scl_nat_full = f"the {cat_parts[1]}"
    else:
        gen_scl_nat_full = gen_scl_nat_parent
    
    gen_scl_nat_full = gen_scl_nat_full.replace("\\_", "_")
    gen_scl_nat_full = gen_scl_nat_full.replace("the shoe component", "one of the parts that form the shoe")
    gen_scl_nat_full = gen_scl_nat_full.replace("place", "user's environment")
    gen_scl_nat_full = gen_scl_nat_full.replace("purchase_context", "context of the purchase")
    gen_scl_nat_full = gen_scl_nat_full.replace("review_temporality", "user's time of use")

    full_mapping.add((gen_scl_nat_full.lower(), category.upper()))
    gen_scl_nat_category_dict[gen_scl_nat_full] = category.upper()

gen_scl_nat_category_mappings[parent_mapping_key] = list(parent_mapping)
gen_scl_nat_category_mappings[full_mapping_key] = list(full_mapping)

with open(cat_dict_filepath, "w") as file:
    json.dump(gen_scl_nat_category_dict, file, indent=4)

with open(mappings_filepath, "w") as file:
    json.dump(gen_scl_nat_category_mappings, file, indent=4)
