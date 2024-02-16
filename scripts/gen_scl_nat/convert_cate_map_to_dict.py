import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cate_map", type=str, default="data/gen_scl_nat_dataset/laptop_category_mappings.json", help="category mappings file"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="data/gen_scl_nat_dataset/laptop_category_dict.json",
    help="output file",
)

args = parser.parse_args()

if "shoes" in args.cate_map:
    dataset_name = "shoes"
elif "rest" in args.cate_map:
    dataset_name = "restaurant"
elif "laptop" in args.cate_map:
    dataset_name = "laptop"
else:
    raise ValueError("Unknown dataset")

with open(args.cate_map, "rb") as f:
    cate_map = json.load(f)

gen_scl_nat_category_dict = {}

full_mapping = cate_map[f"{dataset_name}_full_mapping"]
for key, value in full_mapping:
    gen_scl_nat_category_dict[key] = value

with open(args.output_file, "w") as f:
    json.dump(gen_scl_nat_category_dict, f, indent=4)

