import json
import argparse

CURATOR_IDX = 0

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="input json file", required=True)
parser.add_argument(
    "-o",
    "--output_file",
    help="output json file",
    required=True,
)

args = parser.parse_args()

with open(args.input_file) as f:
    review_data = json.load(f)

products_dict = {}

num_reviews = len(review_data)

for r in review_data:
    if not r["annotations"] or not r["annotations"][CURATOR_IDX]["annotation"]:
        num_reviews -= 1
        continue

    review_metrics = {}
    review_metrics["review"] = r["review"]
    review_metrics["p_name"] = r["p_name"]

    review_metrics["annot1"] = r["annotations"][CURATOR_IDX]["annotation"]

    products_dict.setdefault(r["p_name"], []).append(review_metrics)

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

TRAIN_PERCENT = 0.83
VALIDATION_PERCENT = 0.08
TEST_PERCENT = 0.09

GOAL = "goal"
ACTUAL = "actual"

ttv = [TRAIN, VALIDATION, TEST]

split_dict = {TRAIN: [], VALIDATION: [], TEST: []}

totals = {
    "num_reviews": num_reviews,
    TRAIN: {
        GOAL: num_reviews * TRAIN_PERCENT,
        ACTUAL: 0,
    },
    VALIDATION: {
        GOAL: num_reviews * VALIDATION_PERCENT,
        ACTUAL: 0,
    },
    TEST: {
        GOAL: num_reviews * TEST_PERCENT,
        ACTUAL: 0,
    },
}

ttv_idx = 0
for product in products_dict.keys():
    if len(ttv) == 0:
        split_dict[TRAIN].extend(products_dict[product])
        totals[TRAIN][ACTUAL] += len(products_dict[product])

    idx = ttv_idx % len(ttv)

    split_dict[ttv[idx]].extend(products_dict[product])
    totals[ttv[idx]][ACTUAL] += len(products_dict[product])

    if totals[ttv[idx]][ACTUAL] >= totals[ttv[idx]][GOAL]:
        ttv.pop(idx)
    else:
        ttv_idx += 1

split_dict["stats"] = totals

with open(args.output_file, "w") as f:
    json.dump(split_dict, f, indent=4)


f.close()
