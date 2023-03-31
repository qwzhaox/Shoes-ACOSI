import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    help="input json file",
    required=True
)
parser.add_argument(
    "--output_file",
    help="output json file",
    required=True,
)

args = parser.parse_args()

with open(args.input_file) as f:
    review_data = json.load(f)

products_dict = {}

for r in review_data:
    review_metrics = {}
    review_metrics["review"] = r["review"]
    review_metrics["p_name"] = r["p_name"]

    # name1 = r["annotations"][0]["metadata"]["name"]
    # name2 = r["annotations"][1]["metadata"]["name"]
    # review["annotator_ids"] = [name1, name2]

    review_metrics["annot1"] = r["annotations"][0]["annotation"]
    review_metrics["annot2"] = r["annotations"][1]["annotation"]

    products_dict.setdefault(r["p_name"], []).append(review_metrics)

num_reviews = len(review_data)

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

TRAIN_PERCENT = 0.65
VALIDATION_PERCENT = 0.20
TEST_PERCENT = 0.15

GOAL = "goal"
ACTUAL = "actual"

ttv = [TRAIN, VALIDATION, TEST]

split_dict = {
    TRAIN: {},
    VALIDATION: {},
    TEST: {}
}

totals = {
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
    }
}

ttv_idx = 0
for product in products_dict.keys():
    if (len(ttv) == 0):
        split_dict[TRAIN][product] = products_dict[product]
        totals[TRAIN][ACTUAL] += len(products_dict[product])

    idx = ttv_idx % len(ttv)

    split_dict[ttv[idx]][product] = products_dict[product]
    totals[ttv[idx]][ACTUAL] += len(products_dict[product])

    if totals[ttv[idx]][ACTUAL] >= totals[ttv[idx]][GOAL]:
        ttv.pop(idx)
    else:
        ttv_idx += 1

split_dict["stats"] = totals

with open(args.output_file, "w") as f:
    json.dump(split_dict, f)


f.close()
