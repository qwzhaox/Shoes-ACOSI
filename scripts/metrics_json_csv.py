import argparse
import pandas as pd

# ...
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
    df = pd.read_json(f)

df.to_csv(args.output_file, encoding='utf-8', index=False)
