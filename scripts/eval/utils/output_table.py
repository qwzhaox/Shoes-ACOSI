import json
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_file", help="input json file", required=True)
parser.add_argument(
    "--output_file",
    help="output json file",
    required=True,
)

args = parser.parse_args()

with open(args.input_file, "r") as f:
    json_data = json.load(f)
    json_data.pop("reviews")

# Create a DataFrame from the JSON data
df = pd.DataFrame(json_data).T * 100
df = df.drop(columns=["global IoU", "avg local IoU"])

# Truncate values to 3-4 significant figures
df = df.round(3)

# Save the DataFrame to a CSV file
print(df)
df.to_csv(args.output_file, sep="\t")

print(f"Data saved to {args.output_file}")
