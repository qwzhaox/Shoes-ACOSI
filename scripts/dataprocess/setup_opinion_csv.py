import pandas as pd
import argparse

from utils import get_dataset, OPINION_IDX, IMPL_EXPL_IDX

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path", "-d", type=str, default="data/acosi_dataset/shoes")
parser.add_argument("--output_csv_file", "-o", type=str, default="data/raw_data/fixed_annot_csv/opinion_span_re-annot.csv")

args = parser.parse_args()

dataset = get_dataset(args.dataset_folder_path)

formatted_dataset = []

for example in dataset:
    review = example.split("####")[0]
    annotation = eval(example.split("####")[1])

    for quint in annotation:
        opinion_span = quint[OPINION_IDX]

        try:
            quint[IMPL_EXPL_IDX] = "n" if quint[IMPL_EXPL_IDX] == "direct" else "y"
            implicit_indicator = quint[IMPL_EXPL_IDX]
            formatted_quint = "\n".join(quint[:-1])
        except:
            implicit_indicator = "n" if opinion_span.lower() != "null" else "y"
            formatted_quint = "\n".join(quint)

        row = {
            "review": review,
            "annotation tuple": formatted_quint,
            "opinion span": opinion_span,
            # "implicit indicator": implicit_indicator,
            # "implicit?": "-",
            "opinion rewrite": "-",
            # "tuple rewrrite": "-",
            "additional notes": "-"
        }

        formatted_dataset.append(row)

df = pd.DataFrame(formatted_dataset)

df.to_csv(args.output_csv_file, index=False)