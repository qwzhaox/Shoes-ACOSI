import json
import argparse
from mvp.mvp_evaluate_acosi import get_mvp_output

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset_file", type=str, default="data/main_dataset/test.txt"
)
parser.add_argument(
    "-p",
    "--pkl_file",
    type=str,
    default="data/mvp_dataset/result_cd_acosi_shoes_path5_beam1.pickle",
)
parser.add_argument(
    "-c", "--category_file", type=str, default="data/mvp_dataset/category_dict.json"
)
parser.add_argument(
    "-o", "--output_file", type=str, default="data/mvp_dataset/scores.json"
)
parser.add_argument("-mvp", "--mvp_output", action="store_true")
args = parser.parse_args()


# NOTE: Assumes comparison with the main test dataset; if this is not possible, adjust the dataset file correspondingly


class Evaluation:
    def __init__(self, process_func):
        self.pred_outputs = process_func(
            pkl_file=args.pkl_file, category_file=args.category_file
        )

        with open(args.dataset_file, "r") as file:
            dataset = file.readlines()

        self.reviews = [x.split("####")[0] for x in dataset]

        self.raw_true_outputs = [eval(x.split("####")[1]) for x in dataset]
        self.__get_true_outputs()

    def __get_true_outputs(self):
        self.true_outputs = []
        for output in self.raw_true_outputs:
            self.true_outputs.append([tuple(quint) for quint in output])

    def calc_exact_scores(self):
        n_tp, n_fp, n_fn = 0, 0, 0

        for i in range(len(self.pred_outputs)):
            pred_set = set(self.pred_outputs[i])
            true_set = set(self.true_outputs[i])

            n_tp += len(pred_set & true_set)
            n_fp += len(pred_set - true_set)
            n_fn += len(true_set - pred_set)

        self.precision = float(n_tp) / (n_tp + n_fp) if n_tp + n_fp != 0 else 0
        self.recall = float(n_tp) / (n_tp + n_fn) if n_tp + n_fn != 0 else 0
        self.f1 = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if self.precision + self.recall != 0
            else 0
        )

    def calc_partial_scores(self):
        pass

    def get_scores(self):
        self.calc_exact_scores()
        scores = {
            "precision": self.precision * 100,
            "recall": self.recall * 100,
            "f1": self.f1 * 100,
        }

        scores["reviews"] = []

        for i in range(len(self.pred_outputs)):
            scores["reviews"].append(
                {
                    "idx": i,
                    "review": self.reviews[i],
                    "pred": self.pred_outputs[i],
                    "true": self.true_outputs[i],
                }
            )

        return scores


scores = {}

if args.mvp_output:
    evaluate_mvp_outputs = Evaluation(get_mvp_output)
    scores = evaluate_mvp_outputs.get_scores()

with open(args.output_file, "w") as file:
    json.dump(scores, file, indent=4)
