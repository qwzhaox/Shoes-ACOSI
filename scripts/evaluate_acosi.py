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
args = parser.parse_args()


class Evaluation:
    def __init__(self, process_func):
        self.pred_outpus = process_func(
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

    def calc_scores(self):
        n_tp, n_true, n_pred = 0, 0, 0

        for i in range(len(self.pred_outpus)):
            n_true += len(self.true_outputs[i])
            n_pred += len(self.pred_outpus[i])

            for t in self.pred_outpus[i]:
                if t in self.true_outputs[i]:
                    n_tp += 1

        self.precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        self.recall = float(n_tp) / float(n_true) if n_true != 0 else 0
        self.f1 = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if self.precision != 0 or self.recall != 0
            else 0
        )

    def get_scores(self):
        self.calc_scores()
        scores = {
            "precision": self.precision * 100,
            "recall": self.recall * 100,
            "f1": self.f1 * 100,
        }

        scores["reviews"] = []

        for i in range(len(self.pred_outpus)):
            scores["reviews"].append(
                {
                    "idx": i,
                    "review": self.reviews[i],
                    "pred": self.pred_outpus[i],
                    "true": self.true_outputs[i],
                }
            )

        return scores


evaluate_mvp_outputs = Evaluation(get_mvp_output)
scores = evaluate_mvp_outputs.get_scores()

with open(args.output_file, "w") as file:
    json.dump(scores, file, indent=4)
