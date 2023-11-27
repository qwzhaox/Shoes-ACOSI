import json
import argparse
from mvp.mvp_evaluate import get_mvp_output
from EECS595Project.scripts.evaluate.llm_evaluate import get_llm_output

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
parser.add_argument("-llm", "--llm_output", action="store_true")
args = parser.parse_args()


# NOTE: Assumes comparison with the main test dataset; if this is not possible, adjust the dataset file correspondingly

ASPECT_IDX = 0
CATEGORY_IDX = 1
SENTIMENT_IDX = 2
OPINION_IDX = 3
IMPLICIT_IND_IDX = 4

IDX_LIST = [ASPECT_IDX, CATEGORY_IDX, SENTIMENT_IDX, OPINION_IDX, IMPLICIT_IND_IDX]
TERM_LIST = ["aspect", "category", "sentiment", "opinion", "implicit_indicator"]


def indexify(review, span):
    orig_array = review.split(" ")
    span_array = span.split(" ")
    start_idx = -1
    while orig_array[start_idx] != span_array[0]:
        start_idx += 1
        if start_idx >= len(orig_array):
            return [-1]

    end_idx = start_idx + len(span_array)
    num_span_array = [i for i in range(start_idx, end_idx)]
    return tuple(num_span_array)


def get_precision_recall_fl_IoU(pred_outputs, true_outputs):
    n_tp, n_fp, n_fn, n_union = 0, 0, 0, 0
    for i in range(len(pred_outputs)):
        pred_set = set(pred_outputs[i])
        true_set = set(true_outputs[i])

        n_tp += len(pred_set & true_set)
        n_fp += len(pred_set - true_set)
        n_fn += len(true_set - pred_set)
        n_union += len(pred_set | true_set)

    precision = float(n_tp) / (n_tp + n_fp) if n_tp + n_fp != 0 else 0
    recall = float(n_tp) / (n_tp + n_fn) if n_tp + n_fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    IoU = float(n_tp) / n_union if n_union != 0 else 0
    return precision, recall, f1, IoU


class Evaluator:
    def __init__(self, process_func):
        self.pred_outputs = process_func(
            pkl_file=args.pkl_file, category_file=args.category_file
        )

        with open(args.dataset_file, "r") as file:
            dataset = file.readlines()

        self.reviews = [x.split("####")[0] for x in dataset]

        self.raw_true_outputs = [eval(x.split("####")[1]) for x in dataset]
        self.true_outputs = []
        for output in self.raw_true_outputs:
            self.true_outputs.append([tuple(quint) for quint in output])

        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.IoU = 0

        self.tuple_len = len(self.true_outputs[0][0])
        self.partial_precision = [0] * self.tuple_len
        self.partial_recall = [0] * self.tuple_len

    def calc_exact_scores(self):
        self.precision, self.recall, self.f1, self.IoU = get_precision_recall_fl_IoU(
            self.pred_outputs, self.true_outputs
        )

    def calc_partial_scores(self):
        for idx in IDX_LIST[: self.tuple_len]:
            if idx == ASPECT_IDX or idx == OPINION_IDX:
                pred_outputs = [
                    indexify(x, y[idx]) for x, y in zip(self.reviews, self.pred_outputs)
                ]
                true_outputs = [
                    indexify(x, y[idx]) for x, y in zip(self.reviews, self.true_outputs)
                ]

            else:
                pred_outputs = [x[idx] for x in self.pred_outputs]
                true_outputs = [x[idx] for x in self.true_outputs]

            (
                self.partial_precision[idx],
                self.partial_recall[idx],
                self.partial_f1[idx],
                self.partial_IoU[idx],
            ) = get_precision_recall_fl_IoU(pred_outputs, true_outputs)

    def get_scores(self):
        self.calc_exact_scores()
        scores = {}
        scores["exact precision"] = self.precision
        scores["exact recall"] = self.recall
        scores["exact f1-score"] = self.f1
        scores["exact IoU"] = self.IoU

        for term in TERM_LIST[: self.tuple_len]:
            scores[f"{term} precision"] = self.partial_precision[IDX_LIST.index(term)]
            scores[f"{term} recall"] = self.partial_recall[IDX_LIST.index(term)]
            scores[f"{term} f1-score"] = self.partial_f1[IDX_LIST.index(term)]
            scores[f"{term} IoU"] = self.partial_IoU[IDX_LIST.index(term)]

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
    evaluate_mvp_outputs = Evaluator(get_mvp_output)
    scores = evaluate_mvp_outputs.get_scores()
elif args.llm_output:
    evaluate_llm_outputs = Evaluator(get_llm_output)
    scores = evaluate_llm_outputs.get_scores()

with open(args.output_file, "w") as file:
    json.dump(scores, file, indent=4)
