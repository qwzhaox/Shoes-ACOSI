import json
import argparse
from utils.evaluate_utils import (
    ASPECT_IDX,
    OPINION_IDX,
    IDX_LIST,
    TERM_LIST,
    indexify_outputs,
    listify_outputs,
    get_precision_recall_fl_IoU,
)
from utils.mvp_evaluate import get_mvp_output
from utils.llm_evaluate import get_llm_output
from utils.t5_evaluate import get_t5_output

"""
py evaluate.py -t5 -p '../data/t5_output/predictions.pickle' -o '../data/t5_output/scores.json'
"""
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
parser.add_argument("-t5", "--t5_output", action="store_true")
args = parser.parse_args()


class Evaluator:
    def __init__(self, process_func, **kwargs):
        self.pred_outputs = process_func(pkl_file=args.pkl_file, **kwargs)

        with open(args.dataset_file, "r") as file:
            dataset = file.readlines()

        self.reviews = [x.split("####")[0] for x in dataset]
        self.set_true_outputs(dataset)

        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.global_IoU = 0
        self.local_IoU = []
        self.avg_local_IoU = 0

        self.tuple_len = len(self.true_outputs[0][0])
        if len(self.pred_outputs[0][0]) > self.tuple_len:
            new_pred_outputs = []
            for annotation in self.pred_outputs:
                annotation = [quint[: self.tuple_len] for quint in annotation]
                new_pred_outputs.append(annotation)
            self.pred_outputs = new_pred_outputs

        self.partial_precision = [0] * self.tuple_len
        self.partial_recall = [0] * self.tuple_len
        self.partial_f1 = [0] * self.tuple_len
        self.partial_global_IoU = [0] * self.tuple_len
        self.partial_local_IoU = [[]] * self.tuple_len
        self.partial_avg_local_IoU = [0] * self.tuple_len

    def set_true_outputs(self, dataset):
        raw_true_outputs = [eval(x.split("####")[1]) for x in dataset]
        self.true_outputs = []
        for output in raw_true_outputs:
            list_of_tuples = []
            for quint in output:
                if (
                    quint[ASPECT_IDX].lower() == "null"
                    or quint[ASPECT_IDX].lower() == "implicit"
                ):
                    quint[ASPECT_IDX] = "NULL"
                if (
                    quint[OPINION_IDX] == "null"
                    or quint[OPINION_IDX].lower() == "implicit"
                ):
                    quint[OPINION_IDX] = "NULL"
                list_of_tuples.append(tuple(quint))
            self.true_outputs.append(list_of_tuples)

    def calc_exact_scores(self):
        (
            self.precision,
            self.recall,
            self.f1,
            self.global_IoU,
            self.local_IoU,
            self.avg_local_IoU,
        ) = get_precision_recall_fl_IoU(self.pred_outputs, self.true_outputs)

    def calc_partial_scores(self):
        for idx in IDX_LIST[: self.tuple_len]:
            if idx == ASPECT_IDX or idx == OPINION_IDX:
                pred_outputs = indexify_outputs(self.reviews, self.pred_outputs, idx)
                true_outputs = indexify_outputs(self.reviews, self.true_outputs, idx)
            else:
                pred_outputs = listify_outputs(self.pred_outputs, idx)
                true_outputs = listify_outputs(self.true_outputs, idx)
            (
                self.partial_precision[idx],
                self.partial_recall[idx],
                self.partial_f1[idx],
                self.partial_global_IoU[idx],
                self.partial_local_IoU[idx],
                self.partial_avg_local_IoU[idx],
            ) = get_precision_recall_fl_IoU(pred_outputs, true_outputs)

    def get_scores(self):
        self.calc_exact_scores()
        self.calc_partial_scores()
        scores = {}
        scores["exact"] = {}
        scores["exact"]["precision"] = self.precision
        scores["exact"]["recall"] = self.recall
        scores["exact"]["f1-score"] = self.f1
        scores["exact"]["global IoU"] = self.global_IoU
        scores["exact"]["avg local IoU"] = self.avg_local_IoU

        for i, term in enumerate(TERM_LIST[: self.tuple_len]):
            scores[term] = {}
            scores[term]["precision"] = self.partial_precision[i]
            scores[term]["recall"] = self.partial_recall[i]
            scores[term]["f1-score"] = self.partial_f1[i]
            scores[term]["global IoU"] = self.partial_global_IoU[i]
            scores[term]["avg local IoU"] = self.partial_avg_local_IoU[i]

        scores["avg partial"] = {}
        scores["avg partial"]["precision"] = sum(self.partial_precision) / len(
            self.partial_precision
        )
        scores["avg partial"]["recall"] = sum(self.partial_recall) / len(
            self.partial_recall
        )
        scores["avg partial"]["f1-score"] = sum(self.partial_f1) / len(self.partial_f1)
        scores["avg partial"]["global IoU"] = sum(self.partial_global_IoU) / len(
            self.partial_global_IoU
        )
        scores["avg partial"]["avg local IoU"] = sum(self.partial_avg_local_IoU) / len(
            self.partial_avg_local_IoU
        )

        scores["reviews"] = []

        for i in range(len(self.pred_outputs)):
            scores_for_review_i = {}
            scores_for_review_i["idx"] = i
            scores_for_review_i["exact local IoU"] = self.local_IoU[i]
            sum_partial_local_IoU = 0
            for j, term in enumerate(TERM_LIST[: self.tuple_len]):
                scores_for_review_i[f"{term} local IoU"] = self.partial_local_IoU[j][i]
                sum_partial_local_IoU += self.partial_local_IoU[j][i]

            scores_for_review_i["avg partial local IoU"] = sum_partial_local_IoU / len(
                self.partial_local_IoU
            )

            scores_for_review_i["review"] = self.reviews[i]
            scores_for_review_i["pred"] = self.pred_outputs[i]
            scores_for_review_i["true"] = self.true_outputs[i]

            scores["reviews"].append(scores_for_review_i)

        return scores


scores = {}

if args.mvp_output:
    evaluate_mvp_outputs = Evaluator(get_mvp_output, category_file=args.category_file)
    scores = evaluate_mvp_outputs.get_scores()
elif args.llm_output:
    evaluate_llm_outputs = Evaluator(get_llm_output)
    scores = evaluate_llm_outputs.get_scores()
elif args.t5_output:
    evaluate_t5_outputs = Evaluator(get_t5_output)
    scores = evaluate_t5_outputs.get_scores()

with open(args.output_file, "w") as file:
    json.dump(scores, file, indent=4)
