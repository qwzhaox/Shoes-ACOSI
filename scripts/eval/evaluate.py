import json
import argparse
from statistics import mean, median, stdev
from utils.evaluate_utils import (
    ASPECT_IDX,
    SENTIMENT_IDX,
    OPINION_IDX,
    IMPLICIT_IND_IDX,
    IDX_LIST,
    TERM_LIST,
    NUM_SPANS,
    SPAN_IDX,
    get_combos,
    indexify_outputs,
    listify_outputs,
    comboify_outputs,
    get_precision_recall_fl_IoU,
)
from utils.mvp_evaluate import get_mvp_output
from utils.llm_evaluate import get_llm_output
from utils.t5_evaluate import get_t5_output
from utils.gen_scl_nat_evaluate import get_gen_scl_nat_output

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset_file", type=str, default="data/acosi_dataset/shoes/test.txt"
)
parser.add_argument(
    "-m",
    "--model_output_file",
    type=str,
    default="data/mvp_dataset/result_cd_acosi_shoes_path5_beam1.pickle",
)
parser.add_argument(
    "-c", "--category_file", type=str, default="data/mvp_dataset/category_dict.json"
)
parser.add_argument(
    "-o", "--output_file", type=str, default="data/mvp_dataset/scores.json"
)
parser.add_argument(
    "-t",
    "--task",
    type=str,
    choices=["acos-extract", "acos-extend", "acosi-extract"],
    default="acosi-extract",
)
parser.add_argument("-mvp", "--mvp_output", action="store_true")
parser.add_argument("-llm", "--llm_output", action="store_true")
parser.add_argument("-t5", "--t5_output", action="store_true")
parser.add_argument("-gen-scl-nat", "--gen-scl-nat_output", action="store_true")
args = parser.parse_args()


class Evaluator:
    def __init__(self, process_func, **kwargs):
        self.pred_outputs = process_func(model_output_file=args.model_output_file, **kwargs)

        with open(args.dataset_file, "r") as file:
            dataset = file.readlines()

        self.__set_outputs(dataset)
        self.__set_tuple_len_according_to_task()
        self.__init_exact_scores()
        self.__init_partial_scores()
        self.__init_combo_scores()

    def __set_outputs(self, dataset):
        self.reviews = [x.split("####")[0] for x in dataset]

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

    def __set_tuple_len_according_to_task(self):
        if args.task == "acos-extend" or args.task == "acosi-extract":
            self.tuple_len = len(TERM_LIST)
        elif args.task == "acos-extract":
            self.tuple_len = len(TERM_LIST) - 1
        else:
            raise ValueError("Invalid task")
        self.combos = get_combos(self.tuple_len)
        if len(self.pred_outputs[0][0]) > self.tuple_len:
            new_pred_outputs = []
            for annotation in self.pred_outputs:
                annotation = [quint[: self.tuple_len] for quint in annotation]
                new_pred_outputs.append(annotation)
            self.pred_outputs = new_pred_outputs

    def __init_exact_scores(self):
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.macro_IoU = 0
        self.micro_IoU = []
        self.avg_micro_IoU = 0

    def __init_partial_scores(self):
        self.partial_precision = [0] * self.tuple_len
        self.partial_recall = [0] * self.tuple_len
        self.partial_f1 = [0] * self.tuple_len
        self.partial_macro_IoU = [0] * self.tuple_len
        self.partial_micro_IoU = [[]] * self.tuple_len
        self.partial_avg_micro_IoU = [0] * self.tuple_len

        self.partial_span_precision = [0] * NUM_SPANS
        self.partial_span_recall = [0] * NUM_SPANS
        self.partial_span_f1 = [0] * NUM_SPANS
        self.partial_span_macro_IoU = [0] * NUM_SPANS
        self.partial_span_micro_IoU = [[]] * NUM_SPANS
        self.partial_span_avg_micro_IoU = [0] * NUM_SPANS

    def __init_combo_scores(self):
        self.combo_precision = [0] * len(self.combos)
        self.combo_recall = [0] * len(self.combos)
        self.combo_f1 = [0] * len(self.combos)
        self.combo_macro_IoU = [0] * len(self.combos)
        self.combo_micro_IoU = [[]] * len(self.combos)
        self.combo_avg_micro_IoU = [0] * len(self.combos)

    def __remove_direct_opinions(self):
        pred_outputs_remove_direct_opinion = []
        true_outputs_remove_direct_opinion = []
        for true_output, pred_output in zip(self.true_outputs, self.pred_outputs):
            list_of_tuples_true = []
            list_of_tuples_pred = []
            for true_quint, pred_quint in zip(true_output, pred_output):
                assert true_quint[IMPLICIT_IND_IDX] == pred_quint[IMPLICIT_IND_IDX]
                if true_quint[IMPLICIT_IND_IDX] == "indirect":
                    list_of_tuples_true.append(true_quint)
                    list_of_tuples_pred.append(pred_quint)
            assert len(list_of_tuples_true) == len(list_of_tuples_pred)

            true_outputs_remove_direct_opinion.append(list_of_tuples_true)
            pred_outputs_remove_direct_opinion.append(list_of_tuples_pred)

        self.pred_outputs = pred_outputs_remove_direct_opinion
        self.true_outputs = true_outputs_remove_direct_opinion

    def calc_metadata(self):
        print("Calculating metadata...")
        self.review_len_list = []
        self.num_predicted_list = []
        self.num_true_list = []
        self.aspect_span_len_list = []
        self.opinion_span_len_list = []
        for i, review in enumerate(self.reviews):
            self.review_len_list.append(len(review.split()))
            self.num_predicted_list.append(len(self.pred_outputs[i]))
            self.num_true_list.append(len(self.true_outputs[i]))

        self.num_ea_eo = 0
        self.num_ea_io = 0
        self.num_ia_eo = 0
        self.num_ia_io = 0

        self.num_pred_ea_eo = 0
        self.num_pred_ea_io = 0
        self.num_pred_ia_eo = 0
        self.num_pred_ia_io = 0

        self.num_positive = 0
        self.num_negative = 0
        self.num_neutral = 0

        self.num_pred_positive = 0
        self.num_pred_negative = 0
        self.num_pred_neutral = 0

        for pred_output, true_output in zip(self.pred_outputs, self.true_outputs):
            for quint in pred_output:
                if quint[ASPECT_IDX] == "NULL" and quint[OPINION_IDX] == "NULL":
                    self.num_pred_ia_io += 1
                elif quint[ASPECT_IDX] == "NULL":
                    self.num_pred_ia_eo += 1
                elif quint[OPINION_IDX] == "NULL":
                    self.num_pred_ea_io += 1
                else:
                    self.num_pred_ea_eo += 1

                if quint[SENTIMENT_IDX] == "positive":
                    self.num_pred_positive += 1
                elif quint[SENTIMENT_IDX] == "negative":
                    self.num_pred_negative += 1
                else:
                    self.num_pred_neutral += 1
            
            for quint in true_output:
                if quint[ASPECT_IDX] == "NULL" and quint[OPINION_IDX] == "NULL":
                    self.num_ia_io += 1
                elif quint[ASPECT_IDX] == "NULL":
                    self.num_ia_eo += 1
                elif quint[OPINION_IDX] == "NULL":
                    self.num_ea_io += 1
                else:
                    self.num_ea_eo += 1

                if quint[SENTIMENT_IDX] == "positive":
                    self.num_positive += 1
                elif quint[SENTIMENT_IDX] == "negative":
                    self.num_negative += 1
                else:
                    self.num_neutral += 1

    def calc_exact_scores(self):
        print("Calculating exact scores...")
        (
            self.precision,
            self.recall,
            self.f1,
            self.macro_IoU,
            self.micro_IoU,
            self.avg_micro_IoU,
        ) = get_precision_recall_fl_IoU(self.pred_outputs, self.true_outputs)

    def calc_partial_scores(self):
        for idx in IDX_LIST[: self.tuple_len]:
            print(f"Calculating partial scores for {TERM_LIST[idx]}...")
            pred_outputs = listify_outputs(self.pred_outputs, idx)
            true_outputs = listify_outputs(self.true_outputs, idx)
            (
                self.partial_precision[idx],
                self.partial_recall[idx],
                self.partial_f1[idx],
                self.partial_macro_IoU[idx],
                self.partial_micro_IoU[idx],
                self.partial_avg_micro_IoU[idx],
            ) = get_precision_recall_fl_IoU(pred_outputs, true_outputs)

            if idx == ASPECT_IDX or idx == OPINION_IDX:
                pred_outputs = indexify_outputs(
                    self.reviews, self.pred_outputs, idx)
                true_outputs = indexify_outputs(
                    self.reviews, self.true_outputs, idx)

                (
                    self.partial_span_precision[SPAN_IDX[idx]],
                    self.partial_span_recall[SPAN_IDX[idx]],
                    self.partial_span_f1[SPAN_IDX[idx]],
                    self.partial_span_macro_IoU[SPAN_IDX[idx]],
                    self.partial_span_micro_IoU[SPAN_IDX[idx]],
                    self.partial_span_avg_micro_IoU[SPAN_IDX[idx]],
                ) = get_precision_recall_fl_IoU(pred_outputs, true_outputs)

    def calc_combo_scores(self):
        for combo_idx, combo in enumerate(self.combos):
            combo_str = "-".join([TERM_LIST[idx] for idx in combo])
            print(f"Calculating combo scores for {combo_str}...")
            pred_outputs = comboify_outputs(self.pred_outputs, combo)
            true_outputs = comboify_outputs(self.true_outputs, combo)

            (
                self.combo_precision[combo_idx],
                self.combo_recall[combo_idx],
                self.combo_f1[combo_idx],
                self.combo_macro_IoU[combo_idx],
                self.combo_micro_IoU[combo_idx],
                self.combo_avg_micro_IoU[combo_idx],
            ) = get_precision_recall_fl_IoU(pred_outputs, true_outputs)

    def get_metadata(self):
        metadata = {}

        metadata["num tokens per review"] = {}
        metadata["num tokens per review"]["mean"] = mean(self.review_len_list)
        metadata["num tokens per review"]["median"] = median(
            self.review_len_list)
        metadata["num tokens per review"]["stdev"] = stdev(
            self.review_len_list)

        metadata["num tuples predicted"] = {}
        metadata["num tuples predicted"]["mean"] = mean(
            self.num_predicted_list)
        metadata["num tuples predicted"]["median"] = median(
            self.num_predicted_list)
        metadata["num tuples predicted"]["stdev"] = stdev(
            self.num_predicted_list)

        metadata["num tuples"] = {}
        metadata["num tuples"]["mean"] = mean(self.num_true_list)
        metadata["num tuples"]["median"] = median(self.num_true_list)
        metadata["num tuples"]["stdev"] = stdev(self.num_true_list)

        metadata["num e/i"] = {}
        metadata["num e/i"]["ea eo"] = self.num_ea_eo
        metadata["num e/i"]["ea io"] = self.num_ea_io
        metadata["num e/i"]["ia eo"] = self.num_ia_eo
        metadata["num e/i"]["ia io"] = self.num_ia_io

        metadata["num e/i"]["pred ea eo"] = self.num_pred_ea_eo
        metadata["num e/i"]["pred ea io"] = self.num_pred_ea_io
        metadata["num e/i"]["pred ia eo"] = self.num_pred_ia_eo
        metadata["num e/i"]["pred ia io"] = self.num_pred_ia_io

        metadata["num sentiment"] = {}
        metadata["num sentiment"]["positive"] = self.num_positive
        metadata["num sentiment"]["negative"] = self.num_negative
        metadata["num sentiment"]["neutral"] = self.num_neutral

        metadata["num sentiment"]["pred positive"] = self.num_pred_positive
        metadata["num sentiment"]["pred negative"] = self.num_pred_negative
        metadata["num sentiment"]["pred neutral"] = self.num_pred_neutral

        return metadata

    def get_exact_scores(self):
        scores = {}
        scores["precision"] = self.precision
        scores["recall"] = self.recall
        scores["f1-score"] = self.f1
        scores["macro IoU"] = self.macro_IoU
        scores["avg micro IoU"] = self.avg_micro_IoU
        return scores

    def get_partial_scores(self, term_idx):
        scores = {}
        scores["precision"] = self.partial_precision[term_idx]
        scores["recall"] = self.partial_recall[term_idx]
        scores["f1-score"] = self.partial_f1[term_idx]
        scores["macro IoU"] = self.partial_macro_IoU[term_idx]
        scores["avg micro IoU"] = self.partial_avg_micro_IoU[term_idx]

        if term_idx == ASPECT_IDX or term_idx == OPINION_IDX:
            scores["span precision"] = self.partial_span_precision[SPAN_IDX[term_idx]]
            scores["span recall"] = self.partial_span_recall[SPAN_IDX[term_idx]]
            scores["span f1-score"] = self.partial_span_f1[SPAN_IDX[term_idx]]
            scores["span macro IoU"] = self.partial_span_macro_IoU[SPAN_IDX[term_idx]]
            scores["span avg micro IoU"] = self.partial_span_avg_micro_IoU[
                SPAN_IDX[term_idx]
            ]

        return scores

    def get_combo_scores(self, combo_idx):
        scores = {}
        scores["precision"] = self.combo_precision[combo_idx]
        scores["recall"] = self.combo_recall[combo_idx]
        scores["f1-score"] = self.combo_f1[combo_idx]
        scores["macro IoU"] = self.combo_macro_IoU[combo_idx]
        scores["avg micro IoU"] = self.combo_avg_micro_IoU[combo_idx]
        return scores

    def get_avg_partial_scores(self):
        scores = {}
        scores["precision"] = sum(
            self.partial_precision) / len(self.partial_precision)
        scores["recall"] = sum(self.partial_recall) / len(self.partial_recall)
        scores["f1-score"] = sum(self.partial_f1) / len(self.partial_f1)
        scores["macro IoU"] = sum(
            self.partial_macro_IoU) / len(self.partial_macro_IoU)
        scores["avg micro IoU"] = sum(self.partial_avg_micro_IoU) / len(
            self.partial_avg_micro_IoU
        )
        return scores

    def get_scores_per_review(self):
        scores = []
        for i in range(len(self.pred_outputs)):
            scores_for_rev_i = {}
            scores_for_rev_i["idx"] = i
            scores_for_rev_i["review"] = self.reviews[i]
            scores_for_rev_i["pred"] = self.pred_outputs[i]
            scores_for_rev_i["true"] = self.true_outputs[i]

            scores_for_rev_i["metadata"] = {}
            scores_for_rev_i["metadata"]["review_length"] = self.review_len_list[i]
            scores_for_rev_i["metadata"]["num_predicted"] = self.num_predicted_list[i]
            scores_for_rev_i["metadata"]["num_true"] = self.num_true_list[i]

            if args.task == "acos-extend":
                scores_for_rev_i[
                    f"total {TERM_LIST[OPINION_IDX]} micro IoU"
                ] = self.partial_micro_IoU[OPINION_IDX][i]
                scores_for_rev_i[f"indirect {TERM_LIST[OPINION_IDX]} micro IoU"] = {
                }
            else:
                scores_for_rev_i["exact micro IoU"] = self.micro_IoU[i]
                sum_partial_micro_IoU = 0
                for j, term in enumerate(TERM_LIST[: self.tuple_len]):
                    scores_for_rev_i[f"{term} micro IoU"] = self.partial_micro_IoU[j][i]
                    sum_partial_micro_IoU += self.partial_micro_IoU[j][i]

                # scores_for_rev_i["avg partial micro IoU"] = sum_partial_micro_IoU / len(
                #     self.partial_micro_IoU
                # )

                for combo_idx, combo in enumerate(self.combos):
                    combo_str = "-".join([TERM_LIST[idx] for idx in combo])
                    scores_for_rev_i[f"{combo_str} micro IoU"] = self.combo_micro_IoU[
                        combo_idx
                    ][i]

            scores.append(scores_for_rev_i)

        return scores

    def get_scores(self):
        self.calc_metadata()
        self.calc_exact_scores()
        self.calc_partial_scores()
        self.calc_combo_scores()

        scores = {}

        scores["metadata"] = self.get_metadata()

        if args.task == "acos-extend":
            scores[f"total {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX
            )
            scores[f"indirect {TERM_LIST[OPINION_IDX]}"] = {}
        else:
            scores["exact"] = self.get_exact_scores()
            for i, term in enumerate(TERM_LIST[: self.tuple_len]):
                scores[term] = self.get_partial_scores(i)
            # scores["avg partial"] = self.get_avg_partial_scores()

            for combo_idx, combo in enumerate(self.combos):
                combo_str = "-".join([TERM_LIST[idx] for idx in combo])
                scores[combo_str] = self.get_combo_scores(combo_idx)

        scores["reviews"] = self.get_scores_per_review()

        if args.task == "acos-extend":
            self.__remove_direct_opinions()
            self.calc_partial_scores()
            scores[f"indirect {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX
            )
            for i, review in enumerate(scores["reviews"]):
                review[
                    f"indirect {TERM_LIST[OPINION_IDX]} micro IoU"
                ] = self.partial_micro_IoU[OPINION_IDX][i]

        return scores


scores = {}

if args.mvp_output:
    evaluate_mvp_outputs = Evaluator(
        get_mvp_output, category_file=args.category_file)
    scores = evaluate_mvp_outputs.get_scores()
elif args.llm_output:
    evaluate_llm_outputs = Evaluator(get_llm_output, category_file=args.category_file)
    scores = evaluate_llm_outputs.get_scores()
elif args.t5_output:
    evaluate_t5_outputs = Evaluator(get_t5_output)
    scores = evaluate_t5_outputs.get_scores()
elif args.gen_scl_nat_output:
    evaluate_gen_scl_nat_outputs = Evaluator(get_gen_scl_nat_output, category_file=args.category_file, task=args.task)
    scores = evaluate_gen_scl_nat_outputs.get_scores()

with open(args.output_file, "w") as file:
    json.dump(scores, file, indent=4)
