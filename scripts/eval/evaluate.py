import json
import argparse
from pathlib import Path
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
    process_dataset,
    accumulate_ea_eo_ia_io_sent,
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

        self.reviews, self.true_outputs = process_dataset(dataset)
        self.__set_tuple_len_according_to_task()
        self.__init_exact_scores()
        self.__init_partial_scores()
        self.__init_combo_scores()

    def calc_metadata(self):
        print("Calculating metadata...")
        train_dataset = Path(args.dataset_file).parent / "train.txt"
        dev_dataset = Path(args.dataset_file).parent / "dev.txt"

        with open(train_dataset, "r") as file:
            train_dataset = file.readlines()
        
        with open(dev_dataset, "r") as file:
            dev_dataset = file.readlines()

        train_reviews, train_outputs = process_dataset(train_dataset)
        dev_reviews, dev_outputs = process_dataset(dev_dataset)

        self.test_split = len(self.reviews)
        self.train_split = len(train_reviews)
        self.dev_split = len(dev_reviews)

        self.total_dataset = self.test_split + self.train_split + self.dev_split

        self.review_len_list = []
        self.total_true_list = []

        self.num_true_list = []
        self.num_predicted_list = []

        self.pos_tuples_per_review = []
        self.neg_tuples_per_review = []
        self.neu_tuples_per_rev = []

        # self.aspect_span_len_list = []
        # self.opinion_span_len_list = []

        for i, review in enumerate(self.reviews):
            self.review_len_list.append(len(review.split()))
            self.total_true_list.append(len(self.true_outputs[i]))

            self.num_true_list.append(len(self.true_outputs[i]))
            self.num_predicted_list.append(len(self.pred_outputs[i]))

        for i, review in enumerate(train_reviews):
            self.review_len_list.append(len(review.split()))
            self.total_true_list.append(len(train_outputs[i]))
        
        for i, review in enumerate(dev_reviews):
            self.review_len_list.append(len(review.split()))
            self.total_true_list.append(len(dev_outputs[i]))

        self.__init_ea_eo_ia_io()
        self.__init_sentiment()

        for pred_output, true_output in zip(self.pred_outputs, self.true_outputs):
            pos, neg, neu = self.num_positive, self.num_negative, self.num_neutral

            pos, neg, neu = self.num_positive, self.num_negative, self.num_neutral

            self.__accumulate_true(true_output)
            self.__accumulate_pred(pred_output)

            self.pos_neg_neu_tuples_per_review.append(self.num_positive - pos)
            self.pos_neg_neu_tuples_per_review.append(self.num_negative - neg)
            self.pos_neg_neu_tuples_per_review.append(self.num_neutral - neu)

            self.pos_neg_neu_tuples_per_review.append(self.num_positive - pos)
            self.pos_neg_neu_tuples_per_review.append(self.num_negative - neg)
            self.pos_neg_neu_tuples_per_review.append(self.num_neutral - neu)

        self.__set_test_ea_eo_ia_io()
        self.__set_test_sentiment()

        for output in train_outputs + dev_outputs:
            pos, neg, neu = self.num_positive, self.num_negative, self.num_neutral

            pos, neg, neu = self.num_positive, self.num_negative, self.num_neutral

            self.__accumulate_true(output)

            self.pos_neg_neu_tuples_per_review.append(self.num_positive - pos)
            self.pos_neg_neu_tuples_per_review.append(self.num_negative - neg)
            self.pos_neg_neu_tuples_per_review.append(self.num_neutral - neu)
        
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

        metadata["splits"] = {}
        metadata["splits"]["test"] = self.test_split
        metadata["splits"]["train"] = self.train_split
        metadata["splits"]["dev"] = self.dev_split
        metadata["splits"]["total"] = self.total_dataset

        metadata["tokens/review"] = {}
        metadata["tokens/review"]["mean"] = mean(self.review_len_list)
        metadata["tokens/review"]["median"] = median(
            self.review_len_list)
        metadata["tokens/review"]["stdev"] = stdev(
            self.review_len_list)
        metadata["tokens/review"]["min"] = min(self.review_len_list)
        metadata["tokens/review"]["max"] = max(self.review_len_list)
        metadata["tokens/review"]["count"] = sum(self.review_len_list)

        metadata["tuples predicted"] = {}
        metadata["tuples predicted"]["mean"] = mean(
            self.num_predicted_list)
        metadata["tuples predicted"]["median"] = median(
            self.num_predicted_list)
        metadata["tuples predicted"]["stdev"] = stdev(
            self.num_predicted_list)
        metadata["tuples predicted"]["min"] = min(self.num_predicted_list)
        metadata["tuples predicted"]["max"] = max(self.num_predicted_list)
        metadata["tuples predicted"]["count"] = sum(self.num_predicted_list)

        metadata["tuples test"] = {}
        metadata["tuples test"]["mean"] = mean(self.num_true_list)
        metadata["tuples test"]["median"] = median(self.num_true_list)
        metadata["tuples test"]["stdev"] = stdev(self.num_true_list)
        metadata["tuples test"]["min"] = min(self.num_true_list)
        metadata["tuples test"]["max"] = max(self.num_true_list)
        metadata["tuples test"]["count"] = sum(self.num_true_list)

        metadata["tuples"] = {}
        metadata["tuples"]["mean"] = mean(self.total_true_list)
        metadata["tuples"]["median"] = median(self.total_true_list)
        metadata["tuples"]["stdev"] = stdev(self.total_true_list)
        metadata["tuples"]["min"] = min(self.total_true_list)
        metadata["tuples"]["max"] = max(self.total_true_list)
        metadata["tuples"]["count"] = sum(self.total_true_list)

        metadata["ea/eo/ia/io"] = {}
        metadata["ea/eo/ia/io"]["ea/eo"] = self.num_ea_eo
        metadata["ea/eo/ia/io"]["ea/io"] = self.num_ea_io
        metadata["ea/eo/ia/io"]["ia/eo"] = self.num_ia_eo
        metadata["ea/eo/ia/io"]["ia/io"] = self.num_ia_io

        metadata["ea/eo/ia/io"]["test ea/eo"] = self.num_test_ea_eo
        metadata["ea/eo/ia/io"]["test ea/io"] = self.num_test_ea_io
        metadata["ea/eo/ia/io"]["test ia/eo"] = self.num_test_ia_eo
        metadata["ea/eo/ia/io"]["test ia/io"] = self.num_test_ia_io

        metadata["ea/eo/ia/io"]["pred ea/eo"] = self.num_pred_ea_eo
        metadata["ea/eo/ia/io"]["pred ea/io"] = self.num_pred_ea_io
        metadata["ea/eo/ia/io"]["pred ia/eo"] = self.num_pred_ia_eo
        metadata["ea/eo/ia/io"]["pred ia/io"] = self.num_pred_ia_io

        metadata["sentiment"] = {}
        metadata["sentiment"]["positive"] = self.num_positive
        metadata["sentiment"]["negative"] = self.num_negative
        metadata["sentiment"]["neutral"] = self.num_neutral

        metadata["sentiment"]["test positive"] = self.num_test_positive
        metadata["sentiment"]["test negative"] = self.num_test_negative
        metadata["sentiment"]["test neutral"] = self.num_test_neutral

        metadata["sentiment"]["pred positive"] = self.num_pred_positive
        metadata["sentiment"]["pred negative"] = self.num_pred_negative
        metadata["sentiment"]["pred neutral"] = self.num_pred_neutral

        return metadata

    def get_exact_scores(self):
        scores = {}
        scores["precision"] = self.precision
        scores["recall"] = self.recall
        scores["f1-score"] = self.f1
        scores["macro IoU"] = self.macro_IoU
        scores["avg micro IoU"] = self.avg_micro_IoU
        return scores

    def get_partial_scores(self, term_idx, span=True):
        scores = {}
        scores["precision"] = self.partial_precision[term_idx]
        scores["recall"] = self.partial_recall[term_idx]
        scores["f1-score"] = self.partial_f1[term_idx]
        scores["macro IoU"] = self.partial_macro_IoU[term_idx]
        scores["avg micro IoU"] = self.partial_avg_micro_IoU[term_idx]

        if span and (term_idx == ASPECT_IDX or term_idx == OPINION_IDX):
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
        else:
            scores["exact"] = self.get_exact_scores()
            for i, term in enumerate(TERM_LIST[: self.tuple_len]):
                scores[term] = self.get_partial_scores(i)
            # scores["avg partial"] = self.get_avg_partial_scores()

            for combo_idx, combo in enumerate(self.combos):
                combo_str = "-".join([TERM_LIST[idx] for idx in combo])
                scores[combo_str] = self.get_combo_scores(combo_idx)

        scores[f"exact (only indirect {TERM_LIST[OPINION_IDX]})"] = {}
        scores[f"indirect {TERM_LIST[OPINION_IDX]}"] = {}
        if args.task == "acos-extract" or args.task == "acosi-extract":
            scores[f"exact (only direct {TERM_LIST[OPINION_IDX]})"] = {}
            scores[f"direct {TERM_LIST[OPINION_IDX]}"] = {}
        scores["reviews"] = self.get_scores_per_review()

        full_true_outputs = self.true_outputs
        full_pred_outputs = self.pred_outputs

        self.__remove_opinions(keep_opinion_type="indirect")
        self.calc_exact_scores()
        self.calc_partial_scores()
        scores[f"exact (only indirect {TERM_LIST[OPINION_IDX]})"] = self.get_exact_scores()
        if args.task == "acos-extract":
            scores[f"indirect {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX,
                span=False
            )
        else:
            scores[f"indirect {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX
            )
        for i, review in enumerate(scores["reviews"]):
            review[
                f"indirect {TERM_LIST[OPINION_IDX]} micro IoU"
            ] = self.partial_micro_IoU[OPINION_IDX][i]

        if args.task == "acos-extract" or args.task == "acosi-extract":
            self.true_outputs = full_true_outputs
            self.pred_outputs = full_pred_outputs

            self.__remove_opinions(keep_opinion_type="direct")
            self.calc_exact_scores()
            self.calc_partial_scores()
            scores[f"exact (only direct {TERM_LIST[OPINION_IDX]})"] = self.get_exact_scores()
            scores[f"direct {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX
            )
            for i, review in enumerate(scores["reviews"]):
                review[
                    f"direct {TERM_LIST[OPINION_IDX]} micro IoU"
                ] = self.partial_micro_IoU[OPINION_IDX][i]

        return scores

######################## HELPER FUNCTIONS ########################
    def __set_tuple_len_according_to_task(self):
        if args.task == "acos-extend" or args.task == "acosi-extract":
            self.tuple_len = len(TERM_LIST)
        elif args.task == "acos-extract":
            self.tuple_len = len(TERM_LIST) - 1
        else:
            raise ValueError("Invalid task")
        self.combos = get_combos(self.tuple_len)
        if self.pred_outputs[0] and len(self.pred_outputs[0][0]) > self.tuple_len:
            new_pred_outputs = []
            for annotation in self.pred_outputs:
                annotation = [quint[: self.tuple_len] for quint in annotation]
                new_pred_outputs.append(annotation)
            self.pred_outputs = new_pred_outputs

    def __remove_opinions(self, keep_opinion_type="indirect"):
        true_outputs_remove_direct_opinion = []
        pred_outputs_remove_direct_opinion = []
        for true_output, pred_output in zip(self.true_outputs, self.pred_outputs):
            list_of_tuples_true = []
            for true_quint in true_output:
                if args.task == "acos-extend" or args.task == "acosi-extract":
                    if keep_opinion_type == "indirect" and true_quint[IMPLICIT_IND_IDX] == "indirect":
                        list_of_tuples_true.append(true_quint)
                    elif keep_opinion_type == "direct" and true_quint[IMPLICIT_IND_IDX] == "direct":
                        list_of_tuples_true.append(true_quint)
                else:
                    if keep_opinion_type == "indirect" and true_quint[OPINION_IDX] == "NULL":
                        list_of_tuples_true.append(true_quint)
                    elif keep_opinion_type == "direct" and true_quint[OPINION_IDX] != "NULL":
                        list_of_tuples_true.append(true_quint)

            list_of_tuples_pred = []
            for pred_quint in pred_output:
                if args.task == "acos-extend" or args.task == "acosi-extract":
                    if keep_opinion_type == "indirect" and pred_quint[IMPLICIT_IND_IDX] == "indirect":
                        list_of_tuples_pred.append(pred_quint)
                    elif keep_opinion_type == "direct" and pred_quint[IMPLICIT_IND_IDX] == "direct":
                        list_of_tuples_pred.append(pred_quint)
                else:
                    if keep_opinion_type == "indirect" and pred_quint[OPINION_IDX] == "NULL":
                        list_of_tuples_pred.append(pred_quint)
                    elif keep_opinion_type == "direct" and pred_quint[OPINION_IDX] != "NULL":
                        list_of_tuples_pred.append(pred_quint)

            true_outputs_remove_direct_opinion.append(list_of_tuples_true)
            pred_outputs_remove_direct_opinion.append(list_of_tuples_pred)

        self.pred_outputs = pred_outputs_remove_direct_opinion
        self.true_outputs = true_outputs_remove_direct_opinion

######################## METADATA FUNCTIONS ########################
    def __set_test_ea_eo_ia_io(self):
        self.num_test_ea_eo = self.num_ea_eo
        self.num_test_ea_io = self.num_ea_io
        self.num_test_ia_eo = self.num_ia_eo
        self.num_test_ia_io = self.num_ia_io
    
    def __set_test_sentiment(self):
        self.num_test_positive = self.num_positive
        self.num_test_negative = self.num_negative
        self.num_test_neutral = self.num_neutral

    def __accumulate_true(self, output):
        self.num_ea_eo, \
        self.num_ea_io, \
        self.num_ia_eo, \
        self.num_ia_io, \
        self.num_positive, \
        self.num_negative, \
        self.num_neutral = accumulate_ea_eo_ia_io_sent(
            output, 
            (
                self.num_ea_eo, 
                self.num_ea_io, 
                self.num_ia_eo, 
                self.num_ia_io, 
                self.num_positive, 
                self.num_negative, 
                self.num_neutral
            ),
            self.tuple_len
        )

    def __accumulate_pred(self, output):
        self.num_pred_ea_eo, \
        self.num_pred_ea_io, \
        self.num_pred_ia_eo, \
        self.num_pred_ia_io, \
        self.num_pred_positive, \
        self.num_pred_negative, \
        self.num_pred_neutral = accumulate_ea_eo_ia_io_sent(
            output,
            (
                self.num_pred_ea_eo,
                self.num_pred_ea_io,
                self.num_pred_ia_eo,
                self.num_pred_ia_io,
                self.num_pred_positive,
                self.num_pred_negative,
                self.num_pred_neutral
            ),
            self.tuple_len
        )
        
######################## INITIALIZATION FUNCTIONS ########################
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

    def __init_ea_eo_ia_io(self):
        self.num_ea_eo = 0
        self.num_ea_io = 0
        self.num_ia_eo = 0
        self.num_ia_io = 0

        self.num_test_ea_eo = 0
        self.num_test_ea_io = 0
        self.num_test_ia_eo = 0
        self.num_test_ia_io = 0

        self.num_pred_ea_eo = 0
        self.num_pred_ea_io = 0
        self.num_pred_ia_eo = 0
        self.num_pred_ia_io = 0

    def __init_sentiment(self):
        self.num_positive = 0
        self.num_negative = 0
        self.num_neutral = 0

        self.num_test_positive = 0
        self.num_test_negative = 0
        self.num_test_neutral = 0

        self.num_pred_positive = 0
        self.num_pred_negative = 0
        self.num_pred_neutral = 0


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
