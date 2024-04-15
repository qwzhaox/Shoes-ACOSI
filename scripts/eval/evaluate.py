import json
import argparse
from copy import deepcopy
from pathlib import Path
from statistics import mean, median, stdev
from math import inf
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
    accum_ea_eo_ia_io_sent,
    accum_span_len,
    accum_polarities,
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
        self.pred_outputs = process_func(
            model_output_file=args.model_output_file, **kwargs
        )

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
        self.tuples_list = []

        self.test_tuples_list = []
        self.pred_tuples_list = []

        for i, review in enumerate(self.reviews):
            self.review_len_list.append(len(review.split()))
            self.tuples_list.append(len(self.true_outputs[i]))
            self.pred_tuples_list.append(len(self.pred_outputs[i]))

        self.test_tuples_list = deepcopy(self.tuples_list)

        for i, review in enumerate(train_reviews):
            self.review_len_list.append(len(review.split()))
            self.tuples_list.append(len(train_outputs[i]))

        for i, review in enumerate(dev_reviews):
            self.review_len_list.append(len(review.split()))
            self.tuples_list.append(len(dev_outputs[i]))

        self.__init_ea_eo_ia_io()
        self.__init_sentiment()
        self.__init_span_len()

        for pred_output, true_output in zip(self.pred_outputs, self.true_outputs):
            self.__accum_ea_eo_ia_io_span_len_sent(true_output, accum_true=True)
            self.__accum_ea_eo_ia_io_span_len_sent(pred_output, accum_pred=True)

        self.__set_test_ea_eo_ia_io()
        self.__set_test_sentiment()
        self.__set_test_span_len()

        accum_polarities(
            self.test_polarities, self.test_pos, self.test_neg, self.test_neu
        )
        accum_polarities(
            self.pred_polarities, self.pred_pos, self.pred_neg, self.pred_neu
        )

        for output in train_outputs + dev_outputs:
            self.__accum_ea_eo_ia_io_span_len_sent(output, accum_true=True)

        accum_polarities(self.polarities, self.pos, self.neg, self.neu)

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
                pred_outputs = indexify_outputs(self.reviews, self.pred_outputs, idx)
                true_outputs = indexify_outputs(self.reviews, self.true_outputs, idx)

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
        metadata["tokens/review"]["median"] = median(self.review_len_list)
        metadata["tokens/review"]["stdev"] = stdev(self.review_len_list)
        metadata["tokens/review"]["min"] = min(self.review_len_list)
        metadata["tokens/review"]["max"] = max(self.review_len_list)
        metadata["tokens/review"]["count"] = sum(self.review_len_list)

        metadata["tuples"] = {}
        metadata["tuples"]["mean"] = mean(self.tuples_list)
        metadata["tuples"]["median"] = median(self.tuples_list)
        metadata["tuples"]["stdev"] = stdev(self.tuples_list)
        metadata["tuples"]["min"] = min(self.tuples_list)
        metadata["tuples"]["max"] = max(self.tuples_list)
        metadata["tuples"]["count"] = sum(self.tuples_list)

        metadata["tuples test"] = {}
        metadata["tuples test"]["mean"] = mean(self.test_tuples_list)
        metadata["tuples test"]["median"] = median(self.test_tuples_list)
        metadata["tuples test"]["stdev"] = stdev(self.test_tuples_list)
        metadata["tuples test"]["min"] = min(self.test_tuples_list)
        metadata["tuples test"]["max"] = max(self.test_tuples_list)
        metadata["tuples test"]["count"] = sum(self.test_tuples_list)

        metadata["tuples predicted"] = {}
        metadata["tuples predicted"]["mean"] = mean(self.pred_tuples_list)
        metadata["tuples predicted"]["median"] = median(self.pred_tuples_list)
        metadata["tuples predicted"]["stdev"] = stdev(self.pred_tuples_list)
        metadata["tuples predicted"]["min"] = min(self.pred_tuples_list)
        metadata["tuples predicted"]["max"] = max(self.pred_tuples_list)
        metadata["tuples predicted"]["count"] = sum(self.pred_tuples_list)

        metadata["ea/eo/ia/io"] = {}
        metadata["ea/eo/ia/io"]["ea/eo"] = sum(self.ea_eo) / metadata["tuples"]["count"]
        metadata["ea/eo/ia/io"]["ea/io"] = sum(self.ea_io) / metadata["tuples"]["count"]
        metadata["ea/eo/ia/io"]["ia/eo"] = sum(self.ia_eo) / metadata["tuples"]["count"]
        metadata["ea/eo/ia/io"]["ia/io"] = sum(self.ia_io) / metadata["tuples"]["count"]

        metadata["ea/eo/ia/io"]["test ea/eo"] = (
            sum(self.test_ea_eo) / metadata["tuples test"]["count"]
        )
        metadata["ea/eo/ia/io"]["test ea/io"] = (
            sum(self.test_ea_io) / metadata["tuples test"]["count"]
        )
        metadata["ea/eo/ia/io"]["test ia/eo"] = (
            sum(self.test_ia_eo) / metadata["tuples test"]["count"]
        )
        metadata["ea/eo/ia/io"]["test ia/io"] = (
            sum(self.test_ia_io) / metadata["tuples test"]["count"]
        )

        metadata["ea/eo/ia/io"]["pred ea/eo"] = (
            sum(self.pred_ea_eo) / metadata["tuples predicted"]["count"]
        )
        metadata["ea/eo/ia/io"]["pred ea/io"] = (
            sum(self.pred_ea_io) / metadata["tuples predicted"]["count"]
        )
        metadata["ea/eo/ia/io"]["pred ia/eo"] = (
            sum(self.pred_ia_eo) / metadata["tuples predicted"]["count"]
        )
        metadata["ea/eo/ia/io"]["pred ia/io"] = (
            sum(self.pred_ia_io) / metadata["tuples predicted"]["count"]
        )

        metadata["span len"] = {}

        metadata["span len"]["valid aspect"] = mean(self.valid_aspect_span_len_list)
        metadata["span len"]["expl opinion"] = mean(self.expl_opinion_span_len_list)
        if self.tuple_len == len(TERM_LIST):
            metadata["span len"]["impl opinion"] = mean(self.impl_opinion_span_len_list)

        metadata["span len"]["test valid aspect"] = mean(self.test_valid_aspect_span_len_list)
        metadata["span len"]["test expl opinion"] = mean(self.test_expl_opinion_span_len_list)
        if self.tuple_len == len(TERM_LIST):
            metadata["span len"]["test impl opinion"] = mean(self.test_impl_opinion_span_len_list)

        metadata["span len"]["pred valid aspect"] = mean(self.pred_valid_aspect_span_len_list)
        metadata["span len"]["pred expl opinion"] = mean(self.pred_expl_opinion_span_len_list)
        if self.tuple_len == len(TERM_LIST):
            metadata["span len"]["pred impl opinion"] = mean(self.pred_impl_opinion_span_len_list)

        metadata["sentiment"] = {}
        metadata["sentiment"]["positive"] = sum(self.pos) / metadata["tuples"]["count"]
        metadata["sentiment"]["negative"] = sum(self.neg) / metadata["tuples"]["count"]
        metadata["sentiment"]["neutral"] = sum(self.neu) / metadata["tuples"]["count"]

        metadata["sentiment"]["test positive"] = (
            sum(self.test_pos) / metadata["tuples test"]["count"]
        )
        metadata["sentiment"]["test negative"] = (
            sum(self.test_neg) / metadata["tuples test"]["count"]
        )
        metadata["sentiment"]["test neutral"] = (
            sum(self.test_neu) / metadata["tuples test"]["count"]
        )

        metadata["sentiment"]["pred positive"] = (
            sum(self.pred_pos) / metadata["tuples predicted"]["count"]
        )
        metadata["sentiment"]["pred negative"] = (
            sum(self.pred_neg) / metadata["tuples predicted"]["count"]
        )
        metadata["sentiment"]["pred neutral"] = (
            sum(self.pred_neu) / metadata["tuples predicted"]["count"]
        )

        metadata["sentiment"]["polarities/example"] = {}
        metadata["sentiment"]["test polarities/example"] = {}
        metadata["sentiment"]["pred polarities/example"] = {}

        metadata["sentiment"]["polarities/example"]["mean"] = mean(self.polarities)
        metadata["sentiment"]["test polarities/example"]["mean"] = mean(
            self.test_polarities
        )
        metadata["sentiment"]["pred polarities/example"]["mean"] = mean(
            self.pred_polarities
        )

        metadata["sentiment"]["polarities/example"]["stdev"] = stdev(self.polarities)
        metadata["sentiment"]["test polarities/example"]["stdev"] = stdev(
            self.test_polarities
        )
        metadata["sentiment"]["pred polarities/example"]["stdev"] = stdev(
            self.pred_polarities
        )

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
        scores["precision"] = sum(self.partial_precision) / len(self.partial_precision)
        scores["recall"] = sum(self.partial_recall) / len(self.partial_recall)
        scores["f1-score"] = sum(self.partial_f1) / len(self.partial_f1)
        scores["macro IoU"] = sum(self.partial_macro_IoU) / len(self.partial_macro_IoU)
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
            scores_for_rev_i["metadata"]["num_predicted"] = self.pred_tuples_list[i]
            scores_for_rev_i["metadata"]["num_true"] = self.test_tuples_list[i]

            # if args.task == "acos-extend":
            #     scores_for_rev_i[
            #         f"total {TERM_LIST[OPINION_IDX]} micro IoU"
            #     ] = self.partial_micro_IoU[OPINION_IDX][i]
            #     scores_for_rev_i[f"indirect {TERM_LIST[OPINION_IDX]} micro IoU"] = {
            #     }
            # else:
            #     scores_for_rev_i["exact micro IoU"] = self.micro_IoU[i]
            #     sum_partial_micro_IoU = 0
            #     for j, term in enumerate(TERM_LIST[: self.tuple_len]):
            #         scores_for_rev_i[f"{term} micro IoU"] = self.partial_micro_IoU[j][i]
            #         sum_partial_micro_IoU += self.partial_micro_IoU[j][i]

            #     # scores_for_rev_i["avg partial micro IoU"] = sum_partial_micro_IoU / len(
            #     #     self.partial_micro_IoU
            #     # )

            #     for combo_idx, combo in enumerate(self.combos):
            #         combo_str = "-".join([TERM_LIST[idx] for idx in combo])
            #         scores_for_rev_i[f"{combo_str} micro IoU"] = self.combo_micro_IoU[
            #             combo_idx
            #         ][i]

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

    def get_scores(self, token_limit=inf, tuple_limit=inf):
        if token_limit != inf or tuple_limit != inf:
            self.__remove_examples(self.reviews, token_limit, tuple_limit)

        self.calc_metadata()
        self.calc_exact_scores()
        self.calc_partial_scores()
        self.calc_combo_scores()

        scores = {}

        scores["metadata"] = self.get_metadata()

        # if args.task == "acos-extend":
        #     scores[f"total {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
        #         OPINION_IDX
        #     )
        # else:
        #     scores["exact"] = self.get_exact_scores()
        #     for i, term in enumerate(TERM_LIST[: self.tuple_len]):
        #         scores[term] = self.get_partial_scores(i)
        #     # scores["avg partial"] = self.get_avg_partial_scores()

        #     for combo_idx, combo in enumerate(self.combos):
        #         combo_str = "-".join([TERM_LIST[idx] for idx in combo])
        #         scores[combo_str] = self.get_combo_scores(combo_idx)

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
        scores[f"exact (only indirect {TERM_LIST[OPINION_IDX]})"] = (
            self.get_exact_scores()
        )
        if args.task == "acos-extract":
            scores[f"indirect {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX, span=False
            )
        else:
            scores[f"indirect {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX
            )
        for i, review in enumerate(scores["reviews"]):
            review[f"indirect {TERM_LIST[OPINION_IDX]} micro IoU"] = (
                self.partial_micro_IoU[OPINION_IDX][i]
            )

        if args.task == "acos-extract" or args.task == "acosi-extract":
            self.true_outputs = full_true_outputs
            self.pred_outputs = full_pred_outputs

            self.__remove_opinions(keep_opinion_type="direct")
            self.calc_exact_scores()
            self.calc_partial_scores()
            scores[f"exact (only direct {TERM_LIST[OPINION_IDX]})"] = (
                self.get_exact_scores()
            )
            scores[f"direct {TERM_LIST[OPINION_IDX]}"] = self.get_partial_scores(
                OPINION_IDX
            )
            for i, review in enumerate(scores["reviews"]):
                review[f"direct {TERM_LIST[OPINION_IDX]} micro IoU"] = (
                    self.partial_micro_IoU[OPINION_IDX][i]
                )

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

    def __remove_examples(self, reviews, token_limit=inf, tuple_limit=inf):
        true_outputs_remove_over_limit = []
        pred_outputs_remove_over_limit = []
        for review, true_output, pred_output in zip(
            reviews, self.true_outputs, self.pred_outputs
        ):
            if len(review.split()) <= token_limit and len(true_output) <= tuple_limit:
                true_outputs_remove_over_limit.append(true_output)
                pred_outputs_remove_over_limit.append(pred_output)
        self.pred_outputs = pred_outputs_remove_over_limit
        self.true_outputs = true_outputs_remove_over_limit

    def __remove_opinions(self, keep_opinion_type="indirect"):
        true_outputs_remove_direct_opinion = []
        pred_outputs_remove_direct_opinion = []
        for true_output, pred_output in zip(self.true_outputs, self.pred_outputs):
            list_of_tuples_true = []
            for true_quint in true_output:
                if args.task == "acos-extend" or args.task == "acosi-extract":
                    if (
                        keep_opinion_type == "indirect"
                        and true_quint[IMPLICIT_IND_IDX] == "indirect"
                    ):
                        list_of_tuples_true.append(true_quint)
                    elif (
                        keep_opinion_type == "direct"
                        and true_quint[IMPLICIT_IND_IDX] == "direct"
                    ):
                        list_of_tuples_true.append(true_quint)
                else:
                    if (
                        keep_opinion_type == "indirect"
                        and true_quint[OPINION_IDX] == "NULL"
                    ):
                        list_of_tuples_true.append(true_quint)
                    elif (
                        keep_opinion_type == "direct"
                        and true_quint[OPINION_IDX] != "NULL"
                    ):
                        list_of_tuples_true.append(true_quint)

            list_of_tuples_pred = []
            for pred_quint in pred_output:
                if args.task == "acos-extend" or args.task == "acosi-extract":
                    if (
                        keep_opinion_type == "indirect"
                        and pred_quint[IMPLICIT_IND_IDX] == "indirect"
                    ):
                        list_of_tuples_pred.append(pred_quint)
                    elif (
                        keep_opinion_type == "direct"
                        and pred_quint[IMPLICIT_IND_IDX] == "direct"
                    ):
                        list_of_tuples_pred.append(pred_quint)
                else:
                    if (
                        keep_opinion_type == "indirect"
                        and pred_quint[OPINION_IDX] == "NULL"
                    ):
                        list_of_tuples_pred.append(pred_quint)
                    elif (
                        keep_opinion_type == "direct"
                        and pred_quint[OPINION_IDX] != "NULL"
                    ):
                        list_of_tuples_pred.append(pred_quint)

            true_outputs_remove_direct_opinion.append(list_of_tuples_true)
            pred_outputs_remove_direct_opinion.append(list_of_tuples_pred)

        self.pred_outputs = pred_outputs_remove_direct_opinion
        self.true_outputs = true_outputs_remove_direct_opinion

    ######################## METADATA FUNCTIONS ########################
    def __set_test_ea_eo_ia_io(self):
        self.test_ea_eo = deepcopy(self.ea_eo)
        self.test_ea_io = deepcopy(self.ea_io)
        self.test_ia_eo = deepcopy(self.ia_eo)
        self.test_ia_io = deepcopy(self.ia_io)

    def __set_test_sentiment(self):
        self.test_pos = deepcopy(self.pos)
        self.test_neg = deepcopy(self.neg)
        self.test_neu = deepcopy(self.neu)

    def __set_test_span_len(self):
        self.test_valid_aspect_span_len_list = deepcopy(self.valid_aspect_span_len_list)
        self.test_impl_opinion_span_len_list = deepcopy(self.impl_opinion_span_len_list)
        self.test_expl_opinion_span_len_list = deepcopy(self.expl_opinion_span_len_list)
        
    def __append_zero(self, append_true=False, append_pred=False):
        if append_true:
            self.ea_eo.append(0)
            self.ea_io.append(0)
            self.ia_eo.append(0)
            self.ia_io.append(0)
            self.pos.append(0)
            self.neg.append(0)
            self.neu.append(0)

        if append_pred:
            self.pred_ea_eo.append(0)
            self.pred_ea_io.append(0)
            self.pred_ia_eo.append(0)
            self.pred_ia_io.append(0)
            self.pred_pos.append(0)
            self.pred_neg.append(0)
            self.pred_neu.append(0)

    def __accum_ea_eo_ia_io_span_len_sent(self, output, accum_true=False, accum_pred=False):
        self.__append_zero(append_true=accum_true, append_pred=accum_pred)
        for quint in output:
            if (quint[ASPECT_IDX] == "NULL" and quint[OPINION_IDX] == "NULL") or (
                self.tuple_len == len(TERM_LIST)
                and quint[ASPECT_IDX] == "NULL"
                and quint[IMPLICIT_IND_IDX] == "indirect"
            ):
                accum_ea_eo_ia_io_sent(self.ia_io, self.pred_ia_io, accum_true, accum_pred)
            elif quint[ASPECT_IDX] == "NULL":
                accum_ea_eo_ia_io_sent(self.ia_eo, self.pred_ia_eo, accum_true, accum_pred)
            elif (quint[OPINION_IDX] == "NULL") or (
                self.tuple_len == len(TERM_LIST)
                and quint[IMPLICIT_IND_IDX] == "indirect"
            ):
                accum_ea_eo_ia_io_sent(self.ea_io, self.pred_ea_io, accum_true, accum_pred)
                accum_span_len(
                    quint[ASPECT_IDX],
                    self.valid_aspect_span_len_list,
                    self.pred_valid_aspect_span_len_list,
                    accum_true,
                    accum_pred,
                )
                if self.tuple_len == len(TERM_LIST):
                    accum_span_len(
                        quint[OPINION_IDX],
                        self.impl_opinion_span_len_list,
                        self.pred_impl_opinion_span_len_list,
                        accum_true,
                        accum_pred,
                    )
            else:
                accum_ea_eo_ia_io_sent(self.ea_eo, self.pred_ea_eo, accum_true, accum_pred)
                accum_span_len(
                    quint[ASPECT_IDX],
                    self.valid_aspect_span_len_list,
                    self.pred_valid_aspect_span_len_list,
                    accum_true,
                    accum_pred,
                )
                accum_span_len(
                    quint[OPINION_IDX],
                    self.expl_opinion_span_len_list,
                    self.pred_expl_opinion_span_len_list,
                    accum_true,
                    accum_pred,
                )

            if quint[SENTIMENT_IDX] == "positive":
                accum_ea_eo_ia_io_sent(self.pos, self.pred_pos, accum_true, accum_pred)
            elif quint[SENTIMENT_IDX] == "negative":
                accum_ea_eo_ia_io_sent(self.neg, self.pred_neg, accum_true, accum_pred)
            else:
                accum_ea_eo_ia_io_sent(self.neu, self.pred_neu, accum_true, accum_pred)

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
        self.ea_eo = []
        self.ea_io = []
        self.ia_eo = []
        self.ia_io = []

        self.test_ea_eo = []
        self.test_ea_io = []
        self.test_ia_eo = []
        self.test_ia_io = []

        self.pred_ea_eo = []
        self.pred_ea_io = []
        self.pred_ia_eo = []
        self.pred_ia_io = []

    def __init_sentiment(self):
        self.pos = []
        self.neg = []
        self.neu = []

        self.test_pos = []
        self.test_neg = []
        self.test_neu = []

        self.pred_pos = []
        self.pred_neg = []
        self.pred_neu = []

        self.polarities = []
        self.test_polarities = []
        self.pred_polarities = []

    def __init_span_len(self):
        self.valid_aspect_span_len_list = []
        self.test_valid_aspect_span_len_list = []
        self.pred_valid_aspect_span_len_list = []

        self.expl_opinion_span_len_list = []
        self.test_expl_opinion_span_len_list = []
        self.pred_expl_opinion_span_len_list = []

        self.impl_opinion_span_len_list = []
        self.test_impl_opinion_span_len_list = []
        self.pred_impl_opinion_span_len_list = []
        
scores = {}

if args.mvp_output:
    evaluate_mvp_outputs = Evaluator(get_mvp_output, category_file=args.category_file)
    scores = evaluate_mvp_outputs.get_scores()
elif args.llm_output:
    evaluate_llm_outputs = Evaluator(get_llm_output, category_file=args.category_file)
    scores = evaluate_llm_outputs.get_scores()
elif args.t5_output:
    evaluate_t5_outputs = Evaluator(get_t5_output)
    scores = evaluate_t5_outputs.get_scores()
elif args.gen_scl_nat_output:
    evaluate_gen_scl_nat_outputs = Evaluator(
        get_gen_scl_nat_output, category_file=args.category_file, task=args.task
    )
    scores = evaluate_gen_scl_nat_outputs.get_scores()

with open(args.output_file, "w") as file:
    json.dump(scores, file, indent=4)
