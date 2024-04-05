import re
import numpy as np
from string import punctuation
from itertools import combinations
from utils.metrics_util import indexify_spans

ASPECT_IDX = 0
CATEGORY_IDX = 1
SENTIMENT_IDX = 2
OPINION_IDX = 3
IMPLICIT_IND_IDX = 4

IDX_LIST = [ASPECT_IDX, CATEGORY_IDX, SENTIMENT_IDX, OPINION_IDX, IMPLICIT_IND_IDX]
TERM_LIST = ["aspect", "category", "sentiment", "opinion", "implicit_indicator"]

NUM_SPANS = 2
SPAN_IDX = {ASPECT_IDX: 0, OPINION_IDX: 1}

def clean_punctuation(words):
    punc = re.compile(f"[{re.escape(punctuation)}]")
    words = punc.sub(" \\g<0> ", words)

    # remove extra spaces
    words = words.strip()
    words = " ".join(words.split())
    return words


def extract_spans_para(seq, seq_type, output_type, sentiment_dict={}, category_dict={}):
    quints = []
    sents = [s.strip() for s in seq.split("[SSEP]")]
    for s in sents:
        try:
            tok_list = ["[C]", "[S]", "[A]", "[O]", "[I]"]

            for tok in tok_list:
                if tok not in s:
                    s += " {} null".format(tok)
            index_ac = s.index("[C]")
            index_sp = s.index("[S]")
            index_at = s.index("[A]")
            index_ot = s.index("[O]")
            index_ie = s.index("[I]")

            combined_list = [index_ac, index_sp, index_at, index_ot, index_ie]
            arg_index_list = list(np.argsort(combined_list))

            result = []
            for i in range(len(combined_list)):
                start = combined_list[i] + 4
                sort_index = arg_index_list.index(i)
                if sort_index < 4:
                    next_ = arg_index_list[sort_index + 1]
                    re = s[start : combined_list[next_]]
                else:
                    re = s[start:]
                result.append(re.strip())

            ac, sp, at, ot, ie = result

            if output_type == "mvp":
                if at.lower() == "it":
                    at = "NULL"
                sp = sentiment_dict[sp]
                ac = category_dict[ac]

            at = clean_punctuation(at)
            ot = clean_punctuation(ot)

            if at.lower() == "null" or at.lower() == "implicit":
                at = "NULL"
            if ot.lower() == "null" or ot.lower() == "implicit":
                ot = "NULL"

            quints.append((at, ac.lower(), sp, ot, ie))

        except KeyError:
            ac, at, sp, ot, ie = "", "", "", "", ""
        except ValueError:
            try:
                print(f"In {seq_type} seq, cannot decode: {s}")
                pass
            except UnicodeEncodeError:
                print(f"In {seq_type} seq, a string cannot be decoded")
                pass
            ac, at, sp, ot, ie = "", "", "", "", ""

    return quints


def process_dataset(dataset):
    reviews = [x.split("####")[0] for x in dataset]

    raw_true_outputs = [eval(x.split("####")[1]) for x in dataset]
    true_outputs = []
    for output in raw_true_outputs:
        list_of_tuples = []
        for quint in output:
            if (
                quint[ASPECT_IDX].lower() == "null"
                or quint[ASPECT_IDX].lower() == "implicit"
            ):
                quint[ASPECT_IDX] = "NULL"
            if (
                quint[OPINION_IDX].lower() == "null"
                or quint[OPINION_IDX].lower() == "implicit"
            ):
                quint[OPINION_IDX] = "NULL"
            list_of_tuples.append(tuple(quint))
        true_outputs.append(list_of_tuples)
    return reviews, true_outputs


def accumulate_ea_eo_ia_io_sent(output, nums, tuple_len):
    num_ea_eo, num_ea_io, num_ia_eo, num_ia_io, pos, neg, neu = nums
    for quint in output:
        if (quint[ASPECT_IDX] == "NULL" and quint[OPINION_IDX] == "NULL") or \
            (tuple_len == len(TERM_LIST) and quint[ASPECT_IDX] == "NULL" and quint[IMPLICIT_IND_IDX] == "indirect"):
            num_ia_io += 1
        elif quint[ASPECT_IDX] == "NULL":
            num_ia_eo += 1
        elif (quint[OPINION_IDX] == "NULL") or \
                (tuple_len == len(TERM_LIST) and quint[IMPLICIT_IND_IDX] == "indirect"):
            num_ea_io += 1
        else:
            num_ea_eo += 1

        if quint[SENTIMENT_IDX] == "positive":
            pos += 1
        elif quint[SENTIMENT_IDX] == "negative":
            neg += 1
        else:
            neu += 1

    return num_ea_eo, num_ea_io, num_ia_eo, num_ia_io, pos, neg, neu


def get_combos(tuple_len):
    combos = []
    for i in range(2, tuple_len):
        combos += list(combinations(IDX_LIST[:tuple_len], i))
    return combos


def indexify_outputs(reviews, outputs, idx):
    return [
        indexify_spans(annotation, review, idx)
        for review, annotation in zip(reviews, outputs)
    ]


def listify_outputs(outputs, idx):
    return [[quint[idx] for quint in annotation] for annotation in outputs]


def comboify_outputs(outputs, combo):
    new_outputs = []
    for annotation in outputs:
        new_annotation = []
        for quint in annotation:
            new_tuple = []
            for idx in combo:
                new_tuple.append(quint[idx])
            new_annotation.append(tuple(new_tuple))
        new_outputs.append(new_annotation)              
    return new_outputs


def get_precision_recall_fl_IoU(pred_outputs, true_outputs):
    n_tp, n_fp, n_fn, n_union = 0, 0, 0, 0
    num_empty = 0

    numerical_micro_IoU = []
    micro_IoU = []
    for i in range(len(pred_outputs)):
        pred_set = set(pred_outputs[i])
        true_set = set(true_outputs[i])

        num_intersection = len(pred_set & true_set)
        num_union = len(pred_set | true_set)

        n_tp += num_intersection
        n_union += num_union

        n_fp += len(pred_set - true_set)
        n_fn += len(true_set - pred_set)

        if len(pred_set) == 0 and len(true_set) == 0:
            num_empty += 1

            numerical_micro_IoU.append(0)
            micro_IoU.append(None)
        else:
            assert num_union != 0

            numerical_micro_IoU.append(float(num_intersection) / num_union)
            micro_IoU.append(float(num_intersection) / num_union)

    precision = float(n_tp) / (n_tp + n_fp) if n_tp + n_fp != 0 else 0
    recall = float(n_tp) / (n_tp + n_fn) if n_tp + n_fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    macro_IoU = float(n_tp) / n_union if n_union != 0 else 0
    avg_micro_IoU = (
        sum(numerical_micro_IoU) / (len(numerical_micro_IoU) - num_empty)
        if len(numerical_micro_IoU) - num_empty != 0
        else 0
    )
    return precision, recall, f1, macro_IoU, micro_IoU, avg_micro_IoU
