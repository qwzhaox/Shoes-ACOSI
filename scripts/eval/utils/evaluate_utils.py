import re
import numpy as np
from string import punctuation
from utils.metrics_util import indexify_spans

ASPECT_IDX = 0
CATEGORY_IDX = 1
SENTIMENT_IDX = 2
OPINION_IDX = 3
IMPLICIT_IND_IDX = 4

IDX_LIST = [ASPECT_IDX, CATEGORY_IDX, SENTIMENT_IDX, OPINION_IDX, IMPLICIT_IND_IDX]
TERM_LIST = ["aspect", "category", "sentiment", "opinion", "implicit indicator"]


def clean_punctuation(words):
    punc = re.compile(f"[{re.escape(punctuation)}]")
    words = punc.sub(" \\g<0> ", words)

    # remove extra spaces
    words = words.strip()
    words = " ".join(words.split())
    return words


def extract_spans(seq, seq_type, output_type, sentiment_dict={}, category_dict={}):
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
                    at = "null"
                sp = sentiment_dict[sp]
                ac = category_dict[ac]

            if at.lower() == "null" or at.lower() == "implicit":
                at = "NULL"
            if ot.lower() == "null" or ot.lower() == "implicit":
                ot = "NULL"

            at = clean_punctuation(at)
            ot = clean_punctuation(ot)

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


def indexify_outputs(reviews, outputs, idx):
    return [
        indexify_spans(annotation, review, idx)
        for review, annotation in zip(reviews, outputs)
    ]


def listify_outputs(outputs, idx):
    return [[quint[idx] for quint in annotation] for annotation in outputs]


def get_precision_recall_fl_IoU(pred_outputs, true_outputs):
    n_tp, n_fp, n_fn, n_union = 0, 0, 0, 0

    local_IoU = []
    for i in range(len(pred_outputs)):
        pred_set = set(pred_outputs[i])
        true_set = set(true_outputs[i])

        num_intersection = len(pred_set & true_set)
        num_union = len(pred_set | true_set)

        n_tp += num_intersection
        n_union += num_union

        n_fp += len(pred_set - true_set)
        n_fn += len(true_set - pred_set)

        local_IoU.append(float(num_intersection) / num_union if num_union != 0 else 0)

    precision = float(n_tp) / (n_tp + n_fp) if n_tp + n_fp != 0 else 0
    recall = float(n_tp) / (n_tp + n_fn) if n_tp + n_fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    global_IoU = float(n_tp) / n_union if n_union != 0 else 0
    avg_local_IoU = sum(local_IoU) / len(local_IoU)
    return precision, recall, f1, global_IoU, local_IoU, avg_local_IoU
