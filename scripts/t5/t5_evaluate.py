import numpy as np
import pickle
import json
import re
from string import punctuation
from collections import Counter

def clean_punctuation(words):
    punc = re.compile(f"[{re.escape(punctuation)}]")
    words = punc.sub(" \\g<0> ", words)

    # remove extra spaces
    words = words.strip()
    words = " ".join(words.split())
    return words


def extract_spans(seq, seq_type):
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

            if at.lower() == "it":
                at = "NULL"

            at = clean_punctuation(at)
            ot = clean_punctuation(ot)

            quints.append((at, ac, sp, ot, ie))

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


def get_t5_output(pkl_file, category_file, num_path=5):
    with open(pkl_file, "rb") as f:
        targets = pickle.load(f)

    outputs = []
    for i in range(len(targets)):
        all_quints = []
        all_quints.extend(extract_spans(seq=targets[i], seq_type="pred"))
        outputs.append(all_quints)

    return outputs
