import pickle
import json
from collections import Counter
from evaluate_utils import extract_spans

sentiment_dict = {"great": "positive", "ok": "neutral", "bad": "negative"}


def get_mvp_output(pkl_file, category_file, num_path=5):
    with open(pkl_file, "rb") as f:
        (outputs, targets, _) = pickle.load(f)
    with open(category_file, "rb") as f:
        global category_dict
        category_dict = json.load(f)

    targets = targets[::num_path]
    _outputs = outputs
    outputs = []
    for i in range(0, len(targets)):
        o_idx = i * num_path
        multi_outputs = _outputs[o_idx : o_idx + num_path]

        all_quints = []
        for s in multi_outputs:
            all_quints.extend(
                extract_spans(
                    seq=s,
                    seq_type="pred",
                    output_type="mvp",
                    sentiment_dict=sentiment_dict,
                    category_dict=category_dict,
                )
            )

        output_quints = []
        counter = dict(Counter(all_quints))
        for quint, count in counter.items():
            # keep freq >= num_path / 2
            if count >= len(multi_outputs) / 2:
                output_quints.append(quint)

        outputs.append(output_quints)

    return outputs
