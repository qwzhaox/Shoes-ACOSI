import pickle
from utils.evaluate_utils import extract_spans


def get_t5_output(pkl_file):
    with open(pkl_file, "rb") as f:
        targets = pickle.load(f)

    outputs = []
    for i in range(len(targets)):
        all_quints = []
        all_quints.extend(
            extract_spans(seq=targets[i], seq_type="pred", output_type="t5")
        )
        outputs.append(all_quints)

    return outputs
