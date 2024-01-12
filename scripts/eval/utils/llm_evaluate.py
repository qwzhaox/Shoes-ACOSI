from pickle import load
from utils.evaluate_utils import ASPECT_IDX, OPINION_IDX


def get_llm_output(pkl_file):
    with open(pkl_file, "rb") as file:
        llm_outputs = load(file)

    for output in llm_outputs:
        for i, quint in enumerate(output):
            quint = list(quint)
            for j, _ in enumerate(quint):
                quint[j].lower()

            if quint[ASPECT_IDX] == "it" or quint[ASPECT_IDX] == "null":
                quint[ASPECT_IDX] = "NULL"
            if quint[OPINION_IDX] == "it" or quint[OPINION_IDX] == "null":
                quint[OPINION_IDX] = "NULL"

            quint = tuple(quint)
            output[i] = quint

    return llm_outputs
