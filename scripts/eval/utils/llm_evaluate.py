import pickle
import json
from utils.evaluate_utils import ASPECT_IDX, CATEGORY_IDX, OPINION_IDX


def get_llm_output(model_output_file, category_file=None):
    with open(model_output_file, "rb") as file:
        llm_outputs = pickle.load(file)
    
    with open(category_file, "r") as file:
        category_dict = json.load(file)

    for output in llm_outputs:
        for i, quint in enumerate(output):
            quint = list(quint)
            for j, _ in enumerate(quint):
                if type(quint[j]) == str:
                    quint[j].lower()

            if quint[ASPECT_IDX] == "it" or quint[ASPECT_IDX] == "null":
                quint[ASPECT_IDX] = "NULL"
            if quint[OPINION_IDX] == "it" or quint[OPINION_IDX] == "null":
                quint[OPINION_IDX] = "NULL"

            try:
                quint[CATEGORY_IDX] = category_dict[quint[CATEGORY_IDX]]
            except KeyError:
                print(f"Category not found: {quint[CATEGORY_IDX]}")
                continue

            quint = tuple(quint)
            output[i] = quint

    return llm_outputs
