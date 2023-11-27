from pickle import load


def get_llm_output(pkl_file):
    with open(pkl_file, "rb") as file:
        llm_outputs = load(file)

    return llm_outputs
