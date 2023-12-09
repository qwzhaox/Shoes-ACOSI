from pickle import load


def get_llm_output(pkl_file):
    with open(pkl_file, "rb") as file:
        llm_outputs = load(file)

    for output in llm_outputs:
       for i, quint in enumerate(output):
            quint = tuple([x.lower() for x in quint])
            output[i] = quint

    return llm_outputs
