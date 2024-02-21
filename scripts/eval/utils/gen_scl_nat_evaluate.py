import json
from utils.evaluate_utils import clean_punctuation


def is_weird_quadruple(quad):
    ac, at, sp, ot = quad
    return ac == "" and at == "NULL" and sp == "" and ot == "NULL"

def get_gen_scl_nat_output(model_output_file, category_file, task):
    with open(model_output_file, "rb") as f:
        data = json.load(f)
        examples = data["examples"]
        all_labels_pred = [e["labels_pred"] for e in examples]

    with open(category_file, "rb") as f:
        category_dict = json.load(f)

    outputs = []
    for labels_pred in all_labels_pred:
        output_quints = []
        for quad in labels_pred:
            ac, at, sp, ot = quad

            if is_weird_quadruple(quad):
                # if task == "acosi-extract":
                #     output_quints.append((at, ac, sp, ot, ""))
                # else:
                #     output_quints.append((at, ac, sp, ot))
                continue

            try:
                ac = category_dict[ac].lower().strip()
            except:
                print("Category not found: ", ac)
                ac = ac.lower().strip()

            at = clean_punctuation(at)
            ot = clean_punctuation(ot)
            sp = sp.lower().strip()

            if at == "it" or at == "null":
                at = "NULL"
            
            if task == "acosi-extract":
                ie = ot[ot.rfind(" "):].lower().strip()
                if ie != "direct" and ie != "indirect":
                    ie = "direct"
                else:
                    ot = ot[:ot.rfind(" ")].strip()
                quint = (at, ac, sp, ot, ie)
            else:
                quint = (at, ac, sp, ot)

            output_quints.append(quint)

        outputs.append(output_quints)

    return outputs
