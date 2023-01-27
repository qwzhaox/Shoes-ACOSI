import argparse
import json
from itertools import combinations
from copy import deepcopy

ASPECT = 0
CATEGORY = 1
SENTIMENT = 2
OPINION = 3
IMPL_EXPL = 4

ACOSI = ["Aspect", "Category", "Sentiment", "Opinion", "Implicit/Explict"]
ACOSI_IDX = [ASPECT, CATEGORY, SENTIMENT, OPINION, IMPL_EXPL]

COMBOS = []
for i in range(1, len(ACOSI_IDX)):
    COMBOS += list(combinations(ACOSI_IDX, i))

NUM_QUAD_ELTS = 5

SIMILARITY_THRESHOLD = 0.50


def indexify(original, span):
    # TODO: more processing needed?
    orig_array = original.split(" ")
    span_array = span.split(" ")
    start_idx = -1
    while (orig_array[start_idx] != span_array[0]):
        start_idx += 1
        if start_idx >= len(orig_array):
            return [-1]

    end_idx = start_idx + len(span_array)
    num_span_array = [i for i in range(start_idx, end_idx)]
    return num_span_array


def indexify_spans(annotation, review, idx):
    for quad in annotation:
        # Needs to be tuple for overall conversion to set
        quad[idx] = tuple(indexify(review, quad[idx]))


def set_tuplify(list):
    new_set = set([tuple(sub_list) for sub_list in list])
    return new_set


def get_inter_union(set1, set2):
    return len(set1 & set2), len(set1 | set2)


def print_metric(metric, inter, union):
    print(
        f"{metric}: {inter}/{union}, {(inter/union)*100:.2f}%")


def get_exact_inter_union(annot1, annot2):
    set_annot1 = set_tuplify(annot1)
    set_annot2 = set_tuplify(annot2)
    inter, union = get_inter_union(set_annot1, set_annot2)
    return inter, union, set_annot1, set_annot2


def get_elt_inter_union(idx, annotation1, annotation2):
    set1 = set([quad[idx] for quad in annotation1])
    set2 = set([quad[idx] for quad in annotation2])

    return get_inter_union(set1, set2)


def get_span_inter_union(idx, annotation1, annotation2):
    span1 = [quad[idx] for quad in annotation1]
    span2 = [quad[idx] for quad in annotation2]

    set_span1 = set(
        [idx for span in span1 for idx in span])
    set_span2 = set(
        [idx for span in span2 for idx in span])

    return get_inter_union(set_span1, set_span2)


def get_quad_similarity(quad1, quad2):
    list_quad1 = list(quad1)
    list_quad2 = list(quad2)
    opinion1 = list_quad1.pop(OPINION)
    opinion2 = list_quad2.pop(OPINION)
    aspect1 = list_quad1.pop(ASPECT)
    aspect2 = list_quad2.pop(ASPECT)

    num_agreements = [elt[0] == elt[1] for elt in zip(list_quad1, list_quad2)]
    inter_opinion, union_opinion = get_inter_union(
        set(opinion1), set(opinion2))
    inter_aspect, union_aspect = get_inter_union(set(aspect1), set(aspect2))
    sum_agreements = sum(num_agreements)
    inter_union_opinion = inter_opinion/union_opinion
    inter_union_aspect = inter_aspect/union_aspect

    return (sum_agreements + inter_union_opinion + inter_union_aspect)/NUM_QUAD_ELTS


def get_adj_inter_union(set_annot1, set_annot2, exact_inter, exact_union):
    dif1 = list(set_annot1 - set_annot2)
    dif2 = list(set_annot2 - set_annot1)

    set_list = []
    inter = exact_inter
    union = exact_union

    for i in range(len(dif1)):
        for j in range(len(dif2)):
            if {dif1[i], dif2[j]} not in set_list:
                similarity = get_quad_similarity(dif1[i], dif2[j])
                if similarity > SIMILARITY_THRESHOLD:
                    set_list.append({dif1[i], dif2[j]})
                    inter += similarity
                    union -= similarity

    return round(inter, 2), round(union, 2)


def quad_excl(quad, exclusions):
    new_quad = [v for i, v in enumerate(quad) if i not in exclusions]
    quad.clear()
    quad.extend(new_quad)


def get_incl_elts(list1, exclusions):
    return [v for i, v in enumerate(list1) if i not in exclusions]


def get_excl_elts(list1, exclusions):
    return [v for i, v in enumerate(list1) if i in exclusions]


def exclude(annot1, annot2, exclusions):
    annot1_cp = deepcopy(annot1)
    annot2_cp = deepcopy(annot2)
    for quad in annot1_cp:
        quad_excl(quad, exclusions)
    for quad in annot2_cp:
        quad_excl(quad, exclusions)
    inter, union, _, _ = get_exact_inter_union(annot1_cp, annot2_cp)
    return inter, union


def perform_exclusions(annot1, annot2):
    print("\nExact match w/ exclusions:")
    num_excl = 0
    for exclusions in COMBOS:
        if num_excl != len(exclusions):
            num_excl = len(exclusions)
            print(f"\nExclude: {num_excl}")
        excl_list = get_excl_elts(ACOSI, exclusions)
        incl_list = get_incl_elts(ACOSI, exclusions)
        print(
            f"{incl_list} ({excl_list} excluded):")
        inter, union = exclude(annot1, annot2, exclusions)
        print_metric(
            f"\tIoU match", inter, union)


# ...
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file",
    help="Input json file",
    required=True
)

args = parser.parse_args()

f = open(args.input_file)
review_data = json.load(f)

for idx in range(len(review_data)):
    review = review_data[idx]["review"]
    print(
        f"######################################## REVIEW {idx} ###########################################\n")
    annot1 = review_data[idx]["annotations"][0]["annotation"]
    annot2 = review_data[idx]["annotations"][1]["annotation"]

    # indexify opinions and spans
    indexify_spans(annot1, review, OPINION)
    indexify_spans(annot2, review, OPINION)
    indexify_spans(annot1, review, ASPECT)
    indexify_spans(annot2, review, ASPECT)

    # Delta
    print(f"Delta: {abs(len(annot1) - len(annot2))}\n")

    # Aspect
    aspect_inter, aspect_union = get_span_inter_union(
        ASPECT, annot1, annot2)
    print_metric(ACOSI[ASPECT], aspect_inter, aspect_union)

    # Category
    cat_inter, cat_union = get_elt_inter_union(CATEGORY, annot1, annot2)
    print_metric(ACOSI[CATEGORY], cat_inter, cat_union)

    # Sentiment
    sent_inter, sent_union = get_elt_inter_union(SENTIMENT, annot1, annot2)
    print_metric(ACOSI[SENTIMENT], sent_inter, sent_union)

    # Opinion
    op_inter, op_union = get_span_inter_union(
        OPINION, annot1, annot2)
    print_metric(ACOSI[OPINION], op_inter, op_union)

    # Implicit/Explicit
    ie_inter, ie_outer = get_elt_inter_union(IMPL_EXPL, annot1, annot2)
    print_metric(ACOSI[IMPL_EXPL], ie_inter, ie_outer)

    # Exact Match
    exact_inter, exact_union, set_annot1, set_annot2 = \
        get_exact_inter_union(annot1, annot2)
    print_metric("\nExact match", exact_inter, exact_union)

    # Adjusted overall match (find a way to link annotations)
    adj_inter, adj_union = get_adj_inter_union(
        set_annot1, set_annot2, exact_inter, exact_union)
    print_metric("Adjusted match", adj_inter, adj_union)

    # Exact match when take away each column
    perform_exclusions(annot1, annot2)

    print()


f.close()