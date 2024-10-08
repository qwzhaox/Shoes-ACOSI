import argparse
import json
from itertools import combinations
from copy import deepcopy
from utils.metrics_util import indexify, indexify_spans

ASPECT = 0
CATEGORY = 1
SENTIMENT = 2
OPINION = 3
IMPL_EXPL = 4

ACOSI = ["Aspect", "Category", "Sentiment", "Opinion", "Implicit/Explicit"]
A = ["aspect", "category", "sentiment", "opinion", "impl_expl"]
ACOSI_IDX = [ASPECT, CATEGORY, SENTIMENT, OPINION, IMPL_EXPL]

COMBOS = []
for i in range(1, len(ACOSI_IDX)):
    COMBOS += list(combinations(ACOSI_IDX, i))

NUM_QUAD_ELTS = 5

SIMILARITY_THRESHOLD = 0.50


def set_tuplify(list_in):
    new_set = set([tuple(sub_list) for sub_list in list_in])
    return new_set


def get_inter_union(set1, set2):
    return len(set1 & set2), len(set1 | set2)


def process_metric(metric, inter, union):
    try:
        output_list.append(f"{metric}: {inter}/{union}, {(inter/union)*100:.2f}%")
        return round(inter / union, 4)  # Rounded--can remove if needed
    except:
        output_list.append(f"{metric}: {0}, {0}%")
        return round(0, 4)  # Rounded--can remove if needed


def stringify_span_tuples(annot):
    for quad in annot:
        for idx, elt in enumerate(quad):
            if isinstance(elt, tuple):
                quad[idx] = " ".join([str(i) for i in elt])
            elif isinstance(elt, list):
                quad[idx] = " ".join([str(i) for i in elt])


def get_exact_inter_union(annot1, annot2):
    annot1_cp = deepcopy(annot1)
    stringify_span_tuples(annot1_cp)

    annot2_cp = deepcopy(annot2)
    stringify_span_tuples(annot2_cp)

    set_annot1 = set_tuplify(annot1_cp)
    set_annot2 = set_tuplify(annot2_cp)
    inter, union = get_inter_union(set_annot1, set_annot2)
    return inter, union, set_annot1, set_annot2


def get_elt_inter_union(idx, annotation1, annotation2):
    set1 = set([quad[idx] for quad in annotation1])
    set2 = set([quad[idx] for quad in annotation2])

    return get_inter_union(set1, set2)


def get_span_inter_union(idx, annotation1, annotation2):
    span1 = [quad[idx] for quad in annotation1]
    span2 = [quad[idx] for quad in annotation2]

    set_span1 = set([idx for span in span1 for idx in span])
    set_span2 = set([idx for span in span2 for idx in span])

    return get_inter_union(set_span1, set_span2)


def get_quad_similarity(quad1, quad2):
    list_quad1 = list(quad1)
    list_quad2 = list(quad2)
    opinion1 = list_quad1.pop(OPINION)
    opinion2 = list_quad2.pop(OPINION)
    aspect1 = list_quad1.pop(ASPECT)
    aspect2 = list_quad2.pop(ASPECT)

    num_agreements = [elt[0] == elt[1] for elt in zip(list_quad1, list_quad2)]
    inter_opinion, union_opinion = get_inter_union(set(opinion1), set(opinion2))
    inter_aspect, union_aspect = get_inter_union(set(aspect1), set(aspect2))
    sum_agreements = sum(num_agreements)
    inter_union_opinion = inter_opinion / union_opinion
    inter_union_aspect = inter_aspect / union_aspect

    return (sum_agreements + inter_union_opinion + inter_union_aspect) / NUM_QUAD_ELTS


def get_adj_inter_union(set_annot1, set_annot2, exact_inter, exact_union):
    dif1 = list(set_annot1 - set_annot2)
    dif2 = list(set_annot2 - set_annot1)

    already_linked_list = []
    inter = exact_inter
    union = exact_union

    for i in range(len(dif1)):
        for j in range(len(dif2)):
            if (
                dif1[i] not in already_linked_list
                and dif2[j] not in already_linked_list
            ):
                similarity = get_quad_similarity(dif1[i], dif2[j])
                if similarity > SIMILARITY_THRESHOLD:
                    already_linked_list.append(dif1[i])
                    already_linked_list.append(dif2[j])
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


def process_exclusions(annot1, annot2):
    output_list.append("\nExact match w/ exclusions:")
    num_excl = 0
    output_list.append(f"\nExclude: {num_excl}")

    exclusions_txt = "exclude_"
    key = exclusions_txt

    for exclusions in COMBOS:
        if num_excl != len(exclusions):
            key = f"{exclusions_txt}{num_excl+1}"
            num_excl = len(exclusions)
            output_list.append(f"\nExclude: {num_excl}")

        incl_list = get_incl_elts(A, exclusions)
        excl_list = get_excl_elts(A, exclusions)
        incl_list_print = get_incl_elts(ACOSI, exclusions)
        excl_list_print = get_excl_elts(ACOSI, exclusions)

        curr_key = f"{key}: {excl_list}"
        output_list.append(f"{incl_list_print}\nEXCLUDED: {excl_list_print}")

        inter, union = exclude(annot1, annot2, exclusions)
        review_metrics[curr_key] = process_metric(f"\tIoU match", inter, union)


def flatten_annot(annot):
    flat_string = "\n".join([" | ".join(quad) for quad in annot])
    return flat_string


# ...
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="input json file", required=True)
parser.add_argument(
    "-o", "--output_file",
    help="output json file",
    required=True,
)
parser.add_argument(
    "--verbose",
    action="store_true",
)

args = parser.parse_args()
verbose = args.verbose

with open(args.input_file) as f:
    review_data = json.load(f)

metrics_list = []
output_list = ["MEASURE USED: INTERSECTION / UNION\n"]

delta_total = 0
aspect_total = 0
category_total = 0
opinion_total = 0
sentiment_total = 0
impl_expl_total = 0
exact_total = 0
cat_quad_avg_total = 0
span_quad_avg_total = 0

excluded = 0

for idx in range(len(review_data)):
    review_metrics = {}
    review = review_data[idx]["review"]
    review_metrics["review"] = review
    review_metrics["p_name"] = review_data[idx]["p_name"]

    output_list.append(
        f"######################################## REVIEW {idx} ###########################################\n"
    )

    try:
        name1 = review_data[idx]["annotations"][1]["metadata"]["name"]
        name2 = review_data[idx]["annotations"][2]["metadata"]["name"]
        review_metrics["annotator_ids"] = [name1, name2]

        annot1 = review_data[idx]["annotations"][1]["annotation"]
        review_metrics["annot1"] = flatten_annot(annot1)
        annot2 = review_data[idx]["annotations"][2]["annotation"]
        review_metrics["annot2"] = flatten_annot(annot2)

        if not annot1 or not annot2:
            print("EMPTY:", annot1, annot2)
            raise Exception("Empty annotation")

        for at1, at2 in zip(annot1, annot2):
            if len(at1) != 5 or len(at2) != 5:
                print("ERROR:", at1, at2)
                raise Exception("Invalid annotation")
        
        excluded += 1
    except:
        continue

    # indexify opinions and spans
    indexify_spans(annot1, review, OPINION)
    indexify_spans(annot2, review, OPINION)
    indexify_spans(annot1, review, ASPECT)
    indexify_spans(annot2, review, ASPECT)

    # Delta
    delta = abs(len(annot1) - len(annot2))
    output_list.append(f"Delta: {delta}\n")
    review_metrics["delta"] = delta
    delta_total += delta

    # Aspect
    aspect_inter, aspect_union = get_span_inter_union(ASPECT, annot1, annot2)
    review_metrics[A[ASPECT]] = process_metric(
        ACOSI[ASPECT], aspect_inter, aspect_union
    )
    add_aspect = aspect_inter / aspect_union if aspect_union != 0 else 0
    aspect_total += add_aspect

    # Category
    cat_inter, cat_union = get_elt_inter_union(CATEGORY, annot1, annot2)
    review_metrics[A[CATEGORY]] = process_metric(ACOSI[CATEGORY], cat_inter, cat_union)
    add_cat = cat_inter / cat_union if cat_union != 0 else 0
    category_total += add_cat

    # Sentiment
    sent_inter, sent_union = get_elt_inter_union(SENTIMENT, annot1, annot2)
    review_metrics[A[SENTIMENT]] = process_metric(
        ACOSI[SENTIMENT], sent_inter, sent_union
    )
    add_sent = sent_inter / sent_union if sent_union != 0 else 0
    add_sent = (add_sent - 0.1667)/(1 - 0.1667)
    sentiment_total += add_sent

    # Opinion
    op_inter, op_union = get_span_inter_union(OPINION, annot1, annot2)
    review_metrics[A[OPINION]] = process_metric(ACOSI[OPINION], op_inter, op_union)
    add_op = op_inter / op_union if op_union != 0 else 0
    opinion_total += add_op

    # Implicit/Explicit
    ie_inter, ie_outer = get_elt_inter_union(IMPL_EXPL, annot1, annot2)
    review_metrics[A[IMPL_EXPL]] = process_metric(ACOSI[IMPL_EXPL], ie_inter, ie_outer)
    add_impl_expl = ie_inter / ie_outer if ie_outer != 0 else 0
    add_impl_expl = (add_impl_expl - 0.3333)/(1 - 0.3333)
    impl_expl_total += add_impl_expl

    review_metrics["cat_quad_avg"] = (
        review_metrics[A[CATEGORY]]
        + review_metrics[A[SENTIMENT]]
        + review_metrics[A[IMPL_EXPL]]
    ) / (NUM_QUAD_ELTS - 2)
    cat_quad_avg_total += review_metrics["cat_quad_avg"]

    review_metrics["span_quad_avg"] = (
        review_metrics[A[ASPECT]]
        + review_metrics[A[OPINION]]
    ) / (NUM_QUAD_ELTS - 3)
    span_quad_avg_total += review_metrics["span_quad_avg"]

    # Exact Match
    exact_inter, exact_union, set_annot1, set_annot2 = get_exact_inter_union(
        annot1, annot2
    )
    review_metrics["exact"] = process_metric("\nExact", exact_inter, exact_union)
    add_exact = exact_inter / exact_union if exact_union != 0 else 0
    exact_total += add_exact

    # # Adjusted overall match (find a way to link annotations)
    # adj_inter, adj_union = get_adj_inter_union(
    #     set_annot1, set_annot2, exact_inter, exact_union)
    # review_metrics["adjusted"] = process_metric(
    #     "Adjusted", adj_inter, adj_union)

    # Exact match when take away each column
    process_exclusions(annot1, annot2)

    metrics_list.append(review_metrics)
    output_list.append("\n")

tot_reviews = len(review_data) - excluded
print(tot_reviews)

all_metrics = {}

all_metrics["delta_avg"] = delta_total/tot_reviews
all_metrics["aspect_avg"] = aspect_total/tot_reviews
all_metrics["category_avg"] = category_total/tot_reviews
all_metrics["opinion_avg"] = opinion_total/tot_reviews
all_metrics["sentiment_avg"] = sentiment_total/tot_reviews
all_metrics["impl_expl_avg"] = impl_expl_total/tot_reviews
all_metrics["exact_avg"] = exact_total/tot_reviews
all_metrics["avg_cat_quad_avg"] = cat_quad_avg_total/tot_reviews
all_metrics["avg_span_quad_avg"] = span_quad_avg_total/tot_reviews

all_metrics["metrics_list"] = metrics_list

with open(args.output_file, "w") as f:
    json.dump(all_metrics, f, indent=4)

if verbose:
    print("\n".join(output_list))


f.close()
