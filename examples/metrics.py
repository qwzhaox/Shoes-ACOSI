import json

ASPECT = 0
CATEGORY = 1
SENTIMENT = 2
OPINION = 3
I = 4


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


def set_tuplify(list):
    new_set = set([tuple(sub_list) for sub_list in list])
    return new_set


def get_inter_union(set1, set2):
    return len(set1 & set2), len(set1 | set2)


def print_metric(metric, inter, union):
    print(
        f"{metric}: {inter}/{union}, {(inter/union)*100}%")


def get_quad_elt_metric(idx, metric):
    set1 = set([quad[idx] for quad in annotation1])
    set2 = set([quad[idx] for quad in annotation2])

    inter, union = get_inter_union(set1, set2)

    print_metric(metric, inter, union)


def indexify_opinions(annotation, review):
    for quad in annotation:
        # Needs to be tuple for overall conversion to set
        quad[OPINION] = tuple(indexify(review, quad[OPINION]))


def flatten_quad(quad):
    opinion = quad.pop(OPINION)
    quad.append([idx for idx in opinion])


f = open("example_format.json")
review_data = json.load(f)

for idx in range(len(review_data)):
    review = review_data[idx]["review"]
    print(f"Review {idx}")
    annotation1 = review_data[idx]["annotations"][0]["annotation"]
    annotation2 = review_data[idx]["annotations"][1]["annotation"]

    # indexify opinions
    indexify_opinions(annotation1, review)
    indexify_opinions(annotation2, review)

    # Delta
    print(f"Delta: {abs(len(annotation1) - len(annotation2))}")

    # Exact Match
    set_annotation1 = set_tuplify(annotation1)
    set_annotation2 = set_tuplify(annotation2)

    exact_inter, exact_union = \
        get_inter_union(set_annotation1, set_annotation2)

    print_metric("Exact match", exact_inter, exact_union)

    # Aspect
    get_quad_elt_metric(ASPECT, "Aspect")

    # Category
    get_quad_elt_metric(CATEGORY, "Category")

    # Sentiment
    get_quad_elt_metric(SENTIMENT, "Sentiment")

    # Opinion
    opinion1 = [quad[OPINION] for quad in annotation1]
    opinion2 = [quad[OPINION] for quad in annotation2]

    set_opinion1 = set(
        [idx for span in opinion1 for idx in span])
    set_opinion2 = set(
        [idx for span in opinion2 for idx in span])

    inter, union = get_inter_union(set_opinion1, set_opinion2)

    print_metric("Opinion", inter, union)

    # TODO: Adjusted overall match (find a way to link annotations)

    print()


f.close()
