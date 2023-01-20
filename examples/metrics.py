import json

ASPECT = 0
CATEGORY = 1
SENTIMENT = 2
OPINION = 3
I = 4

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


def get_quad_similarity(quad1, quad2):
    set_flat_quad1 = set(tuple(flatten_quad(list(quad1))))
    set_flat_quad2 = set(tuple(flatten_quad(list(quad2))))
    inter, union = get_inter_union(set_flat_quad1, set_flat_quad2)
    return inter/union


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
    dif1 = list(set_annotation1 - set_annotation2)
    dif2 = list(set_annotation2 - set_annotation1)

    set_list = []
    num = exact_inter
    denom = exact_union

    for i in range(len(dif1)):
        for j in range(len(dif2)):
            if i != j and ({dif1[i], dif2[j]} not in set_list):
                similarity = get_quad_similarity(dif1[i], dif2[j])
                if similarity > SIMILARITY_THRESHOLD:
                    set_list.append({dif1[i], dif2[j]})
                    num += similarity
                    denom -= similarity

    print(f"Adjusted match: {num}/{denom}, {(num/denom)*100}%")

    print()


f.close()
