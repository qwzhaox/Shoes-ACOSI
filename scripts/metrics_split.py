from metrics import *

with open(args.input_file) as f:
    review_data = json.load(f)

products_dict = {}

for idx in range(len(review_data)):
    review_metrics = {}
    review = review_data[idx]["review"]
    review_metrics["review"] = review
    review_metrics["p_name"] = review_data[idx]["p_name"]

    output_list.append(
        f"######################################## REVIEW {idx} ###########################################\n")

    name1 = review_data[idx]["annotations"][0]["metadata"]["name"]
    name2 = review_data[idx]["annotations"][1]["metadata"]["name"]
    review_metrics["annotator_ids"] = [name1, name2]

    annot1 = review_data[idx]["annotations"][0]["annotation"]
    review_metrics["annot1"] = deepcopy(annot1)
    annot2 = review_data[idx]["annotations"][1]["annotation"]
    review_metrics["annot2"] = deepcopy(annot2)

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
    aspect_inter, aspect_union = get_span_inter_union(
        ASPECT, annot1, annot2)
    review_metrics[A[ASPECT]] = process_metric(
        ACOSI[ASPECT], aspect_inter, aspect_union)
    aspect_total += aspect_inter/aspect_union

    # Category
    cat_inter, cat_union = get_elt_inter_union(CATEGORY, annot1, annot2)
    review_metrics[A[CATEGORY]] = process_metric(
        ACOSI[CATEGORY], cat_inter, cat_union)
    category_total += cat_inter/cat_union

    # Sentiment
    sent_inter, sent_union = get_elt_inter_union(SENTIMENT, annot1, annot2)
    review_metrics[A[SENTIMENT]] = process_metric(
        ACOSI[SENTIMENT], sent_inter, sent_union)
    sentiment_total += sent_inter/sent_union

    # Opinion
    op_inter, op_union = get_span_inter_union(
        OPINION, annot1, annot2)
    review_metrics[A[OPINION]] = process_metric(
        ACOSI[OPINION], op_inter, op_union)
    opinion_total += op_inter/op_union

    # Implicit/Explicit
    ie_inter, ie_outer = get_elt_inter_union(IMPL_EXPL, annot1, annot2)
    review_metrics[A[IMPL_EXPL]] = process_metric(
        ACOSI[IMPL_EXPL], ie_inter, ie_outer)
    impl_expl_total += ie_inter/ie_outer

    review_metrics["quad_avg"] = (review_metrics[A[ASPECT]] +
                                  review_metrics[A[CATEGORY]] +
                                  review_metrics[A[SENTIMENT]] +
                                  review_metrics[A[OPINION]] +
                                  review_metrics[A[IMPL_EXPL]]) / NUM_QUAD_ELTS

    # Exact Match
    exact_inter, exact_union, set_annot1, set_annot2 = \
        get_exact_inter_union(annot1, annot2)
    review_metrics["exact"] = process_metric(
        "\nExact", exact_inter, exact_union)
    exact_total += exact_inter/exact_union

    # # Adjusted overall match (find a way to link annotations)
    # adj_inter, adj_union = get_adj_inter_union(
    #     set_annot1, set_annot2, exact_inter, exact_union)
    # review_metrics["adjusted"] = process_metric(
    #     "Adjusted", adj_inter, adj_union)

    # Exact match when take away each column
    process_exclusions(annot1, annot2)

    products_dict.setdefault(
        review_data[idx]["p_name"], []).append(review_metrics)

    output_list.append("\n")

# review_metrics["delta_avg"] = delta_total/len(review_data)
# review_metrics["aspect_avg"] = aspect_total/len(review_data)
# review_metrics["category_avg"] = category_total/len(review_data)
# review_metrics["opinion_avg"] = opinion_total/len(review_data)
# review_metrics["sentiment_avg"] = sentiment_total/len(review_data)
# review_metrics["impl_expl_avg"] = impl_expl_total/len(review_data)
# review_metrics["exact_avg"] = exact_total/len(review_data)

num_reviews = len(review_data)

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

TRAIN_PERCENT = 0.65
VALIDATION_PERCENT = 0.20
TEST_PERCENT = 0.15

GOAL = "goal"
ACTUAL = "actual"

ttv = [TRAIN, VALIDATION, TEST]

split_dict = {
    TRAIN: {},
    VALIDATION: {},
    TEST: {}
}

totals = {
    TRAIN: {
        GOAL: num_reviews * TRAIN_PERCENT,
        ACTUAL: 0,
    },
    VALIDATION: {
        GOAL: num_reviews * VALIDATION_PERCENT,
        ACTUAL: 0,
    },
    TEST: {
        GOAL: num_reviews * TEST_PERCENT,
        ACTUAL: 0,
    }
}

ttv_idx = 0
for product in products_dict.keys():
    if (len(ttv) == 0):
        split_dict[TRAIN][product] = products_dict[product]
        totals[TRAIN][ACTUAL] += len(products_dict[product])

    idx = ttv_idx % len(ttv)

    split_dict[ttv[idx]][product] = products_dict[product]
    totals[ttv[idx]][ACTUAL] += len(products_dict[product])

    if totals[ttv[idx]][ACTUAL] >= totals[ttv[idx]][GOAL]:
        ttv.pop(idx)
    else:
        ttv_idx += 1

split_dict["stats"] = totals

with open(args.output_file, "w") as f:
    json.dump(split_dict, f)


f.close()
