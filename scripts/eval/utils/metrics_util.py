def indexify(review, span):
    orig_array = review.split(" ")
    span_array = span.split(" ")
    start_idx = -1
    while orig_array[start_idx] != span_array[0]:
        start_idx += 1
        if start_idx >= len(orig_array):
            return [-1]

    end_idx = start_idx + len(span_array)
    num_span_array = [i for i in range(start_idx, end_idx)]
    return tuple(num_span_array)


def indexify_spans(annotation, review, idx):
    indexified_list = []
    for quad in annotation:
        quad[idx] = indexify(review, quad[idx])
        indexified_list.extend(quad[idx])
    return indexified_list
