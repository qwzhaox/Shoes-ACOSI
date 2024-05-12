from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import argparse

from utils import get_dataset, OPINION_IDX, IMPL_EXPL_IDX

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path", "-d", type=str, default="data/acosi_dataset/shoes")
parser.add_argument("--output_csv_file", "-o", type=str, default="data/raw_data/fixed_annot_csv/k_most_similar_opinion_spans.csv")
parser.add_argument("--k", type=int, default=500)

args = parser.parse_args()

def most_similar_k_sentences(list1, list2, k):
    combined_list = [item["text"] for item in list1 + list2]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_list)

    tfidf_list1 = tfidf_matrix[:len(list1)]
    tfidf_list2 = tfidf_matrix[len(list1):]

    similarity_matrix = cosine_similarity(tfidf_list1, tfidf_list2)

    similarities = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            similarities.append((i, j, similarity_matrix[i, j]))

    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

    unique_pairs = []
    used_list1 = set()
    used_list2 = set()

    for pair in similarities:
        if pair[0] not in used_list1 and pair[1] not in used_list2:
            unique_pairs.append({
                "explicit opinion span": list1[pair[0]]["text"],
                "implicit opinion span": list2[pair[1]]["text"],
                "explicit review": list1[pair[0]]["review"],
                "implicit review": list2[pair[1]]["review"],
                "explicit quint": list1[pair[0]]["quint"],
                "implicit quint": list2[pair[1]]["quint"],
                "similarity score": pair[2]
            })
            used_list1.add(pair[0])
            used_list2.add(pair[1])
        if len(unique_pairs) >= k:
            break

    return unique_pairs

dataset = get_dataset(args.dataset_folder_path)

implicit_opinion_spans = []
explicit_opinion_spans = []

for example in dataset:
    review = example.split("####")[0]
    annotation = eval(example.split("####")[1])

    for quint in annotation:
        data_entry = {"text": quint[OPINION_IDX], "review": review, "quint": "\n".join(quint)}
        if quint[IMPL_EXPL_IDX] == "direct":
            explicit_opinion_spans.append(data_entry)
        else:
            implicit_opinion_spans.append(data_entry)

df = pd.DataFrame(most_similar_k_sentences(list1=explicit_opinion_spans, list2=implicit_opinion_spans, k=args.k))

df.rename(columns={
    "explicit opinion span": "explicit opinion span",
    "implicit opinion span": "implicit opinion span"
}, inplace=True)

df.to_csv(args.output_csv_file, index=False)
