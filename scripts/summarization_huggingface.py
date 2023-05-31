"""
Generate summaries for each review cluster (reviews for an individual product) using t5-base
"""
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

with open("data/results.json", "r") as f:
    data = json.load(f)

reviews_per_product_dict = {}

for review_dict in data:
    product = review_dict["p_name"]
    text = review_dict["review"]
    if product not in reviews_per_product_dict:
        reviews_per_product_dict[product] = [text]
    else:
        reviews_per_product_dict[product].append(text)

for product in reviews_per_product_dict:
    print(product, reviews_per_product_dict[product])
    concatenated_reviews = " <doc-sep> ".join(reviews_per_product_dict[product])
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    product_summary = summarizer(concatenated_reviews, min_length=5, max_length=20)
    print(product_summary)

print(len(reviews_per_product_dict))
