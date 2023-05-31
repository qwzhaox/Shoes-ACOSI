import csv
import json
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

# USING ANNOTATOR ANNOTATIONS ON REVIEWS
# e.g. if annotator 1 has 3 quads with sentiments (+, +, -) and annotator 2 has 2 quads with (+, +), the average sentiment for that review would be 80% (4 +’s, 1 -
def compute_cumulative_sentiment_per_product(p_name, quint):
    products[p_name][1] += 1

    if quint[2] == "Positive":
        products[p_name][0] += 1
    elif quint[2] == "Netural":
        products[p_name][0] += 0.5


def process_products_dict(review, products):
    p_name = review["p_name"]
    if p_name not in products:
        products[p_name] = [0, 0] # cumulative_sentiment_score, number_of_quints
    
    for quint in review["annot1"]:
        compute_cumulative_sentiment_per_product(p_name, quint)
    
    for quint in review["annot2"]:
        compute_cumulative_sentiment_per_product(p_name, quint)


# Open the JSON file
with open('data/review_splits.json', 'r') as f:
    # Load the contents of the file as a dictionary
    reviews = json.load(f)

products = {}

num_reviews = 0
for review in reviews["train"]:
    process_products_dict(review, products)
    num_reviews += 1

for review in reviews["validation"]:
    process_products_dict(review, products)
    num_reviews += 1

for review in reviews["test"]:
    process_products_dict(review, products)
    num_reviews += 1

print(num_reviews)

# Compute mean sentiment scores per product; output to histogram and append to spreadsheet
# Create a list to store the rows of the CSV file
rows = []

# Add the header row to the list
rows.append(['Product Name', 'Mean Sentiment Score'])

mean_sentiment_scores = []
for product_name, sentiment_metrics in products.items():
    mean_sentiment_score = sentiment_metrics[0] / sentiment_metrics[1]
    mean_sentiment_scores.append(mean_sentiment_score)
    rows.append([product_name, mean_sentiment_score])

# Plot the histogram
plt.hist(mean_sentiment_scores, bins=10, alpha=0.5)
plt.xlabel('Average Sentiment Score')
plt.ylabel('Number of Products')
plt.ylim(top=100)
plt.title('Distribution of Average Sentiment Scores, Annotators')
plt.savefig("Distribution of Sentiment Mean Scores, Annotators")

# Write the list of rows to a CSV file
with open('output_annotator_sentiment_means_TEST.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)


# USING HUGGINGFACE PRE-TRAINED SENTIMENT ANALYSIS MODEL ON REVIEWS
# Load the sentiment analysis model and tokenizer
nlp = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", tokenizer="nlptown/bert-base-multilingual-uncased-sentiment")

# Open the local JSON file and read the contents
with open("data/results.json", "r") as f:
    data = json.load(f)

# Create a dictionary to store the sentiment scores for each product
product_sentiment_scores = {}

# Iterate over the data and calculate the sentiment score for each review
for review_dict in data:
    product = review_dict["p_name"]
    text = review_dict["review"]
    result = nlp(text)[0]
    print("result: ", result)
    label = result['label']
    star_rating = int(label.split()[0])
    print(star_rating) 

    # sentiment_score = 1 if result["label"] == "POSITIVE" else 0
    if product not in product_sentiment_scores:
        product_sentiment_scores[product] = [star_rating]
    else:
        product_sentiment_scores[product].append(star_rating)

# Create a dictionary to store the average sentiment score for each product
product_avg_sentiment_scores = {}

# Calculate the average sentiment score for each product
for product, scores in product_sentiment_scores.items():
    avg_score = sum(scores) / len(scores)
    product_avg_sentiment_scores[product] = avg_score

# Write the product names and mean sentiment scores to a CSV file
with open("output_huggingface_sentiment_means.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Mean Sentiment Score"])
    for product, avg_score in product_avg_sentiment_scores.items():
        writer.writerow([product, avg_score])

# Create a list of the average sentiment scores
scores = list(product_avg_sentiment_scores.values())

# Plot the histogram
plt.hist(scores, bins=10, alpha=0.5)
plt.xlabel('Average Sentiment Score')
plt.ylabel('Number of Products')
plt.title('Distribution of Average Sentiment Scores, Sentiment Analysis Model')
plt.savefig("Distribution of Sentiment Mean Scores, Hugging Face")
