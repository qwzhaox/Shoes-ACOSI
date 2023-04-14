import csv
import json
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the pre-trained sentiment analysis pipeline
nlp = pipeline("sentiment-analysis")

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
    sentiment_score = 1 if result["label"] == "POSITIVE" else 0
    if product not in product_sentiment_scores:
        product_sentiment_scores[product] = [sentiment_score]
    else:
        product_sentiment_scores[product].append(sentiment_score)

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
