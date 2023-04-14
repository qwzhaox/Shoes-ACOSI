import csv
import json
import matplotlib.pyplot as plt
import pandas as pd

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

"""
# Read the CSV file into a DataFrame
df = pd.read_csv('data/result_metrics.csv')

# Group the data by product name and calculate the mean sentiment score for each group
grouped = df.groupby('p_name')['sentiment'].mean()

# Convert the grouped data into a new DataFrame with appropriate column names
result_df = pd.DataFrame({'Product Name': grouped.index, 'Average Sentiment': grouped.values})

# Write the new DataFrame to a CSV file
# result_df.to_csv('output_annotator_sentiment_means.csv', index=False)

# Extract the sentiment scores from the DataFrame
sentiment_scores = result_df['Average Sentiment']

# Create a histogram of the sentiment scores
plt.hist(sentiment_scores)

# Add labels and title
plt.xlabel('Average Sentiment Score')
plt.ylabel('Number of Products')
plt.ylim(top=100)
plt.title('Distribution of Average Sentiment Scores, Annotators')

# Show the plot
plt.savefig("Distribution of Sentiment Mean Scores, Annotators")
"""
