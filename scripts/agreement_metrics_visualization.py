import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import sys

metrics_csv = sys.argv[1] # result_metrics.csv
cols = list(pd.read_csv(metrics_csv, nrows=1))
df = pd.read_csv(metrics_csv)

# exact, fuzzy, final
usecols = ['delta', 'aspect', 'category', 'sentiment', 'opinion', 'impl_expl', 'quint_avg', 'exact']
# exact, exact component, fuzzy component, overall match
usecols = ['exact', 'sentiment', 'category', 'impl_expl', 'aspect', 'opinion', 'delta', 'quint_avg']

# Create a single figure with subplots for each metric
fig, axs = plt.subplots(ncols=len(usecols), figsize=(15, 5), gridspec_kw={'wspace': 0.5})

# Iterate over each column in the dataframe and create a box plot in the corresponding subplot
for i, col in enumerate(usecols):
    df.boxplot([col], ax=axs[i])

# Save the figure as a single PNG file
plt.savefig('boxplots.png', dpi=300)
