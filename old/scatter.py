import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load datasets
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

# Ensure dataframes are aligned by index
outcomes = outcomes.loc[experiments.index]

# Check necessary columns
if 'crc' not in experiments.columns or 'regret' not in outcomes.columns or 'decision' not in experiments.columns:
    raise ValueError("Missing required columns: 'crc', 'regret', or 'decision' in datasets.")

# Merge decision column from experiments into outcomes
outcomes['crc'] = experiments['crc']
outcomes['decision'] = experiments['decision']

# Create a color map based on the unique categories in 'decision'
unique_decisions = outcomes["decision"].unique()
color_map = {category: color for category, color in zip(unique_decisions, plt.cm.Set1.colors)}

# Assign colors to each decision in the dataframe
outcomes["color"] = outcomes["decision"].map(color_map)

# Create scatter plot
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    outcomes["crc"],  # X-axis (CRC)
    outcomes["regret"],  # Y-axis (Regret)
    c=outcomes["color"],  # Colors based on 'decision'
    # edgecolor="k",
    alpha=0.8,
)

# Add legend for decision categories
legend_patches = [mpatches.Patch(color=color, label=category) for category, color in color_map.items()]
ax.legend(handles=legend_patches, title="Decision")

# Set labels and title
ax.set_xlabel("CRC")
ax.set_ylabel("Regret (Decision)")
ax.set_title("Scatterplot of CRC vs Regret (Colored by Decision)")

plt.show()
