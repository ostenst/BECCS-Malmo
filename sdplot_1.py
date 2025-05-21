import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from matplotlib.lines import Line2D

# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Filter Integration == False
data = data[data["Integration"] == False].reset_index(drop=True)

# Bin the filtered data where "Auction" == False
data_density = data[data["Auction"] == False].reset_index(drop=True)
num_bins = 12
data_density["crc_bin"] = pd.cut(data_density["crc"], bins=num_bins)
regret_fraction = (
    data_density.groupby("crc_bin")["regret_1"]
    .apply(lambda x: (x > 0).mean())
)
bin_centers = data_density.groupby("crc_bin")["crc"].mean()

# Latin Hypercube Sampling (20%)
sample_size = int(len(data) * 0.02)
lhs = qmc.LatinHypercube(d=1)
sample = lhs.random(n=sample_size)
sample_indices = (sample * len(data)).astype(int).flatten()
data_lhs = data.iloc[sample_indices].reset_index(drop=True)

# Assign colors based on conditions
def assign_color(row):
    if row["Auction"] == False:
        return "crimson"
    # elif row["Auction"] == False:
    #     return "orange"
    # elif row["crc"] > 180 and row["Auction"] == True:
    #     return "green"
    else:
        return "gray"

data_lhs["color"] = data_lhs.apply(assign_color, axis=1)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# # Plot Auction == False (crimson)
# subset_crimson = data_lhs[data_lhs["color"] == "crimson"]
# ax.scatter(
#     subset_crimson["crc"],
#     subset_crimson["regret_1"],
#     color="crimson",
#     alpha=0.20,
#     label="Auction = False"
# )

# # Plot Auction == True (gray)
# subset_gray = data_lhs[data_lhs["color"] == "gray"]
# ax.scatter(
#     subset_gray["crc"],
#     subset_gray["regret_1"],
#     color="gray",
#     alpha=0.20,
#     label="Auction = True"
# )
ax.scatter(
    data_lhs["crc"],
    data_lhs["regret_1"],
    color=data_lhs["color"],
    alpha=0.4,
    label=None  # We'll handle the legend separately
)

# Overlay: highlight with distinct edge
overlay = data_lhs[(data_lhs["crc"] < 180) & (data_lhs["Auction"] == False)]
ax.scatter(
    overlay["crc"],
    overlay["regret_1"],
    facecolors='none',
    edgecolors='black',
    linewidths=0.6,
    s=40,
    alpha=0.30,
    label="crc < 180 & Auction = False"
)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='ti',
           markerfacecolor='crimson', markersize=8, alpha=1.0),
    Line2D([0], [0], marker='o', color='w', label='im',
       markerfacecolor='crimson', markeredgecolor='black', markersize=8, markeredgewidth=0.6),
    Line2D([0], [0], marker='o', color='w', label='ti',
           markerfacecolor='gray', markersize=8, alpha=0.6),
]


ax.legend(handles=legend_elements, loc="lower left", fontsize=10, title="Point Categories", title_fontsize=10)

# Formatting
ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
ax.set_xlabel("crc", fontsize=12)
ax.set_ylabel("Regret 1", fontsize=12)
ax.set_title("Regret 1 vs crc (LHS Sample, Colored Simultaneously)", fontsize=14)
ax.tick_params(labelsize=10)

# Plot DENSITY on secondary y-axis
ax2 = ax.twinx()
ax2.plot(
    bin_centers,
    regret_fraction*100,
    linestyle="-",
    marker="D",
    markerfacecolor="white",     # Fill color
    markeredgecolor="maroon",   # Outline color
    markeredgewidth=1.5,
    color="maroon",
    label="Fraction with Regret > 0"
)
ax2.set_ylabel("Fraction Regret > 0", fontsize=12)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelsize=10)

# Optional legend for overlay line
ax2.legend(loc="upper right", fontsize=10)


plt.tight_layout()
plt.savefig("sd_1_auction", dpi=600)
plt.show()
