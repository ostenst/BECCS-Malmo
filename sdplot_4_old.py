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
# data = data[data["Integration"] == False].reset_index(drop=True)

# Bin the data
data_density = data[(data["timing"] == 5) | (data["timing"] == 10)].reset_index(drop=True)
num_bins = 12
data_density["immature_bin"] = pd.cut(data_density["immature"], bins=num_bins)
regret_fraction = (
    data_density.groupby("immature_bin")["regret_3"]
    .apply(lambda x: (x > 0).mean())
)
bin_centers = data_density.groupby("immature_bin")["immature"].mean()

# Latin Hypercube Sampling (20%)
sample_size = int(len(data) * 0.01)
lhs = qmc.LatinHypercube(d=1)
sample = lhs.random(n=sample_size)
sample_indices = (sample * len(data)).astype(int).flatten()
data_lhs = data.iloc[sample_indices].reset_index(drop=True)

# Assign colors based on conditions
def assign_color(row):
    if (row["timing"] == 5) or (row["timing"] == 10):
        return "mediumseagreen"
    # elif row["Auction"] == False:
    #     return "orange"
    # elif row["immature"] > 180 and row["Auction"] == True:
    #     return "green"
    else:
        return "gray"

data_lhs["color"] = data_lhs.apply(assign_color, axis=1)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# # Plot Auction == False (mediumseagreen)
# subset_mediumseagreen = data_lhs[data_lhs["color"] == "mediumseagreen"]
# ax.scatter(
#     subset_mediumseagreen["immature"],
#     subset_mediumseagreen["regret_3"],
#     color="mediumseagreen",
#     alpha=0.60,
#     label="timing < 12.5"
# )

# # Plot Auction == True (gray)
# subset_gray = data_lhs[data_lhs["color"] == "gray"]
# ax.scatter(
#     subset_gray["immature"],
#     subset_gray["regret_3"],
#     color="gray",
#     alpha=0.60,
#     label="timing > 12.5"
# )
# Single scatter call with per-point color from 'color' column
ax.scatter(
    data_lhs["immature"]*100,
    data_lhs["regret_3"],
    color=data_lhs["color"],
    alpha=0.4,
    label=None  # We'll handle the legend separately
)

# Overlay: highlight with distinct edge
overlay = data_lhs[((data_lhs["timing"] == 5) | (data_lhs["timing"] == 10)) & (data_lhs["immature"] < 1.27)]
ax.scatter(
    overlay["immature"]*100,
    overlay["regret_3"],
    facecolors='none',
    edgecolors='black',
    linewidths=0.6,
    s=40,
    alpha=0.50,
    label="immature < 1.27 & timing < 12.5"
)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Early investments',
           markerfacecolor='mediumseagreen', markersize=8, alpha=1.0),
    Line2D([0], [0], marker='o', color='w', label='Early investments and low immaturity',
       markerfacecolor='mediumseagreen', markeredgecolor='black', markersize=8, markeredgewidth=0.6),
    Line2D([0], [0], marker='o', color='w', label='Other',
           markerfacecolor='gray', markersize=8, alpha=0.6),
]

ax.legend(handles=legend_elements, loc="lower left", fontsize=10, title="Point Categories", title_fontsize=10)

# Formatting
ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
ax.set_xlabel("immature", fontsize=12)
ax.set_ylabel("Regret 3", fontsize=12)
ax.set_title("Regret 3 vs immature (LHS Sample, Colored Simultaneously)", fontsize=14)
ax.tick_params(labelsize=10)

# Plot DENSITY on secondary y-axis
ax2 = ax.twinx()
ax2.plot(
    bin_centers*100,
    regret_fraction*100,
    linestyle="-",
    marker="D",
    markerfacecolor="white",     # Fill color
    markeredgecolor="seagreen",   # Outline color
    markeredgewidth=1.5,
    color="seagreen",
    label="Fraction with Regret > 0"
)
ax2.set_ylabel("Fraction Regret > 0", fontsize=12)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelsize=10)

# Optional legend for overlay line
ax2.legend(loc="upper right", fontsize=10)


plt.tight_layout()
plt.savefig("sd_4_clc", dpi=600)
plt.show()
