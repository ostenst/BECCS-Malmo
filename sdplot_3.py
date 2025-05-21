import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Bin the density data
data_density = data
num_bins = 15
data_density["cASU_bin"] = pd.cut(data_density["cASU"], bins=num_bins)
regret_fraction = (
    data_density.groupby("cASU_bin")["regret_2"]
    .apply(lambda x: (x > 0).mean())
)
bin_centers = data_density.groupby("cASU_bin")["cASU"].mean()

# Latin Hypercube Sampling (20%)
sample_size = int(len(data) * 0.05)
lhs = qmc.LatinHypercube(d=1)
sample = lhs.random(n=sample_size)
sample_indices = (sample * len(data)).astype(int).flatten()
data_lhs = data.iloc[sample_indices].reset_index(drop=True)

# Assign colors based on conditions
def assign_color(row):
    if row["cASU"] < 0.90:
        return "gray"
    # elif row["Auction"] == False:
    #     return "orange"
    # elif row["cASU"] > 180 and row["Auction"] == True:
    #     return "green"
    else:
        return "gray"

data_lhs["color"] = data_lhs.apply(assign_color, axis=1)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot Auction == False (mediumseagreen)
subset_mediumseagreen = data_lhs[data_lhs["color"] == "mediumseagreen"]
ax.scatter(
    subset_mediumseagreen["cASU"],
    subset_mediumseagreen["regret_2"],
    color="mediumseagreen",
    alpha=0.20,
    label="Auction = False"
)

# Plot Auction == True (gray)
subset_gray = data_lhs[data_lhs["color"] == "gray"]
ax.scatter(
    subset_gray["cASU"],
    subset_gray["regret_2"],
    color="gray",
    alpha=0.20,
    label="Auction = True"
)

# Overlay: highlight with distinct edge
overlay = data_lhs[((data_lhs["timing"] == 5) | (data_lhs["timing"] == 10)) & (data_lhs["cASU"] < 0.90)]
ax.scatter(
    overlay["cASU"],
    overlay["regret_2"],
    facecolors='none',
    edgecolors='black',
    linewidths=0.6,
    s=40,
    alpha=0.60,
    label="cASU < 0.90 and timing < 12.5"
)

ax.legend(loc="lower left", fontsize=10, title="Point Categories", title_fontsize=10)

# Formatting
ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
ax.set_xlabel("cASU", fontsize=12)
ax.set_ylabel("Regret 2", fontsize=12)
ax.set_title("Regret 2 vs cASU (LHS Sample, Colored Simultaneously)", fontsize=14)
ax.tick_params(labelsize=10)

# Plot DENSITY on secondary y-axis
ax2 = ax.twinx()
ax2.plot(
    bin_centers,
    regret_fraction,
    linestyle="-",
    marker="D",
    markerfacecolor="white",     # Fill color
    markeredgecolor="gray",   # Outline color
    markeredgewidth=1.5,
    color="gray",
    label="Fraction with Regret > 0"
)
ax2.set_ylabel("Fraction Regret > 0", fontsize=12)
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelsize=10)

# Optional legend for overlay line
ax2.legend(loc="upper right", fontsize=10)


plt.tight_layout()
plt.savefig("sd_3_oxy", dpi=600)
plt.show()
