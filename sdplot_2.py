import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Filter Integration == True and Auction == False
data = data[(data["Integration"] == True) & (data["Auction"] == False) & (data["crc"] < 210)].reset_index(drop=True)
# data = data[(data["Integration"] == True) & (data["Auction"] == False)].reset_index(drop=True)

# Bin the filtered data where "Procurement" == False
data_density = data[data["Procurement"] == False].reset_index(drop=True)
num_bins = 8
data_density["EUA_bin"] = pd.cut(data_density["EUA"], bins=num_bins)
regret_fraction = (
    data_density.groupby("EUA_bin")["regret_1"]
    .apply(lambda x: (x > 0).mean())
)
bin_centers = data_density.groupby("EUA_bin")["EUA"].mean()

# Latin Hypercube Sampling (20%)
sample_size = int(len(data) * 0.2)
lhs = qmc.LatinHypercube(d=1)
sample = lhs.random(n=sample_size)
sample_indices = (sample * len(data)).astype(int).flatten()
data_lhs = data.iloc[sample_indices].reset_index(drop=True)
data_lhs = data # NOTE testing all data!

# Assign colors based on conditions
def assign_color(row):
    if row["Procurement"] == False:
        return "deepskyblue"
    # elif row["Auction"] == False:
    #     return "orange"
    # elif row["crc"] > 180 and row["Auction"] == True:
    #     return "green"
    else:
        return "gray"

data_lhs["color"] = data_lhs.apply(assign_color, axis=1)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot Auction == False (deepskyblue)
subset_deepskyblue = data_lhs[data_lhs["color"] == "deepskyblue"]
ax.scatter(
    subset_deepskyblue["EUA"],
    subset_deepskyblue["regret_1"],
    color="deepskyblue",
    s=60,
    alpha=0.08,
    label="Procurement = False"
)

# Plot Auction == True (gray)
subset_gray = data_lhs[data_lhs["color"] == "gray"]
ax.scatter(
    subset_gray["EUA"],
    subset_gray["regret_1"],
    color="gray",
    s=60,
    alpha=0.08,
    label="Procurement = True"
)

# Overlay: highlight with distinct edge
overlay = data_lhs[(data_lhs["EUA"] < 4.6) & (data_lhs["Procurement"] == False)]
ax.scatter(
    overlay["EUA"],
    overlay["regret_1"],
    facecolors='none',
    edgecolors='black',
    linewidths=0.6,
    s=60,
    alpha=0.40,
    label="EUA < 4.6 & Procurement = False"
)

ax.legend(loc="lower left", fontsize=10, title="Point Categories", title_fontsize=10)

# Formatting
ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
ax.set_xlabel("EUA", fontsize=12)
ax.set_ylabel("Regret 1", fontsize=12)
ax.set_title("Regret 1 vs EUA (LHS Sample, Colored Simultaneously)", fontsize=14)
ax.tick_params(labelsize=10)

# Plot DENSITY on secondary y-axis
ax2 = ax.twinx()
ax2.plot(
    bin_centers,
    regret_fraction,
    linestyle="-",
    marker="D",
    markerfacecolor="white",     # Fill color
    markeredgecolor="deepskyblue",   # Outline color
    markeredgewidth=1.5,
    color="deepskyblue",
    label="Fraction with Regret > 0"
)
ax2.set_ylabel("Fraction Regret > 0", fontsize=12)
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelsize=10)

# Optional legend for overlay line
ax2.legend(loc="upper right", fontsize=10)


plt.tight_layout()
plt.savefig("sd_2_integration", dpi=600)
plt.show()
