import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set global font size
plt.rcParams.update({'font.size': 11})

# Load data
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

# Sample some data for faster plotting
sample_fraction = 0.05  
sampled_indices = experiments.sample(frac=sample_fraction, random_state=42).index
experiments = experiments.loc[sampled_indices]
outcomes = outcomes.loc[sampled_indices]

# Prepare data
df_original = pd.concat([experiments[["crc", "Auction"]], outcomes[["regret_1"]]], axis=1)
df_true = df_original[df_original["Auction"] == True]
df_false = df_original[df_original["Auction"] == False]

# Define number of bins
n_bins = 10

def compute_regret_fraction(df):
    df = df.copy()
    df["crc_bin"] = pd.cut(df["crc"], bins=n_bins)
    grouped = df.groupby("crc_bin")["regret_1"]
    regret_fraction = grouped.apply(lambda x: (x > 0).mean())
    bin_centers = [interval.mid for interval in regret_fraction.index]
    return bin_centers, regret_fraction.values

# Compute regret fractions
x_true, y_true = compute_regret_fraction(df_true)
x_false, y_false = compute_regret_fraction(df_false)

# ======== Plotting ======== #
fig, ax1 = plt.subplots(figsize=(8.5, 6))

# Scatter on left y-axis
ax1.scatter(df_true["crc"], df_true["regret_1"], alpha=0.3, label="Scenario with auction subsidy (160 EUR/t)", color="deepskyblue", s=10)
ax1.scatter(df_false["crc"], df_false["regret_1"], alpha=0.3, label="Scenario without subsidy", color="crimson", s=10)

ax1.set_xlabel("CRC")
ax1.set_ylabel("regret_1")
ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.8)
ax1.grid(True)
ax1.set_ylim(-1000,1000)
ax1.set_xlim(50, 400)

# Create second y-axis
ax2 = ax1.twinx()
ax2.plot(x_true, y_true*100, marker='o', label="Fraction of regrettable scenarios (with subsidy)", color="dodgerblue", linewidth=2)
ax2.plot(x_false, y_false*100, marker='o', label="Fraction of regrettable scenarios (no subsidy)", color="firebrick", linewidth=2)
ax2.set_ylabel("Fraction of regret_1 > 0 [%]")
ax2.set_ylim(-5, 105)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=11)

plt.title("Scatterplot of regret_1 vs CRC with Regret Fractions (Dual Y-Axis)")
plt.tight_layout()
plt.savefig('regret_1.png', dpi=450)
plt.show()