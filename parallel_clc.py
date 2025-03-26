import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler

# Load data
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

# Sample some data for faster plotting
sample_fraction = 0.05  
sampled_indices = experiments.sample(frac=sample_fraction, random_state=42).index
experiments = experiments.loc[sampled_indices]
outcomes = outcomes.loc[sampled_indices]

# # Hardcoded columns for plotting PARALLEL
# experiment_columns = ["Auction", "crc"]
# outcome_columns = ["regret_ref", "regret_amine", "regret_oxy", "regret_clc"]

# # Combine selected columns
# df = pd.concat([experiments[experiment_columns], outcomes[outcome_columns]], axis=1).reset_index(drop=True)

# # Create a new categorical feature based on Auction and crc
# df["Auction_Category"] = np.select(
#     [
#         (df["Auction"] == True) & (df["crc"] > 133),
#         (df["Auction"] == True) & (df["crc"] <= 133),
#         (df["Auction"] == False) & (df["crc"] > 133),
#         (df["Auction"] == False) & (df["crc"] <= 133),
#     ],
#     [0, 1, 2, 3],  # Assign category values
# )

# # Drop original Auction column as it's now encoded
# df.drop(columns=["Auction"], inplace=True)

# # Normalize all columns, including Auction_Category
# scaler = MinMaxScaler()
# df[df.columns] = scaler.fit_transform(df)

# # Set up parallel coordinates plot
# num_features = len(df.columns)
# x_ticks = np.arange(num_features)  # X-axis positions for features
# fig, ax = plt.subplots(figsize=(12, 6))

# # Color by 'Auction_Category'
# auction_colors = cm.viridis(df["Auction_Category"].values)  # Normalize for colormap

# # Draw parallel coordinate lines
# for i in range(len(df)):
#     y_values = df.iloc[i].values  # Get row values
#     ax.plot(x_ticks, y_values, color=auction_colors[i], alpha=0.2)

# # Configure plot aesthetics
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(df.columns, rotation=45)
# ax.set_xlabel("Features and Outcomes")
# ax.set_ylabel("Normalized Values")
# ax.set_title("Custom Parallel Coordinates Plot Colored by Auction Category")
# ax.grid(axis="y", linestyle="--", alpha=0.5)

# # Create color legend for 'Auction_Category'
# legend_labels = [
#     "Auction=True & crc>133",
#     "Auction=True & crc≤133",
#     "Auction=False & crc>133",
#     "Auction=False & crc≤133"
# ]
# legend_colors = [cm.viridis(i / 3) for i in range(4)]  # Ensure consistent color scaling

# for color, label in zip(legend_colors, legend_labels):
#     ax.plot([], [], color=color, label=label, linewidth=4)

# ax.legend(title="Auction Categories", bbox_to_anchor=(1.05, 1), loc="upper left")


# ======== Scatter Plot: regret_clc vs. crc ======== #
df_original = pd.concat([experiments[["crc", "Auction", "timing"]], outcomes[["regret_clc"]]], axis=1)

df_true = df_original[df_original["timing"] > 17.5]
df_false = df_original[df_original["Auction"] == False]

fig, ax1 = plt.subplots(figsize=(7, 6))
ax1.scatter(df_false["crc"], df_false["regret_clc"], color="crimson", alpha=0.35, label="Auction=False")
ax1.scatter(df_true["crc"], df_true["regret_clc"], color="mediumseagreen", alpha=0.35, label="Delay>17.5 years")

ax1.set_xlabel("crc")
ax1.set_ylabel("regret_clc", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_title("regret_clc+density vs CRC price")
ax1.legend(loc="upper left")

# Create secondary y-axis for frequency
ax2 = ax1.twinx()
density_legend_handles = []  # Store legend handles for density lines

for df, line, auction_label in [(df_false, "--", "Auction=False"), (df_true, "-", "Delay>17.5 years")]:
    num_bins = 8  # Number of bins for crc
    crc_min, crc_max = df["crc"].min(), df["crc"].max()
    bins = np.linspace(crc_min, crc_max, num_bins + 1)  # Bin edges
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Midpoints for plotting

    # Count occurrences of regret_ref = 0 in each bin
    df_filtered = df[df["regret_clc"] == 0]
    freq_counts_in_bins, _ = np.histogram(df_filtered["crc"], bins=bins)
    total_counts_in_bins, _ = np.histogram(df["crc"], bins=bins)
    density = freq_counts_in_bins / total_counts_in_bins

    # Validate sum of frequencies
    assert freq_counts_in_bins.sum() == len(df_filtered), "Frequencies do not sum up correctly!"

    # Plot density with a label
    line_handle, = ax2.plot(bin_centers, density, color="black", linestyle=line, marker="o", label=f"Density ({auction_label})")
    density_legend_handles.append(line_handle)

ax2.set_ylabel("Density of regret_clc = 0", color="black")
ax2.tick_params(axis="y", labelcolor="black")
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom,1.0) # Hard-coding a ylim for density

# Add density legend explicitly
ax2.legend(handles=density_legend_handles, loc="upper center")

# Grid & Show Plot
ax1.grid(True, linestyle="--", alpha=0.5)
plt.savefig('regret_clc.png', dpi=450)
plt.show()