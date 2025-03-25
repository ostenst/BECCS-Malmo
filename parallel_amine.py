import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler

# # Load data
# experiments = pd.read_csv("experiments.csv")
# outcomes = pd.read_csv("outcomes.csv")

# # Sample some data for faster plotting
# sample_fraction = 0.005  
# sampled_indices = experiments.sample(frac=sample_fraction, random_state=42).index
# experiments = experiments.loc[sampled_indices]
# outcomes = outcomes.loc[sampled_indices]

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

# Sample some data for faster plotting
sample_fraction = 0.05  
sampled_indices = experiments.sample(frac=sample_fraction, random_state=42).index
experiments = experiments.loc[sampled_indices]
outcomes = outcomes.loc[sampled_indices]

# ======== Scatter Plot: regret_amine vs. timing ======== #
df_original = pd.concat([experiments[["timing", "dr", "cAM"]], outcomes[["regret_amine"]]], axis=1)

df_true = df_original[(df_original["cAM"] < 1.381) & (df_original["dr"] > 0.078)] # Select a subset, corresponding to our SD findings
df_false = df_original[~((df_original["cAM"] < 1.381) & (df_original["dr"] > 0.078))]

# Perturb the timing values based on conditions
def add_perturbation(df, direction='left'):
    perturbed_timings = df["timing"].copy()
    for i, row in df.iterrows():
        if row["regret_amine"] == 0:
            # Add a greater random perturbation if regret_amine is 0
            perturbation = np.random.uniform(-0.3, -.6) if direction == 'left' else np.random.uniform(0.3, 0.6)
        else:
            # Add a small random perturbation if regret_amine is non-zero
            perturbation = np.random.uniform(-0.2, -0.5) if direction == 'left' else np.random.uniform(0.2, 0.5)
        
        perturbed_timings[i] += perturbation
    
    return perturbed_timings

# Apply perturbation
df_true["timing"] = add_perturbation(df_true, direction='left')
df_false["timing"] = add_perturbation(df_false, direction='right')

fig, ax1 = plt.subplots(figsize=(7, 6))
ax1.scatter(df_true["timing"], df_true["regret_amine"], color="deepskyblue", alpha=0.15, label="Scenarios of DR>7.8% and low CAPEX")
ax1.scatter(df_false["timing"], df_false["regret_amine"], color="crimson", alpha=0.15, label="All scenarios")

ax1.set_xlabel("timing")
ax1.set_ylabel("regret_amine", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_title("regret_amine vs timing with perturbation")
ax1.legend(loc="upper left")

# Reset the dfs # Create secondary y-axis for frequency (density)
df_true = df_original[(df_original["cAM"] < 1.381) & (df_original["dr"] > 0.078)] # Select a subset, corresponding to our SD findings
df_false = df_original[~((df_original["cAM"] < 1.381) & (df_original["dr"] > 0.078))]

ax2 = ax1.twinx()
timing_values = sorted(df_true["timing"].unique())  # Get the unique timing values
density_legend_handles = []  # Store legend handles for density lines

for df, line, scenario_label in [(df_true, "-", "Density (Low-regret)"), (df_false, "--", "Density (High-regret)")]:

    density_in_timings = []
    for timing_value in timing_values:
        df_bin = df[df["timing"] == timing_value]  # Select rows for this specific timing
        count_zeros = (df_bin["regret_amine"] == 0).sum()  # Count regret_amine == 0
        density = count_zeros / len(df_bin)  
        density_in_timings.append(density)

    # Plot the count of regret_amine == 0 for each timing value
    line_handle, = ax2.plot(timing_values, density_in_timings, color="black", linestyle=line, marker="o", label=scenario_label)
    density_legend_handles.append(line_handle)

ax2.set_ylabel("Density of regret_amine = 0", color="black")
ax2.tick_params(axis="y", labelcolor="black")
ax2.legend(handles=density_legend_handles, loc="upper right")
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom,0.4) # Hard-coding a ylim for density

# Grid & Show Plot
ax1.grid(True, linestyle="--", alpha=0.5)
plt.savefig('regret_amine.png', dpi=450)
plt.show()
