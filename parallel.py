import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

# Sampling just some of the data for faster plotting
sample_fraction = 0.005  # Adjust as needed
sampled_indices = experiments.sample(frac=sample_fraction, random_state=42).index
experiments = experiments.loc[sampled_indices]
outcomes = outcomes.loc[sampled_indices]

# Hardcoded columns for plotting
experiment_columns = ["crc", "cbio", "Auction"]
outcome_columns = ["regret_ref","regret_amine","regret_oxy","regret_clc"]

# Combine selected columns
df = pd.concat([experiments[experiment_columns], outcomes[outcome_columns]], axis=1).reset_index(drop=True)

# Normalize numerical columns & encode categorical/boolean features
scaler = MinMaxScaler()
label_encoders = {}

for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'bool':  # Encode categorical and boolean columns
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    else:  # Normalize numerical columns
        df[col] = scaler.fit_transform(df[[col]])

# Set up parallel coordinates plot
num_features = len(df.columns)
x_ticks = np.arange(num_features)  # X-axis positions for features
fig, ax = plt.subplots(figsize=(12, 6))

# Color by 'Auction' categorical feature
auction_colors = cm.viridis(df["Auction"].values / df["Auction"].max())  # Normalize for colormap

# Draw parallel coordinate lines
for i in range(len(df)):
    y_values = df.iloc[i].values  # Get row values
    ax.plot(x_ticks, y_values, color=auction_colors[i], alpha=0.5)

# Configure plot aesthetics
ax.set_xticks(x_ticks)
ax.set_xticklabels(df.columns, rotation=45)
ax.set_xlabel("Features and Outcomes")
ax.set_ylabel("Normalized Values")
ax.set_title("Custom Parallel Coordinates Plot Colored by Auction")
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Create color legend for 'Auction'
unique_auctions = sorted(df["Auction"].unique())
legend_colors = [cm.viridis(a / df["Auction"].max()) for a in unique_auctions]
auction_labels = label_encoders["Auction"].inverse_transform(unique_auctions)

for color, label in zip(legend_colors, auction_labels):
    ax.plot([], [], color=color, label=label, linewidth=4)

ax.legend(title="Auction", bbox_to_anchor=(1.05, 1), loc="upper left")

# ======== Scatter Plot: regret_ref vs. crc ======== #
fig, ax2 = plt.subplots(figsize=(6, 6))

# Get original (unnormalized) values for 'Auction', 'crc', and 'regret_ref'
df_original = pd.concat([experiments[["crc", "Auction"]], outcomes[["regret_ref"]]], axis=1)

# Separate rows where Auction is True or False
df_true = df_original[df_original["Auction"] == True]
df_false = df_original[df_original["Auction"] == False]

# Scatter plot
ax2.scatter(df_true["crc"], df_true["regret_ref"], color="blue", alpha=0.6, label="Auction = True")
ax2.scatter(df_false["crc"], df_false["regret_ref"], color="red", alpha=0.6, label="Auction = False")

# Compute frequency of regret_ref == 0 at each crc level
freq_counts = df_original[df_original["regret_ref"] == 0].groupby("crc").size()

# Normalize frequency to match y-axis scale
if not freq_counts.empty:
    freq_scaled = freq_counts / freq_counts.max() * df_original["regret_ref"].max()

    # Plot frequency as a line
    ax2.plot(freq_counts.index, freq_scaled, color="black", linestyle="--", marker="o", label="Freq of regret_ref = 0")

# Configure scatter plot aesthetics
ax2.set_xlabel("crc")
ax2.set_ylabel("regret_ref")
ax2.set_title("Scatter Plot: regret_ref vs. crc (Auction Groups) + Frequency of regret_ref=0")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

# Show plot
plt.show()