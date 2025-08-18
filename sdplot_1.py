import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Filter Integration == False
data = data[data["Integration"] == False].reset_index(drop=True)
data_true = data[data["Auction"] == True].reset_index(drop=True)
data_false = data[data["Auction"] == False].reset_index(drop=True)

# Create boxplots for each CRC interval
# crc_intervals = [(25, 120), (120, 170), (170, 220), (220, 300)]
# labels = ["25-120", "120-170", "170-220", "220-300"]

crc_intervals = [(25, 85), (85, 135), (135, 185), (185, 225), (225, 300)]
labels = ["25-85", "85-135", "135-185", "185-225", "225-300"]

plt.figure(figsize=(12, 6))
box_data_true = []
box_data_false = []
densities_true = []
densities_false = []

for start, end in crc_intervals:
    data_true_interval = data_true[(data_true["crc"] >= start) & (data_true["crc"] < end)]
    data_false_interval = data_false[(data_false["crc"] >= start) & (data_false["crc"] < end)]
    
    box_data_true.append(data_true_interval["regret_1"])
    box_data_false.append(data_false_interval["regret_1"])
    
    # Calculate density (fraction with regret_1 > 0)
    if len(data_true_interval) > 0:
        density_true = (data_true_interval["regret_1"] > 0).mean()
    else:
        density_true = 0
    densities_true.append(density_true)
    
    if len(data_false_interval) > 0:
        density_false = (data_false_interval["regret_1"] > 0).mean()
    else:
        density_false = 0
    densities_false.append(density_false)

print("Densities (fraction with regret_1 > 0):")
for i, (start, end) in enumerate(crc_intervals):
    print(f"CRC {start}-{end}: Auction=True: {densities_true[i]:.3f}, Auction=False: {densities_false[i]:.3f}")

# Create boxplots with proper positioning
# Position boxes at the center of each interval
positions_true = [0.7, 2.9, 5.1, 7.3, 9.5]
positions_false = [1.7, 3.9, 6.1, 8.3, 10.5]

# Color boxes based on density (red for high density, blue for low density)
colors_true = [plt.cm.coolwarm(d) for d in densities_true]
colors_false = [plt.cm.coolwarm(d) for d in densities_false]

bp1 = plt.boxplot(box_data_true, positions=positions_true, patch_artist=True, 
                  boxprops=dict(linewidth=1),
                  medianprops=dict(color='black', linewidth=2),
                  widths=1, whis=2)
bp2 = plt.boxplot(box_data_false, positions=positions_false, patch_artist=True, 
                  boxprops=dict(linewidth=1),
                  medianprops=dict(color='black', linewidth=2),
                  widths=1, whis=2)

# Apply colors to boxes
for patch, color in zip(bp1["boxes"], colors_true):
    patch.set_facecolor(color)

# Color the outlines, whiskers, and outliers in gray for Auction=False data
for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
    if element in bp2:
        if element == 'fliers':  # Outlier circles
            plt.setp(bp2[element], markeredgecolor='gray', markerfacecolor='gray')
        elif element == 'medians':  # Keep median lines gray
            plt.setp(bp2[element], color='gray')
        else:  # Lines (outlines, whiskers, caps) - but not box faces
            plt.setp(bp2[element], color='gray')

# Re-apply density-based face colors to Auction=False boxes (overriding the gray outline)
for patch, color in zip(bp2["boxes"], colors_false):
    patch.set_facecolor(color)

plt.xlabel("CRC Intervals", fontsize=12)
plt.ylabel("Regret 1", fontsize=12)
plt.title("Regret 1 Distribution by CRC Intervals", fontsize=14)
# Create tick positions that align with vertical grid lines and interval boundaries
# Ticks should be at: start of first interval, midpoints between box pairs, and end of last interval
tick_positions = [0, 2.3, 4.5, 6.7, 8.9, 11.1]  # Align with axvline positions and full range
tick_labels = ["25", "85", "135", "185", "225", "300"]  # Start and end values of intervals

plt.xticks(tick_positions, tick_labels, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
# Add vertical grid lines between CRC intervals
for pos in [0, 2.3, 4.5, 6.7, 8.9, 11.1]:
    plt.axvline(x=pos, color='gray', alpha=0.3, linestyle='-')
plt.xlim(0, 11.1)

# Add horizontal line at y = 0
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

plt.savefig("sd_1_auction.png", dpi=600)
# plt.show()

# Create a separate figure showing only Auction=False boxes
plt.figure(figsize=(12, 6))

# Create boxplots for Auction=False only
bp_false_only = plt.boxplot(box_data_false, positions=range(1, len(crc_intervals) + 1), 
                           patch_artist=True, 
                           boxprops=dict(linewidth=1),
                           medianprops=dict(color='black', linewidth=2),
                           widths=0.7, whis=2)

# Apply density-based colors to boxes
for patch, color in zip(bp_false_only["boxes"], colors_false):
    patch.set_facecolor(color)

# Customize the plot
plt.xlabel("CRC Intervals", fontsize=12)
plt.ylabel("Regret 1", fontsize=12)
plt.title("Regret 1 Distribution by CRC Intervals (Auction=False Only)", fontsize=14)

# Set tick positions and labels
tick_positions = range(1, len(crc_intervals) + 1)
tick_labels = labels
plt.xticks(tick_positions, tick_labels, fontsize=11)
plt.yticks(fontsize=11)

plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig("sd_1_auction_false_only.png", dpi=600)
plt.show()
