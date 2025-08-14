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
crc_intervals = [(25, 120), (120, 170), (170, 220), (220, 300)]
labels = ["25-120", "120-170", "170-220", "220-300"]

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
positions_true = [1, 4, 7, 10]
positions_false = [2, 5, 8, 11]

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

plt.xlabel("CRC Intervals")
plt.ylabel("Regret 1")
plt.title("Regret 1 Distribution by CRC Intervals")
plt.xticks([1.5, 4.5, 7.5, 10.5], labels)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Auction=True", "Auction=False"])
plt.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
# Add vertical grid lines at positions 3, 6, 9, 12
for pos in [3, 6, 9, 12]:
    plt.axvline(x=pos, color='gray', alpha=0.3, linestyle='-')
plt.xlim(0, 12)
plt.show()
