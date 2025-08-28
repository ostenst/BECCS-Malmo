import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Filter Integration == False
data = data[data["Auction"] == False].reset_index(drop=True)
data = data[data["Integration"] == True].reset_index(drop=True)
data = data[data["crc"] < 100].reset_index(drop=True)
data_true = data[data["Procurement"] == True].reset_index(drop=True)
data_false = data[data["Procurement"] == False].reset_index(drop=True)

# Create boxplots for each CRC interval
# crc_intervals = [(25, 120), (120, 170), (170, 220), (220, 300)]
# labels = ["25-120", "120-170", "170-220", "220-300"]

ceiling_intervals = [(0, 220), (220, 255), (255, 290), (290, 350)]
labels = ["0-220", "220-255", "255-290", "290-350"]

plt.figure(figsize=(12, 6))
box_data_true = []
box_data_false = []
densities_true = []
densities_false = []

for start, end in ceiling_intervals:
    data_true_interval = data_true[(data_true["ceiling"] >= start) & (data_true["ceiling"] < end)]
    data_false_interval = data_false[(data_false["ceiling"] >= start) & (data_false["ceiling"] < end)]
    
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

print("Densities (fraction with regret_1 > 0) and data point counts:")
for i, (start, end) in enumerate(ceiling_intervals):
    count_true = len(box_data_true[i])
    count_false = len(box_data_false[i])
    print(f"ceiling {start}-{end}: Procurement=True: {densities_true[i]:.3f} ({count_true} points), Procurement=False: {densities_false[i]:.3f} ({count_false} points)")

# Create boxplots with proper positioning
positions_true = [0.7, 2.9, 5.1, 7.3]
positions_false = [1.7, 3.9, 6.1, 8.3]

# Color boxes based on density (red for high density, blue for low density)
colors_true = [plt.cm.coolwarm(d) for d in densities_true]
colors_false = [plt.cm.coolwarm(d) for d in densities_false]

bp1 = plt.boxplot(box_data_true, positions=positions_true, patch_artist=True, 
                  boxprops=dict(linewidth=1),
                  medianprops=dict(color='black', linewidth=2),
                  widths=0.99, whis=2)
bp2 = plt.boxplot(box_data_false, positions=positions_false, patch_artist=True, 
                  boxprops=dict(linewidth=1),
                  medianprops=dict(color='black', linewidth=2),
                  widths=0.99, whis=2)

# Apply colors to boxes
for patch, color in zip(bp1["boxes"], colors_true):
    patch.set_facecolor(color)

# Color the outlines, whiskers, and outliers in gray for Procurement=False data
for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
    if element in bp2:
        if element == 'fliers':  # Outlier circles
            plt.setp(bp2[element], markeredgecolor='gray', markerfacecolor='gray')
        elif element == 'medians':  # Keep median lines gray
            plt.setp(bp2[element], color='gray')
        else:  # Lines (outlines, whiskers, caps) - but not box faces
            plt.setp(bp2[element], color='gray')

# Re-apply density-based face colors to Procurement=False boxes (overriding the gray outline)
for patch, color in zip(bp2["boxes"], colors_false):
    patch.set_facecolor(color)

plt.xlabel("Ceiling Intervals", fontsize=12)
plt.ylabel("Regret 1", fontsize=12)
plt.title("Regret 1 Distribution by Ceiling Intervals", fontsize=14)
# Create tick positions that align with vertical grid lines and interval boundaries
# Ticks should be at: start of first interval, midpoints between box pairs, and end of last interval
tick_positions = [0, 2.3, 4.5, 6.7, 8.9]  # Align with axvline positions and full range
tick_labels = ["0", "220", "255", "290", "350"]  # Start and end values of intervals

plt.xticks(tick_positions, tick_labels, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
# Add vertical grid lines between ceiling intervals
for pos in [0, 2.3, 4.5, 6.7, 8.9]:
    plt.axvline(x=pos, color='gray', alpha=0.3, linestyle='-')
plt.xlim(0, 8.9)

# Add horizontal line at y = 0
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

plt.savefig("sd_2_integration.png", dpi=600)
plt.show()
