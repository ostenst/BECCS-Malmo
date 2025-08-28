import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Adapt to: show importance of BOTH immaturity and cASU!")
print("Label according to immaturity>=2.70, and plot cASU intervals")
# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Filter data based on immature >= 2.70
data_true = data[data["immature"] >= 2.70].reset_index(drop=True)
data_false = data[data["immature"] < 2.70].reset_index(drop=True)

# Define intervals for cASU instead of immature
intervals = [(0.68, 0.94), (0.94, 0.96), (0.96, 0.98), (0.98, 1.00), (1.00, 1.02)]  
labels = ["0.68-0.94", "0.94-0.96", "0.96-0.98", "0.98-1.00", "1.00-1.02"] 

# Create a single figure with two subplots side by side, sharing y-axis
# Use gridspec to control relative widths - make ax2 narrower than ax1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True, 
                                gridspec_kw={'width_ratios': [3, 2]})

# Left subplot: Boxplot
box_data_true = []
box_data_false = []
densities_true = []
densities_false = []

for start, end in intervals:
    data_true_interval = data_true[(data_true["cASU"] >= start) & (data_true["cASU"] < end)]
    data_false_interval = data_false[(data_false["cASU"] >= start) & (data_false["cASU"] < end)]
    
    box_data_true.append(data_true_interval["regret_3"])
    box_data_false.append(data_false_interval["regret_3"])
    
    # Calculate density (fraction with regret_3 > 0)
    if len(data_true_interval) > 0:
        density_true = (data_true_interval["regret_3"] > 0).mean()
    else:
        density_true = 0
    densities_true.append(density_true)
    
    if len(data_false_interval) > 0:
        density_false = (data_false_interval["regret_3"] > 0).mean()
    else:
        density_false = 0
    densities_false.append(density_false)

print("Densities (fraction with regret_3 > 0):")
for i, (start, end) in enumerate(intervals):
    count_true = len(box_data_true[i])
    count_false = len(box_data_false[i])
    print(f"cASU {start}-{end}: immature>=2.70: {densities_true[i]:.3f} ({count_true} points), immature<2.70: {densities_false[i]:.3f} ({count_false} points)")

# Create boxplots with proper positioning for 5 intervals
positions_true = [0.7, 2.9, 5.1, 7.3, 9.5]
positions_false = [1.7, 3.9, 6.1, 8.3, 10.5]

# Color boxes based on density (red for high density, blue for low density)
colors_true = [plt.cm.coolwarm(d) for d in densities_true]
colors_false = [plt.cm.coolwarm(d) for d in densities_false]

bp1 = ax1.boxplot(box_data_true, positions=positions_true, patch_artist=True, 
                  boxprops=dict(linewidth=1),
                  medianprops=dict(color='black', linewidth=2),
                  widths=0.99, whis=2)
bp2 = ax1.boxplot(box_data_false, positions=positions_false, patch_artist=True, 
                  boxprops=dict(linewidth=1),
                  medianprops=dict(color='black', linewidth=2),
                  widths=0.99, whis=2)

# Apply colors to boxes
for patch, color in zip(bp1["boxes"], colors_true):
    patch.set_facecolor(color)

# Color the outlines, whiskers, and outliers in gray for immature<2.70 data
for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
    if element in bp2:
        if element == 'fliers':  # Outlier circles
            plt.setp(bp2[element], markeredgecolor='gray', markerfacecolor='gray')
        elif element == 'medians':  # Keep median lines gray
            plt.setp(bp2[element], color='gray')
        else:  # Lines (outlines, whiskers, caps) - but not box faces
            plt.setp(bp2[element], color='gray')

# Re-apply density-based face colors to immature<2.70 boxes (overriding the gray outline)
for patch, color in zip(bp2["boxes"], colors_false):
    patch.set_facecolor(color)

ax1.set_xlabel("cASU Intervals", fontsize=14)
ax1.set_ylabel("Regret 3", fontsize=14)
ax1.set_title("Regret 3 Distribution by cASU Intervals", fontsize=16)

# Create tick positions that align with vertical grid lines and interval boundaries
# Ticks should be at: start of first interval, midpoints between box pairs, and end of last interval
tick_positions = [0, 2.3, 4.5, 6.7, 8.9, 11.1]  # Align with axvline positions and full range
tick_labels = ["0.68", "0.94", "0.96", "0.98", "1.00", "1.02"]  # Start and end values of intervals

ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, fontsize=13)
ax1.tick_params(axis='y', labelsize=13)

ax1.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines

# Add vertical grid lines between intervals, at midpoints between box pairs
for pos in [0, 2.3, 4.5, 6.7, 8.9, 11.1]:
    ax1.axvline(x=pos, color='gray', alpha=0.3, linestyle='-')

ax1.set_xlim(0, 11.1)

# Add horizontal line at y = 0 for the boxplot
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)


# PRINT ADDITIONAL SCATTER PLOT
# Calculate the ratio
data['capex_ratio'] = data['capex_amine'] / data['capex_clc']

# Now sample 1% of the data points for the scatter plot
sample_size = int(len(data) * 0.01)
sampled_data = data.sample(n=sample_size, random_state=42)

# Right subplot: Scatter plot
# Create a color array for all points based on immature value
colors_scatter = ['#DC6ACF' if immature < 2.70 else 'gray' for immature in sampled_data['immature']]

# Plot all points at once with the color array
ax2.scatter(sampled_data['capex_ratio'], sampled_data['regret_3'], 
           alpha=0.6, s=20, c=colors_scatter, edgecolors='black', linewidth=0.5)

ax2.set_xlabel("Capex Ratio (Amine/CLC)", fontsize=14)
ax2.set_title("Regret 3 vs Capex Ratio (Amine/CLC)", fontsize=16)
ax2.tick_params(axis='both', labelsize=13)
ax2.tick_params(axis='x', labelsize=13)  # Explicitly set x-axis tick fontsize to match ax1

# Add horizontal line at y = 0
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

# # Create a second y-axis for the density line plot
# ax2_twin = ax2.twinx()

# # Overlay line plot of density vs capex_ratio on the second y-axis
# ax2_twin.plot(bin_centers, scatter_densities, 'r-', linewidth=2, marker='o', markersize=6, label='Density (regret_3 > 0)')
# ax2_twin.set_ylabel('Density (%)', color='red')
# ax2_twin.tick_params(axis='y', labelcolor='red')

# # Add legend for the density line
# ax2_twin.legend(loc='upper right')

ax2.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig("sd_4_clc.png", dpi=600)
# plt.show()

# Create a separate figure showing only the immature>=2.70 boxes
plt.figure(figsize=(12, 6))

# Create boxplots for immature>=2.70 intervals only
bp_immature_only = plt.boxplot(box_data_true, positions=range(1, len(intervals) + 1), 
                              patch_artist=True, 
                              boxprops=dict(linewidth=1),
                              medianprops=dict(color='black', linewidth=2),
                              widths=0.7, whis=2)

# Apply density-based colors to boxes
for patch, color in zip(bp_immature_only["boxes"], colors_true):
    patch.set_facecolor(color)

# Customize the plot
plt.xlabel("cASU Intervals", fontsize=12)
plt.ylabel("Regret 3", fontsize=12)
plt.title("Regret 3 Distribution by cASU Intervals (immature>=2.70 Only)", fontsize=14)

# Set tick positions and labels
tick_positions = range(1, len(intervals) + 1)
tick_labels = labels
plt.xticks(tick_positions, tick_labels, fontsize=11)
plt.yticks(fontsize=11)

plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig("sd_4_clc_immature_only.png", dpi=600)

# Evaluate CAPEX for specific cASU values
cASU_values = [0.68, 0.94, 0.96, 0.98, 1.00, 1.02]

print("\nCAPEX values for different cASU:")
for cASU in cASU_values:
    CAPEX_ASU = 0.02*(59)**0.067/((1-0.95)**0.073) * (0.40*1000*3600/453.592 * 1.2 * 0.10)**cASU * 0.96 * 800/499.6 * 1.3
    print(f"cASU = {cASU}: CAPEX = {CAPEX_ASU:.2f} MEUR")

plt.show()
