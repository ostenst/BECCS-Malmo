import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Filter data - PLACEHOLDER: Define your filtering criteria here
# Example: data = data[data["SomeVariable"] == SomeValue].reset_index(drop=True)
data = data.reset_index(drop=True)

# Use non-filtered data for single group analysis
data_single = data.copy().reset_index(drop=True)

# Define intervals - PLACEHOLDER: Define your interval ranges here
# Example: intervals = [(0, 10), (10, 20), (20, 30)]
# Example: labels = ["0-10", "10-20", "20-30"]
intervals = [(0.0, 1.7), (1.7, 2.7), (2.7, 3.0), (3.0, 3.5), (3.5, 4.0)]  
labels = ["0.0-1.7", "1.7-2.7", "2.7-3.0", "3.0-3.5", "3.5-4.0"] 

# Create a single figure with two subplots side by side, sharing y-axis
# Use gridspec to control relative widths - make ax2 narrower than ax1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True, 
                                gridspec_kw={'width_ratios': [3, 2]})

# Left subplot: Boxplot
box_data = []
densities = []

for start, end in intervals:
    data_interval = data_single[(data_single["immature"] >= start) & (data_single["immature"] < end)]
    
    box_data.append(data_interval["regret_3"])
    
    # Calculate density (fraction with regret_3 > 0)
    if len(data_interval) > 0:
        density = (data_interval["regret_3"] > 0).mean()
    else:
        density = 0
    densities.append(density)

print("Densities (fraction with regret_3 > 0):")
for i, (start, end) in enumerate(intervals):
    print(f"immature {start}-{end}: {densities[i]:.3f}")

# Create boxplots with proper positioning for 5 intervals
positions = [0.75, 2, 3.25, 4.5, 5.75]

# Color boxes based on density (red for high density, blue for low density)
colors = [plt.cm.coolwarm(d) for d in densities]

bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, 
                 boxprops=dict(linewidth=1),
                 medianprops=dict(color='black', linewidth=2),
                 widths=0.99, whis=2)

# Apply colors to boxes
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

ax1.set_xlabel("immature Intervals", fontsize=14)
ax1.set_ylabel("Regret 3", fontsize=14)
ax1.set_title("Regret 3 Distribution by immature Intervals", fontsize=16)

# Create tick positions that align with vertical grid lines and interval boundaries
# Ticks should be at: start of first interval, midpoints between box pairs, and end of last interval
tick_positions = [0, 1.375, 2.625, 3.875, 5.125, 6.5]  # Align with axvline positions and full range
tick_labels = ["0.0", "1.7", "2.7", "3.0", "3.5", "4.0"]  # Start and end values of intervals

ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, fontsize=13)
ax1.tick_params(axis='y', labelsize=13)

ax1.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines

# Add vertical grid lines between intervals, at midpoints between box positions
vlines = [(positions[i] + positions[i+1]) / 2 for i in range(len(positions)-1)]
for pos in vlines:
    ax1.axvline(x=pos, color='gray', alpha=0.3, linestyle='-')

ax1.set_xlim(positions[0] - 0.75, positions[-1] + 0.75)

# Add horizontal line at y = 0 for the boxplot
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

# Calculate the ratio
data_single['capex_ratio'] = data_single['capex_amine'] / data_single['capex_clc']

# # Bin the data into 24 bins and calculate density for each bin (using full dataset)
# capex_ratios = data_single['capex_ratio']
# regret_values = data_single['regret_3']

# # Create non-uniform bins with more density around capex_ratio=1
# # Use a custom approach: more bins in the middle range, fewer at extremes
# min_ratio = capex_ratios.min()
# max_ratio = capex_ratios.max()

# # Create custom bin edges with more concentration around ratio=1
# # Use a combination of linear and exponential spacing
# left_side = np.linspace(min_ratio, 0.95, 4)  # Fewer bins from min to 0.8
# middle_left = np.linspace(0.95, 1.0, 6)     # More bins from 0.8 to 1.0
# middle_right = np.linspace(1.0, 1.05, 6)    # More bins from 1.0 to 1.2
# right_side = np.linspace(1.05, max_ratio, 4) # Fewer bins from 1.2 to max

# # Combine all bin edges and remove duplicates
# bin_edges = np.concatenate([left_side, middle_left[1:], middle_right[1:], right_side[1:]])
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# print(f"Number of bins: {len(bin_centers)}")
# print(f"Bin edges: {bin_edges}")
# print(f"Bin centers: {bin_centers}")

# # Calculate density (fraction with regret_3 > 0) for each bin
# scatter_densities = []
# for i in range(len(bin_edges) - 1):
#     mask = (capex_ratios >= bin_edges[i]) & (capex_ratios < bin_edges[i + 1])
#     if mask.sum() > 0:
#         density = (regret_values[mask] > 0).mean()
#     else:
#         density = 0
#     scatter_densities.append(density)

# Now sample 1% of the data points for the scatter plot
sample_size = int(len(data_single) * 0.01)
sampled_data = data_single.sample(n=sample_size, random_state=42)

# Right subplot: Scatter plot
# Create a color array for all points based on immature value
colors_scatter = ['#DC6ACF' if immature < 1.9 else 'gray' for immature in sampled_data['immature']]
#548687
#1BE7FF

# Plot all points at once with the color array
ax2.scatter(sampled_data['capex_ratio'], sampled_data['regret_3'], 
           alpha=0.6, s=20, c=colors_scatter, edgecolors='black', linewidth=0.5)

ax2.set_xlabel("Capex Ratio (Amine/CLC)", fontsize=14)
ax2.set_title("Regret 3 vs Capex Ratio (Amine/CLC)", fontsize=16)
ax2.tick_params(axis='both', labelsize=13)

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

# Create a separate figure showing only the immature interval boxes
plt.figure(figsize=(12, 6))

# Create boxplots for immature intervals only
bp_immature_only = plt.boxplot(box_data, positions=range(1, len(intervals) + 1), 
                              patch_artist=True, 
                              boxprops=dict(linewidth=1),
                              medianprops=dict(color='black', linewidth=2),
                              widths=0.7, whis=2)

# Apply density-based colors to boxes
for patch, color in zip(bp_immature_only["boxes"], colors):
    patch.set_facecolor(color)

# Customize the plot
plt.xlabel("immature Intervals", fontsize=12)
plt.ylabel("Regret 3", fontsize=12)
plt.title("Regret 3 Distribution by immature Intervals", fontsize=14)

# Set tick positions and labels
tick_positions = range(1, len(intervals) + 1)
tick_labels = labels
plt.xticks(tick_positions, tick_labels, fontsize=11)
plt.yticks(fontsize=11)

plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig("sd_4_clc_immature_only.png", dpi=600)
plt.show()
