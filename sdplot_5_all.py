import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and merge
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
data = pd.concat([experiments, outcomes], axis=1)

# Define filter conditions for columns
def get_filtered_data(data, filter_name):
    if filter_name == "Unfiltered":
        return data.copy()
    elif filter_name == "Auction=True":
        return data[data["Auction"] == True].reset_index(drop=True)
    elif filter_name == "Auction=False":
        return data[data["Auction"] == False].reset_index(drop=True)
    elif filter_name == "Integration=True,\nProcurement=False":
        CRC_interval = (0, 100)
        filter = data[(data["Integration"] == True) & (data["crc"] >= CRC_interval[0]) & (data["crc"] < CRC_interval[1])].reset_index(drop=True)
        return filter[filter["Procurement"] == False].reset_index(drop=True)
    elif filter_name == "Integration=True,\nProcurement=True":
        CRC_interval = (0, 100)
        filter = data[(data["Integration"] == True) & (data["crc"] >= CRC_interval[0]) & (data["crc"] < CRC_interval[1])].reset_index(drop=True)
        return filter[filter["Procurement"] == True].reset_index(drop=True)

filter_names = [
    "Unfiltered",
    "Auction=True",
    "Auction=False",
    "Integration=True,\nProcurement=False",
    "Integration=True,\nProcurement=True"
]

# Define regret columns for rows
regret_cols = ["regret_1", "regret_2"]
row_labels = ["Regret 1", "Regret 2"]

# Create the figure with 2 rows x 5 columns
fig, axes = plt.subplots(2, 5, figsize=(5.5, 6), sharey='row')

# Collect box data and densities for both regrets
all_box_data = []
all_densities = []

for row_idx, regret_col in enumerate(regret_cols):
    row_box_data = []
    row_densities = []
    
    for col_idx, filter_name in enumerate(filter_names):
        filtered_data = get_filtered_data(data, filter_name)
        
        # Get the regret values for this filter
        box_data = filtered_data[regret_col]
        row_box_data.append(box_data)
        
        # Calculate density (fraction with regret > 0)
        if len(filtered_data) > 0:
            density = (filtered_data[regret_col] > 0).mean()
        else:
            density = 0
        row_densities.append(density)
    
    all_box_data.append(row_box_data)
    all_densities.append(row_densities)

# Create the plots
for row_idx, regret_col in enumerate(regret_cols):
    for col_idx, filter_name in enumerate(filter_names):
        ax = axes[row_idx, col_idx]
        
        box_data = all_box_data[row_idx][col_idx]
        density = all_densities[row_idx][col_idx]
        
        # Color box based on density using coolwarm colormap
        color = plt.cm.coolwarm(density)
        
        # Create single boxplot
        bp = ax.boxplot([box_data], positions=[1],
                        patch_artist=True,
                        boxprops=dict(linewidth=1),
                        medianprops=dict(color='black', linewidth=2),
                        widths=0.6, whis=2)
        
        # Apply color to box
        bp["boxes"][0].set_facecolor(color)
        
        # Add horizontal line at y = 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        
        # Remove x-axis ticks
        ax.set_xticks([])
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set title for first row only
        if row_idx == 0:
            ax.set_title(filter_name, fontsize=9, fontweight='bold')
        
        # Set y-axis label for first column only
        if col_idx == 0:
            ax.set_ylabel(row_labels[row_idx], fontsize=11, fontweight='bold')
        
        # # Add density and count annotation
        # n_points = len(box_data)
        # ax.text(0.95, 0.95, f'd={density:.2f}\nn={n_points}', 
        #         transform=ax.transAxes, fontsize=8, verticalalignment='top',
        #         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout
plt.tight_layout()

plt.savefig("sdplot_5_all.png", dpi=600, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("Summary: Density (fraction with regret > 0) for each filter")
print("="*70)

for row_idx, regret_col in enumerate(regret_cols):
    print(f"\n{row_labels[row_idx]}:")
    print("-" * 50)
    for col_idx, filter_name in enumerate(filter_names):
        filter_label = filter_name.replace('\n', ' ')
        density = all_densities[row_idx][col_idx]
        n_points = len(all_box_data[row_idx][col_idx])
        print(f"  {filter_label}: density={density:.3f} (n={n_points})")
