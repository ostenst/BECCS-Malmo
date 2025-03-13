import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.patches as mpatches

# Load datasets
experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

# Ensure dataframes are aligned by index
outcomes = outcomes.loc[experiments.index]
outcomes['decision'] = experiments['decision']
outcomes['operating_increase'] = experiments['operating_increase']

numerical_columns = ['regret_decision']
categorical_columns = ['decision', 'operating_increase']

# Normalize
scaler = MinMaxScaler()
data_scaled_numerical = pd.DataFrame(scaler.fit_transform(outcomes[numerical_columns]), columns=numerical_columns)
viridis = cm.get_cmap('viridis')
# capture_costs = filtered_outcomes['capture_cost']
# capture_costs_norm = (capture_costs - capture_costs.min()) / (capture_costs.max() - capture_costs.min())  # Normalize 0-1
# colors = cm.viridis(capture_costs_norm)  # Use colormap (viridis) to assign colors

# Normalize categorical columns using LabelEncoder and then MinMaxScaler
label_encoder = LabelEncoder()

# Apply LabelEncoder to categorical columns and then normalize using MinMaxScaler
categorical_data = experiments[categorical_columns].apply(label_encoder.fit_transform)
categorical_scaled = pd.DataFrame(scaler.fit_transform(categorical_data), columns=categorical_columns)

# Combine numerical and categorical data into one dataframe
data_scaled = pd.concat([data_scaled_numerical, categorical_scaled], axis=1)

# Define color function (adjust this condition as needed)
def get_color(row):
    if row['decision'] == "ref":  
        return viridis(0.0), 0.4
    elif row['decision'] == "amine": 
        return viridis(0.33), 0.4
    elif row['decision'] == "oxy": 
        return viridis(0.66), 0.4
    elif row['decision'] == "clc": 
        return viridis(1.0), 0.4
    # if row['duration_increase'] == 1000 and row['heat_pump'] == True:  
    #     return viridis(0.5), 0.9
    # else:
    #     return "grey", 0.4

# Generate colors based on conditions
colors = [get_color(row) for _, row in outcomes.iterrows()]

# Set up plot (Assuming you want to plot, like in previous examples)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

# Plot parallel coordinates (assuming you already have data_scaled and colors from earlier)
for i, row in data_scaled.iterrows():
    ax.plot(data_scaled.columns, row, color=colors[i][0], alpha=colors[i][1])

# Add min, 1/3, 2/3, and max tick labels for each parameter (as in previous examples)
for i, column in enumerate(data_scaled.columns):
    if column in numerical_columns:
        min_val, max_val = outcomes[column].min(), outcomes[column].max()
        tick_values = np.linspace(min_val, max_val, 4)  # Evenly spaced values
        tick_positions = np.linspace(0, 1, 4)  # Normalized positions

        for pos, val in zip(tick_positions, tick_values):
            ax.text(i, pos, f"{val:.1f}", ha='center', va='center', fontsize=10, color='black')
    ax.plot([i, i], [0, 1], color='black', linestyle='-', alpha=1)

# Add x-axis labels
ax.set_xticks(range(len(data_scaled.columns)))
ax.set_xticklabels(data_scaled.columns, rotation=45)

# Remove black box (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_ylabel("Normalized Scale")
ax.set_title("Parallel Coordinates Plot for CHP Outcomes")

legend_patches = [
    mpatches.Patch(color=viridis(0.0), label="ref"),
    mpatches.Patch(color=viridis(0.33), label="amine"),
    mpatches.Patch(color=viridis(0.66), label="oxy"),
    mpatches.Patch(color=viridis(1.0), label="clc"),
]

# Add legend to the plot
ax.legend(handles=legend_patches, title="Decision", loc="upper right")

plt.show()

