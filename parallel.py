import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

outcomes = outcomes.loc[experiments.index]
outcomes['decision'] = experiments['decision']

# Normalize numerical columns
scaler = MinMaxScaler()
if 'regret_ref' in outcomes.columns:
    outcomes["regret_ref_scaled"] = scaler.fit_transform(outcomes[["regret_ref"]])

categorical_columns = ['decision', 'operating_increase']
label_encoder = LabelEncoder()
categorical_data = experiments[categorical_columns].apply(label_encoder.fit_transform)
categorical_scaled = pd.DataFrame(scaler.fit_transform(categorical_data), columns=categorical_columns)

data_scaled = pd.concat([categorical_scaled, outcomes[["regret_ref_scaled"]]], axis=1)

# Define colormap and specific colors
USE_COLORMAP = False  # Set to False to use predefined colors instead of colormap
viridis = cm.get_cmap('viridis')
decision_colors = {
    "ref": viridis(0.0),
    "amine": viridis(0.33),
    "clc": viridis(0.66),
    "oxy": viridis(1.0),
}
def get_color(row):
    if USE_COLORMAP:
        color_value = row["regret_ref_scaled"]  # Use regret_ref for color mapping
        return viridis(color_value), 0.8  # Higher alpha for better visibility
    else:
        if row['regret_ref'] != 0:
            return "red", 0.8
        else:
            return "grey", 0.4

        # return decision_colors.get(row['decision'], "grey"), 0.8  # Default color is grey

colors = outcomes.apply(get_color, axis=1)

# Set up plot
fig, ax = plt.subplots(figsize=(8, 5))
for i, row in data_scaled.iterrows():
    ax.plot(data_scaled.columns, row, color=colors[i][0], alpha=colors[i][1])

# Add colorbar if using colormap
if USE_COLORMAP:
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=outcomes["regret_ref"].min(), vmax=outcomes["regret_ref"].max()))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Regret (REF)")

# Add legend for decision-based colors if colormap is not used
if not USE_COLORMAP:
    legend_patches = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in decision_colors.items()]
    ax.legend(handles=legend_patches, title="Decision")

# Set x-axis labels
ax.set_xticks(range(len(data_scaled.columns)))
ax.set_xticklabels(data_scaled.columns, rotation=45)

# Remove black box (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title("Parallel Coordinates Plot for CHP Outcomes")

plt.show()
