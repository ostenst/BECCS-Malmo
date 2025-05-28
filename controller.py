import numpy as np
from model import *
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    ScalarOutcome,
    ArrayOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments
)
from ema_workbench.em_framework import get_SALib_problem
from SALib.analyze import sobol
import matplotlib.cm as cm
import matplotlib.colors as mcolors

model = Model("BECCSMalmo", function=regret_BECCS)

model.uncertainties = [
    RealParameter("O2eff", 0.85, 0.95),     # [-] for CLC
    RealParameter("Wasu", 800, 900),        # MJ/tO2 (converted from 230*3.6, Macroscopic or Lyngfelt or Ramboll)

    RealParameter("operating", 4000, 5000), # Operating hours
    RealParameter("dr", 0.05, 0.10),        # Discount rate
    IntegerParameter("lifetime", 20, 30),      # Economic lifetime
    RealParameter("celc", 10, 120),         # Beiron tycker 100?
    RealParameter("cheat", 0.50, 0.95),
    RealParameter("cbio", 20, 100),         # BEIRON, insikter
    RealParameter("CEPCI", 700, 900),
    # RealParameter("sek", 0.08, 0.10),
    # RealParameter("usd", 0.90, 1.00),
    RealParameter("ctrans", 554, 755),        # SEK/tCO2, Kjärstad @Gävle 555nm, INCLUDES C&L???
    RealParameter("cstore", 201, 496),        # SEK/tCO2, Kjärstad @NL, calculate by total_system - only_transport
    RealParameter("crc", 25, 300),          # Reference cost
    RealParameter("cmea", 25, 35),         # SEK/kg, Ramboll
    RealParameter("coc", 200, 600),         # EUR/t, Magnus/Felicia

    RealParameter("cAM", 1723, 2585),       # +-20% of Ramboll CAPEX
    RealParameter("cFR", 0.48, 0.72),       # +-20% of Macroscopic CAPEX exponent   
    RealParameter("cASU", 0.68, 1.02),      # +-20% of Macroscopic CAPEX exponent   

    RealParameter("EPC", 0.14, 0.21),       # +-20% of Macroscopic EPC
    RealParameter("contingencies", 0.15, 0.35), # Ramboll uses 25%
    RealParameter("ownercost", 0.03, 0.07),             # Ramboll uses 5%
    RealParameter("overrun", 0.00, 0.45),           #Beiron FOAK=NOAK costs
    RealParameter("immature", 0.00, 3.00),       

    RealParameter("EUA", 0, 10),    # +ETS increases
    RealParameter("ceiling", 200, 350),

    CategoricalParameter("Bioshortage", [True, False]),
    CategoricalParameter("Powersurge", [True, False]),
    CategoricalParameter("Auction", [True, False]),
    # CategoricalParameter("Denial", [True, False]),
    CategoricalParameter("Integration", [True, False]),
    # CategoricalParameter("Capping", [True, False]),
    CategoricalParameter("Procurement", [True, False]),
    CategoricalParameter("Time", ["Baseline", "Downtime", "Uptime"]),
]
print("Need to re-run the controller after implementing Procure/Capping variables ... next revision!")

model.levers = [
    # CategoricalParameter("decision", ["ref", "amine", "oxy", "clc"]),
    RealParameter("rate", 0.86, 0.94),      # High capture rates needed
    CategoricalParameter("timing", [5, 10, 15, 20]),  # When capture + storage starts
]

model.outcomes = [
    ScalarOutcome("regret_1", ScalarOutcome.MINIMIZE),
    ScalarOutcome("regret_2", ScalarOutcome.MINIMIZE),
    ScalarOutcome("regret_3", ScalarOutcome.MINIMIZE),

    ScalarOutcome("npv_ref", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("npv_amine", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("npv_oxy", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("npv_clc", ScalarOutcome.MAXIMIZE),
]

# model.constants = [
#     RealParameter("sek", 0.089),
#     RealParameter("usd", 0.96),
# ]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 500
n_policies = 20
# Regular LHS sampling:
results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results

outcomes_df = pd.DataFrame(outcomes)
experiments.to_csv("experiments.csv", index=False)
outcomes_df.to_csv("outcomes.csv", index=False)

outcomes_df["Auction"] = experiments["Auction"]
outcomes_df["Time"] = experiments["Time"]
outcomes_df["cASU"] = experiments["cASU"]
outcomes_df["timing"] = experiments["timing"]
outcomes_df["immature"] = experiments["immature"]
print(outcomes_df)


## ------------------ PLOTTING REGRET_1 ------------------ ##
# Define the subsets
subsets = {
    "All Data": outcomes_df,
    "Time = Downtime": outcomes_df[outcomes_df["Time"] == "Downtime"],
    "Auction = False and Time = Downtime": outcomes_df[
        (outcomes_df["Auction"] == False) & (outcomes_df["Time"] == "Downtime")
    ]
}

# Set up color mapping
regret_col = "regret_1"
cmap = cm.RdYlGn_r
cmap = cm.coolwarm
norm = mcolors.Normalize(vmin=0, vmax=1)

ymin, ymax = -800, 800

# Prepare data for plotting (reversed order)
plot_data = []
box_colors = []
labels = ["3", "2", "1"]  # Reversed labels to match reversed plotting order

# Reverse the subset order
for subset in reversed(list(subsets.values())):
    plot_data.append(subset[regret_col])
    regret_freq = (subset[regret_col] > 0).mean()
    box_colors.append(cmap(regret_freq))

# Plot
fig, ax = plt.subplots(figsize=(10, 4.5))  # Horizontal layout

positions = [1, 2, 3]  # Position still 1-3, just reversed data
box = ax.boxplot(
    plot_data,
    patch_artist=True,
    labels=labels,
    positions=positions,
    widths=0.8,
    vert=False  # Horizontal boxes
)

# Color boxes and fix median lines
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)

# Vertical line at 0
ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

# X-axis and font styling
ax.set_xlim(ymin, ymax)
ax.set_xlabel("Regret", fontsize=13)
ax.set_title(f"Regret Distribution for '{regret_col}'", fontsize=15)
ax.tick_params(axis='both', labelsize=11)

# Colorbar with thicker width
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=1.0, aspect=25)  # Thicker via `aspect`
cbar.set_label("% of Regret > 0", rotation=270, labelpad=20, fontsize=13)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in cbar.get_ticks()])

plt.tight_layout()
plt.savefig("regret_1.png", dpi=600)

## ------------------ PLOTTING REGRET_2 ------------------ ##
# Define the subsets
subsets = {
    "All Data": outcomes_df,
    "cASU_low": outcomes_df[outcomes_df["cASU"] < 0.85],
    "timing_early and cASU": outcomes_df[
        ((outcomes_df["timing"] == 5) | (outcomes_df["timing"] == 10)) &
        (outcomes_df["cASU"] < 0.85)
    ]
}

# Set up color mapping
regret_col = "regret_2"
cmap = cm.RdYlGn_r
cmap = cm.coolwarm
norm = mcolors.Normalize(vmin=0, vmax=1)

ymin, ymax = -250, 250

# Prepare data for plotting (reversed order)
plot_data = []
box_colors = []
labels = ["3", "2", "1"]  # Reversed labels to match reversed plotting order

# Reverse the subset order
for subset in reversed(list(subsets.values())):
    plot_data.append(subset[regret_col])
    regret_freq = (subset[regret_col] > 0).mean()
    box_colors.append(cmap(regret_freq))

# Plot
fig, ax = plt.subplots(figsize=(10, 4.5))  # Horizontal layout

positions = [1, 2, 3]  # Position still 1-3, just reversed data
box = ax.boxplot(
    plot_data,
    patch_artist=True,
    labels=labels,
    positions=positions,
    widths=0.8,
    vert=False  # Horizontal boxes
)

# Color boxes and fix median lines
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)

# Vertical line at 0
ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

# X-axis and font styling
ax.set_xlim(ymin, ymax)
ax.set_xlabel("Regret", fontsize=13)
ax.set_title(f"Regret Distribution for '{regret_col}'", fontsize=15)
ax.tick_params(axis='both', labelsize=11)

# Colorbar with thicker width
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=1.0, aspect=25)  # Thicker via `aspect`
cbar.set_label("% of Regret > 0", rotation=270, labelpad=20, fontsize=13)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in cbar.get_ticks()])

plt.tight_layout()
plt.savefig("regret_2.png", dpi=600)

## ------------------ PLOTTING REGRET_3 ------------------ ##
# Define the subsets
subsets = {
    "All Data": outcomes_df,
    "immature_low": outcomes_df[outcomes_df["immature"] < 1.50],
    # "timing_early": outcomes_df[
    #     (outcomes_df["timing"] == 5) | (outcomes_df["timing"] == 10)
    # ],
    "timing_early and immature": outcomes_df[
        ((outcomes_df["timing"] == 5) | (outcomes_df["timing"] == 10)) &
        (outcomes_df["immature"] < 1.50)
    ]
}

# Set up color mapping
regret_col = "regret_3"
cmap = cm.RdYlGn_r
cmap = cm.coolwarm
norm = mcolors.Normalize(vmin=0, vmax=1)

ymin, ymax = -250, 250

# Prepare data for plotting (reversed order)
plot_data = []
box_colors = []
labels = ["3", "2", "1"]  # Reversed labels to match reversed plotting order

# Reverse the subset order
for subset in reversed(list(subsets.values())):
    plot_data.append(subset[regret_col])
    regret_freq = (subset[regret_col] > 0).mean()
    box_colors.append(cmap(regret_freq))

# Plot
fig, ax = plt.subplots(figsize=(10, 4.5))  # Horizontal layout

positions = [1, 2, 3]  # Position still 1-3, just reversed data
box = ax.boxplot(
    plot_data,
    patch_artist=True,
    labels=labels,
    positions=positions,
    widths=0.8,
    vert=False  # Horizontal boxes
)

# Color boxes and fix median lines
for patch, color in zip(box['boxes'], box_colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)

# Vertical line at 0
ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

# X-axis and font styling
ax.set_xlim(ymin, ymax)
ax.set_xlabel("Regret", fontsize=13)
ax.set_title(f"Regret Distribution for '{regret_col}'", fontsize=15)
ax.tick_params(axis='both', labelsize=11)

# Colorbar with thicker width
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=1.0, aspect=25)  # Thicker via `aspect`
cbar.set_label("% of Regret > 0", rotation=270, labelpad=20, fontsize=13)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in cbar.get_ticks()])

plt.tight_layout()
plt.savefig("regret_3.png", dpi=600)


## ------------- SEPARATE COLORBAR -------------
# Use the exact same colormap and normalization from the boxplot
cmap = cm.RdYlGn_r
cmap = cm.coolwarm
norm = mcolors.Normalize(vmin=0, vmax=1)

# Create separate figure for colorbar
fig, ax = plt.subplots(figsize=(2, 6))  # Adjust width and height as needed

# ScalarMappable to link color map and normalization
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required even if we don’t use it for plotting data

# Create colorbar
cbar = fig.colorbar(sm, cax=ax, orientation="vertical")
cbar.set_label("% of Regret > 0", rotation=270, labelpad=20, fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Set ticks as percentage labels
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in cbar.get_ticks()])
plt.tight_layout()

plt.show()