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
    RealParameter("celc", 10, 120),
    RealParameter("cheat", 0.50, 0.95),
    RealParameter("cbio", 20, 100),
    RealParameter("CEPCI", 700, 900),
    RealParameter("sek", 0.08, 0.10),
    RealParameter("usd", 0.90, 1.00),
    RealParameter("ctrans", 554, 755),        # SEK/tCO2, Kjärstad @Gävle 555nm, INCLUDES C&L???
    RealParameter("cstore", 201, 496),        # SEK/tCO2, Kjärstad @NL, calculate by total_system - only_transport
    RealParameter("crc", 50, 400),          # Reference cost
    RealParameter("cmea", 25, 35),         # SEK/kg, Ramboll
    RealParameter("coc", 200, 600),         # EUR/t, Magnus/Felicia

    RealParameter("cAM", 1, 2),
    RealParameter("cFR", 1, 2),
    RealParameter("cycl", 1, 2),
    RealParameter("cASU", 1, 2),

    RealParameter("EPC", 0.15, 0.25),
    RealParameter("contingency_process", 0.03, 0.07),
    RealParameter("contingency_clc", 0.30, 0.50),
    RealParameter("contingency_project", 0.15, 0.25),
    RealParameter("ownercost", 0.15, 0.25),

    CategoricalParameter("Bioshortage", [True, False]),
    CategoricalParameter("Powersurge", [True, False]),
    CategoricalParameter("Auction", [True, False]),
]

model.levers = [
    CategoricalParameter("decision", ["ref", "amine", "oxy", "clc"]),
    RealParameter("rate", 0.86, 0.94),      # High capture rates needed
    CategoricalParameter("operating_increase", [0, 600, 1200]),
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

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 800
n_policies = 300

# Regular LHS sampling:
results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results

outcomes_df = pd.DataFrame(outcomes)
experiments.to_csv("experiments.csv", index=False)
outcomes_df.to_csv("outcomes.csv", index=False)
outcomes_df["decision"] = experiments["decision"]
outcomes_df["Auction"] = experiments["Auction"]
outcomes_df["Bioshortage"] = experiments["Bioshortage"]
print(outcomes_df)

# ---- Plot 1: Regret boxplots with color legend ----
# Define regret columns
regret_cols = ["regret_1", "regret_2", "regret_3"]
cmap = cm.RdYlGn_r
norm = mcolors.Normalize(vmin=0, vmax=1)

# Loop over all combinations of Auction and Bioshortage
combinations = list(itertools.product([True, False], [True, False]))

for auction_val, bio_val in combinations:
    # Subset the dataframe
    subset = outcomes_df[(outcomes_df["Auction"] == auction_val) & (outcomes_df["Bioshortage"] == bio_val)]
    
    # Melt for plotting
    regret_df = subset[regret_cols].melt(var_name="Regret Type", value_name="Value")
    
    # Calculate regret frequencies
    regret_frequencies = (subset[regret_cols] > 0).mean()
    colors = [cmap(regret_frequencies[col]) for col in regret_cols]
    custom_palette = dict(zip(regret_cols, colors))
    
    # Create boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=regret_df, x="Regret Type", y="Value", palette=custom_palette)
    for y in [-200, 0, 200]:
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

    # Title and labels
    plt.title(f"Regret Distribution\nAuction: {auction_val}, Bioshortage: {bio_val}\n(Box Color = % Regret > 0)")
    plt.ylabel("Regret")
    plt.xlabel("Regret Type")
    
    # Colorbar legend
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label("% of Regret > 0", rotation=270, labelpad=15)
    cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in cbar.get_ticks()])
    
    # Layout and show
    plt.tight_layout()
# plt.show()

# Create horizontal colorbar figure
fig, ax = plt.subplots(figsize=(5, 1.3))  # Wider and short height
fig.subplots_adjust(bottom=0.5)

# Dummy mappable for colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create horizontal colorbar
cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label("% of Regret > 0", fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=14)  # Increase tick label size
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.set_xticklabels([f"{int(t * 100)}%" for t in cbar.get_ticks()])

plt.title("Colorbar Legend", fontsize=13, pad=10)
plt.tight_layout()
plt.show()