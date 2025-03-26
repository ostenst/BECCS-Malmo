import numpy as np
from model import *
import matplotlib.pyplot as plt
import seaborn as sns
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
    RealParameter("ctrans", 550, 650),        # SEK/tCO2, Kjärstad
    RealParameter("cstore", 10, 25),        # Storage cost, ZEP
    RealParameter("crc", 50, 300),          # Reference cost
    RealParameter("cmea", 25, 35),         # SEK/kg, Ramboll
    RealParameter("coc", 400, 600),         # EUR/t, Magnus/Felicia

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
    ScalarOutcome("regret", ScalarOutcome.MINIMIZE),
    ScalarOutcome("regret_ref", ScalarOutcome.MINIMIZE),
    ScalarOutcome("regret_amine", ScalarOutcome.MINIMIZE),
    ScalarOutcome("regret_oxy", ScalarOutcome.MINIMIZE),
    ScalarOutcome("regret_clc", ScalarOutcome.MINIMIZE),

    ScalarOutcome("npv_ref", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("npv_amine", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("npv_oxy", ScalarOutcome.MAXIMIZE),
    ScalarOutcome("npv_clc", ScalarOutcome.MAXIMIZE),
]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 1000
n_policies = 500

# Regular LHS sampling:
results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results

outcomes_df = pd.DataFrame(outcomes)
experiments.to_csv("experiments.csv", index=False)
outcomes_df.to_csv("outcomes.csv", index=False)
outcomes_df["decision"] = experiments["decision"]
print(outcomes_df)

zero_regret_counts = outcomes_df[outcomes_df["regret"] == 0].groupby("decision")["regret"].count()
print(zero_regret_counts)

# Create new columns based on npv_ref and cbio/celc comparison
outcomes_df["npv_ref_bio"] = outcomes_df["npv_ref"].where(experiments["cbio"] > experiments["celc"])
outcomes_df["npv_ref_elc"] = outcomes_df["npv_ref"].where(experiments["cbio"] < experiments["celc"])
print(outcomes_df[["npv_ref", "npv_ref_bio", "npv_ref_elc"]].head())

# Define regret columns
regret_columns = ["npv_ref_bio", "npv_ref_elc", "npv_amine", "npv_oxy", "npv_clc"]

# Merge experiments and outcomes by index
df = pd.concat([experiments[["Auction", "Bioshortage"]], outcomes_df[regret_columns]], axis=1)

# Define subsets based on Auction and Bioshortage values
subsets = {
    "Auction=False, Bioshortage=False": df[(df["Auction"] == False) & (df["Bioshortage"] == False)],
    "Auction=False, Bioshortage=True": df[(df["Auction"] == False) & (df["Bioshortage"] == True)],
    "Auction=True, Bioshortage=False": df[(df["Auction"] == True) & (df["Bioshortage"] == False)],
    "Auction=True, Bioshortage=True": df[(df["Auction"] == True) & (df["Bioshortage"] == True)]
}

# Get global min/max values for y-axis synchronization
global_min = df[regret_columns].min().min()
global_max = df[regret_columns].max().max()

column_means = df[regret_columns].mean()

# Normalize mean values for colormap mapping
norm = mcolors.Normalize(vmin=column_means.min(), vmax=column_means.max())
cmap = cm.get_cmap("RdYlGn")  # Red-Yellow-Green colormap

# Generate dynamic colors based on means
box_colors = [mcolors.to_hex(cmap(norm(value))) for value in column_means]

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)  # Synchronize y-axis

# Loop through subsets and plot boxplots
for ax, (title, subset) in zip(axes.flatten(), subsets.items()):
    sns.boxplot(data=subset[regret_columns], ax=ax, palette=box_colors)
    ax.set_title(title)
    ax.set_ylabel("NPV Values")
    ax.set_xticklabels(regret_columns, rotation=20)
    ax.set_ylim(global_min - 50, global_max)  # Ensure same y-axis scale
    ax.axhline(0, color="black", linestyle="dashed", linewidth=1, alpha=0.8)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
