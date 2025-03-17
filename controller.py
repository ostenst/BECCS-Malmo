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
    RealParameter("ctrans", 40, 80),        # Transport cost
    RealParameter("cstore", 15, 35),        # Storage cost
    RealParameter("crc", 50, 250),          # Reference cost
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
    CategoricalParameter("Invasion", [True, False]),
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
]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 100
n_policies = 40

results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results

outcomes_df = pd.DataFrame(outcomes)
experiments.to_csv("experiments.csv", index=False)
outcomes_df.to_csv("outcomes.csv", index=False)
outcomes_df["decision"] = experiments["decision"]
print(outcomes_df.head())

# zero_regret_counts = outcomes_df[outcomes_df["regret_decision"] == 0].groupby("decision")["regret_decision"].count()
# print(zero_regret_counts)

plt.figure(figsize=(8, 4))
sns.boxplot(x="decision", y=outcomes_df["regret"], data=outcomes_df)
plt.title(f"Regret (based on NPV in [MEUR]) by technology decision (circle=outlier)")

# Creating 8 bar plots for regret_ref based on combinations of Bioshortage, Powersurge, Invasion, and Auction
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)

# Define scenarios
scenarios = [
    (False, False, False, False),
    (False, False, False, True),
    (False, False, True, False),
    (False, False, True, True),
    (True, False, False, False),
    (True, False, False, True),
    (True, False, True, False),
    (True, False, True, True),
]

# Loop through each scenario and plot
for ax, (bio, power, inv, auc) in zip(axes.flatten(), scenarios):
    subset = outcomes_df[
        (experiments["Bioshortage"] == bio) &
        (experiments["Powersurge"] == power) &
        (experiments["Invasion"] == inv) &
        (experiments["Auction"] == auc)
    ]
    
    sns.barplot(x=subset["decision"], y=subset["regret"], ax=ax)
    ax.set_title(f"B:{bio}, P:{power}, I:{inv}, A:{auc}")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Regret")

plt.tight_layout()
plt.show()