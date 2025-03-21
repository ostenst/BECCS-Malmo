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
]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 400
n_policies = 75

# Regular LHS sampling:
results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results

# # If Sobol sampling:
# results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.SOBOL, lever_sampling = Samplers.SOBOL)
# experiments, outcomes = results
# def analyze(results, ooi):
#     """analyze results using SALib sobol, returns a dataframe"""
#     _, outcomes = results

#     problem = get_SALib_problem(model.uncertainties)
#     y = outcomes[ooi]
#     sobol_indices = sobol.analyze(problem, y)
#     sobol_stats = {key: sobol_indices[key] for key in ["ST", "ST_conf", "S1", "S1_conf"]}
#     sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
#     sobol_stats.sort_values(by="ST", ascending=False)
#     s2 = pd.DataFrame(sobol_indices["S2"], index=problem["names"], columns=problem["names"])
#     s2_conf = pd.DataFrame(
#         sobol_indices["S2_conf"], index=problem["names"], columns=problem["names"]
#     )
#     return sobol_stats, s2, s2_conf, problem
# sobol_stats, s2, s2_conf, problem = analyze(results, "regret")
# print(sobol_stats)
# print(s2)
# print(s2_conf)
# sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
# sobol_stats.to_csv("sobol_stats.csv")
# sobol_stats_sorted = sobol_stats.sort_values(by="ST", ascending=False)  # Ascending for better readability

# # Create horizontal bar plot
# plt.figure(figsize=(8, 10))  # Adjust figure size for better layout
# sns.barplot(
#     y=sobol_stats_sorted.index,  # Parameters on y-axis
#     x=sobol_stats_sorted["ST"],  # Sobol indices on x-axis
#     xerr=sobol_stats_sorted["ST_conf"],  # Confidence intervals as error bars
#     capsize=0.2,
#     color="crimson"
# )
# plt.ylabel("Parameter")
# plt.xlabel("Total Sobol Index (ST)")
# plt.title("Total-Order Sobol Indices with Confidence Intervals")
# plt.grid(axis="x", linestyle="--", alpha=0.7)
# plt.show()

outcomes_df = pd.DataFrame(outcomes)
experiments.to_csv("experiments.csv", index=False)
outcomes_df.to_csv("outcomes.csv", index=False)
outcomes_df["decision"] = experiments["decision"]
print(outcomes_df)

zero_regret_counts = outcomes_df[outcomes_df["regret"] == 0].groupby("decision")["regret"].count()
print(zero_regret_counts)

# sns.pairplot(outcomes_df, hue="decision", vars=list(outcomes.keys())) # This plots ALL outcomes
# #  sns.pairplot(df_outcomes, hue="heat_pump", vars=["capture_cost","penalty_services","penalty_biomass"])
# plt.show()