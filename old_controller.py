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
    RealParameter("Wasu", 800, 900),        # MJ/tO2 (converted from 230*3.6, Macroscopic or Lyngfelt)

    RealParameter("operating", 4000, 5000), # Operating hours
    RealParameter("dr", 0.05, 0.10),        # Discount rate
    IntegerParameter("lifetime", 20, 30),      # Economic lifetime
    RealParameter("celc", 30, 80),
    RealParameter("cheat", 0.50, 0.95),
    RealParameter("cbio", 20, 50),
    RealParameter("CEPCI", 700, 900),
    RealParameter("sek", 0.08, 0.10),
    RealParameter("usd", 0.90, 1.00),
    RealParameter("ctrans", 40, 70),        # Transport cost
    RealParameter("cstore", 15, 35),        # Storage cost
    RealParameter("crc", 60, 200),          # Reference cost
    RealParameter("cmea", 25, 35),         # SEK/kg, Ramboll
    RealParameter("coc", 400, 600),         # EUR/t, Magnus/Felicia

    RealParameter("EPC", 0.15, 0.25),
    RealParameter("contingency_process", 0.03, 0.07),
    RealParameter("contingency_clc", 0.30, 0.50),
    RealParameter("contingency_project", 0.15, 0.25),
    RealParameter("ownercost", 0.15, 0.25),
]

model.levers = [
    CategoricalParameter("decision", ["ref", "amine", "oxy", "clc"]),
    RealParameter("rate", 0.86, 0.94),      # High capture rates needed
    CategoricalParameter("operating_increase", [0, 600, 1200]),
    CategoricalParameter("timing", [5, 10, 15, 20]),  # When capture + storage starts
]

model.outcomes = [
    ScalarOutcome("regret_decision", ScalarOutcome.MINIMIZE),
    ScalarOutcome("amine_capex", ScalarOutcome.MINIMIZE),
    ScalarOutcome("clc_capex", ScalarOutcome.MINIMIZE),
    ScalarOutcome("oxy_capex", ScalarOutcome.MINIMIZE),
    ScalarOutcome("ref_capex", ScalarOutcome.MINIMIZE),
]

# model.constants = [
# ]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 100
n_policies = 100

results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results
# print(type(outcomes["regret_decision"]), outcomes["regret_decision"])

outcomes_df = pd.DataFrame(outcomes)
outcomes_df["decision"] = experiments["decision"]
print(outcomes_df.head())

# # Use boxplot to show distributions of all numerical outcomes grouped by decision
# for outcome in outcomes_df.keys():
#     if outcome != "decision":  # Exclude categorical variable
plt.figure(figsize=(8, 4))
sns.boxplot(x="decision", y=outcomes_df["regret_decision"], data=outcomes_df)
plt.title(f"Regret (based on NPV in [MEUR]) by technology decision (circle=outlier)")
        

# Analyze the "ref" data separately:
ref_outcomes_df = outcomes_df[outcomes_df["decision"] == "ref"]
print("Descriptive Statistics for regret_decision when decision='ref':")
print(ref_outcomes_df["regret_decision"].describe())
regret_zero_count = (ref_outcomes_df["regret_decision"] == 0).sum()
total_count = len(ref_outcomes_df)
print(f"Instances where regret = 0: {regret_zero_count}/{total_count}")

plt.figure(figsize=(8, 5))
sns.histplot(ref_outcomes_df["regret_decision"], bins=15, kde=True, color="red")
plt.axvline(0, color="red", linestyle="dashed")  # Mark regret = 0
plt.title('Histogram of Regret for REFERENCE')
plt.xlabel('Regret in NPV (MEUR)')
plt.ylabel('Frequency')

# AND OF CLC????
plt.figure(figsize=(8, 5))
sns.histplot(outcomes_df[outcomes_df["decision"] == "clc"]["regret_decision"], bins=15, kde=True, color="orange")
plt.axvline(0, color="red", linestyle="dashed")  # Mark regret = 0
plt.title('Histogram of Regret for CLC')
plt.xlabel('Regret in NPV (MEUR)')
plt.ylabel('Frequency')
# AND OF OXY????
plt.figure(figsize=(8, 5))
sns.histplot(outcomes_df[outcomes_df["decision"] == "oxy"]["regret_decision"], bins=15, kde=True, color="blue")
plt.axvline(0, color="red", linestyle="dashed")  # Mark regret = 0
plt.title('Histogram of Regret for OXY')
plt.xlabel('Regret in NPV (MEUR)')
plt.ylabel('Frequency')
plt.figure(figsize=(8, 5))
# AND OF AMINE????
sns.histplot(outcomes_df[outcomes_df["decision"] == "amine"]["regret_decision"], bins=15, kde=True, color="green")
plt.axvline(0, color="red", linestyle="dashed")  # Mark regret = 0
plt.title('Histogram of Regret for AMINE')
plt.xlabel('Regret in NPV (MEUR)')
plt.ylabel('Frequency')

# Boxplot for all outcomes (each outcome on x-axis)
plt.figure(figsize=(8, 5))
melted_df = outcomes_df.drop(columns=["decision", "regret_decision"]).melt(var_name="Outcome", value_name="Value")
sns.boxplot(x="Outcome", y="Value", data=melted_df)
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
plt.title("Distribution of CAPEX per technology decision")
plt.ylabel("CAPEX [MEUR]")
plt.xlabel("Outcomes")

offsets = {"ref": -0.6, "amine": -0.2, "oxy": 0.2, "clc": 0.6}

x_numeric = pd.to_numeric(experiments["timing"], errors="coerce")

# Ensure mapped offsets are numeric before adding
offset_values = experiments["decision"].map(offsets).astype(float)

# Apply jitter based on decision category
x_jittered = x_numeric + offset_values

plt.figure(figsize=(8, 5))
scatter = sns.scatterplot(
    x=x_jittered,
    y=outcomes_df["regret_decision"],
    hue=experiments["decision"],
    palette="tab10",
    edgecolor="black",
    alpha=0.2
)
for legend_handle in scatter.legend_.legend_handles:
    legend_handle.set_alpha(1)

plt.xlabel("Timing of Investment [years after boiler investment]")
plt.ylabel("Regret in NPV (MEUR)")
plt.title("Regret vs. Timing of Investment")
plt.xticks(sorted(x_numeric.unique()))  # Keep original timing values on x-axis
plt.legend(title="Decision")

plt.show()