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
    RealParameter("Wasu", 800, 900),        # MJ/tO2 (converted from 230*3.6)

    RealParameter("dr", 0.06, 0.08),        # Discount rate
    IntegerParameter("lifetime", 20, 30),      # Economic lifetime
    RealParameter("celc", 35, 45),
    RealParameter("cheat", 0.75, 0.85),
    RealParameter("cbio", 20, 30),
    RealParameter("CEPCI", 700, 900),
    RealParameter("sek", 0.08, 0.10),
    RealParameter("usd", 0.90, 1.00),
    RealParameter("ctrans", 40, 60),        # Transport cost
    RealParameter("cstore", 25, 35),        # Storage cost
    RealParameter("crc", 90, 110),          # Reference cost

    RealParameter("EPC", 0.15, 0.20),
    RealParameter("contingency_process", 0.03, 0.07),
    RealParameter("contingency_clc", 0.35, 0.45),
    RealParameter("contingency_project", 0.15, 0.25),
    RealParameter("ownercost", 0.15, 0.25),
]

model.levers = [
    CategoricalParameter("decision", ["ref", "amine", "oxy", "clc"]),
    RealParameter("rate", 0.86, 0.94),      # High capture rates needed
    RealParameter("operating", 4000, 5000), # Operating hours
    CategoricalParameter("operating_increase", [0, 600, 1200]),
    CategoricalParameter("timing", [5, 10, 15, 20]),  # When capture + storage starts
]

model.outcomes = [
    ScalarOutcome("regret_decision", ScalarOutcome.MINIMIZE),
]

model.constants = [
    Constant("Qnet", 140),
    Constant("P", 48.3),
    Constant("Qfuel", 174),
    Constant("LHV", 10.44),
    Constant("psteam", 95),
    Constant("Tsteam", 525),
    Constant("isentropic", 0.85), 
]

ema_logging.log_to_stderr(ema_logging.INFO)
n_scenarios = 80
n_policies = 50

results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
experiments, outcomes = results
print(type(outcomes["regret_decision"]), outcomes["regret_decision"])

outcomes_df = pd.DataFrame(outcomes)
outcomes_df["decision"] = experiments["decision"]

# Use boxplot to show distributions of all numerical outcomes grouped by decision
for outcome in outcomes_df.keys():
    if outcome != "decision":  # Exclude categorical variable
        plt.figure(figsize=(8, 4))
        sns.boxplot(x="decision", y=outcome, data=outcomes_df)
        plt.title(f"Distribution of Regret (based on NPV difference in [MEUR]) by CAPTURE TECHNOLOGY")
        plt.show()
