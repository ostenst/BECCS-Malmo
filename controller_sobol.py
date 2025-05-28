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

    RealParameter("rate", 0.86, 0.94),      # High capture rates needed
    CategoricalParameter("timing", [5, 10, 15, 20]),  # When capture + storage starts
]

model.levers = [
    # CategoricalParameter("decision", ["ref", "amine", "oxy", "clc"]),
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
n_scenarios = 3000
n_policies = 0

# If Sobol sampling:
print(" NOTE : You must specify 1 lever to analyze sensitivity on")
results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.SOBOL, lever_sampling = Samplers.SOBOL)
experiments, outcomes = results

def analyze(results, ooi):
    """analyze results using SALib sobol, returns a dataframe"""
    _, outcomes = results

    problem = get_SALib_problem(model.uncertainties)
    y = outcomes[ooi]
    sobol_indices = sobol.analyze(problem, y)
    sobol_stats = {key: sobol_indices[key] for key in ["ST", "ST_conf", "S1", "S1_conf"]}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
    sobol_stats.sort_values(by="ST", ascending=False)
    s2 = pd.DataFrame(sobol_indices["S2"], index=problem["names"], columns=problem["names"])
    s2_conf = pd.DataFrame(
        sobol_indices["S2_conf"], index=problem["names"], columns=problem["names"]
    )
    return sobol_stats, s2, s2_conf, problem
sobol_stats, s2, s2_conf, problem = analyze(results, "regret_2")
print(sobol_stats)
print(s2)
print(s2_conf)
sobol_stats = pd.DataFrame(sobol_stats, index=problem["names"])
sobol_stats.to_csv("sobol_stats.csv")
sobol_stats_sorted = sobol_stats.sort_values(by="ST", ascending=False)  # Ascending for better readability

# Create horizontal bar plot
plt.figure(figsize=(8, 10))  # Adjust figure size for better layout
sns.barplot(
    y=sobol_stats_sorted.index,  # Parameters on y-axis
    x=sobol_stats_sorted["ST"],  # Sobol indices on x-axis
    xerr=sobol_stats_sorted["ST_conf"],  # Confidence intervals as error bars
    capsize=0.2,
    color="crimson"
)
plt.ylabel("Parameter")
plt.xlabel("Total Sobol Index (ST)")
plt.title("Total-Order Sobol Indices with Confidence Intervals")
plt.grid(axis="x", linestyle="--", alpha=0.7)


# ---- Sobol Circular Radial Plot: Top 10 ST with S1 overlay and S2 lines ----
# Sort and select top 10 parameters based on ST
top_10 = sobol_stats.sort_values(by="ST", ascending=False).head(10)
labels = top_10.index.tolist()
ST = top_10["ST"].values
S1 = top_10["S1"].values

# Extract corresponding S2 matrix
indices = [sobol_stats.index.get_loc(label) for label in labels]
S2_matrix_top = s2.values[np.ix_(indices, indices)]

n = len(labels)

# Fixed circle layout
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
radius = 1.0
x = radius * np.cos(angles)
y = radius * np.sin(angles)

# Normalize ST and S1 for node sizes
min_size, max_size = 20, 2000
ST_scaled = min_size + (max_size - min_size) * (ST / ST.max())
S1_scaled = min_size + (max_size - min_size) * (S1 / S1.max())

# --- Plot ---
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.set_aspect('equal')
ax.axis('off')

# Plot S2 interaction edges
max_s2 = np.nanmax(S2_matrix_top)
for i in range(n):
    for j in range(i + 1, n):
        s2_val = S2_matrix_top[i, j]
        if not np.isnan(s2_val) and s2_val > 0.01:
            x_vals = [x[i], x[j]]
            y_vals = [y[i], y[j]]
            lw = 0.5 + 5 * (s2_val / max_s2)
            ax.plot(x_vals, y_vals, color='grey', alpha=0.6, linewidth=lw)

# Plot ST nodes (base)
ax.scatter(x, y, s=ST_scaled, color='crimson', alpha=1.0, edgecolor='none')

# Overlay S1 nodes (on top)
ax.scatter(x, y, s=S1_scaled, color='lightpink', alpha=1.0, edgecolor='none')

# Add labels
for i in range(n):
    ax.text(x[i]*1.5, y[i]*1.5, labels[i], ha='center', va='center', fontsize=7)
    if ST[i] > 0.08:
        ax.text(x[i], y[i], round(ST[i],2), color='black', ha='center', va='center', fontsize=7)

circle = plt.Circle((0, 0), 1.2, color='black', fill=False, linewidth=1.0, zorder=10)
ax.add_patch(circle)

# Set axis limits with margin
padding = 1.3  # Adjust this based on how big your nodes are
ax.set_xlim(-padding, padding)
ax.set_ylim(-padding, padding)

# Title
ax.set_title("Top 10 Parameters - Radial Sensitivity Map\nRed = ST, Blue = S1, Gray Lines = S2", fontsize=10, pad=20)

# plt.tight_layout()
plt.savefig('sobol.png', dpi=450, bbox_inches='tight')
plt.show()