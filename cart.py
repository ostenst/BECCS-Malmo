"""
Created on May 26, 2015

@author: jhkwakkel
"""
import pandas as pd
import matplotlib.pyplot as plt

import ema_workbench.analysis.cart as cart
from ema_workbench import ema_logging, load_results

ema_logging.log_to_stderr(level=ema_logging.INFO)


def classify(data):
    # Get the regret_ref outcome
    result = data["regret_ref"]
    classes = result == 0
    return classes

experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")

# Convert boolean columns in experiments to integers (1 and 0)
experiments = experiments.astype({col: int for col in experiments.select_dtypes(include=bool).columns})

# extract results for 1 policy
logical = experiments["decision"] == "ref"
new_experiments = experiments[logical]
new_outcomes = {}
for key, value in outcomes.items():
    new_outcomes[key] = value[logical]

results = (new_experiments, new_outcomes)

# perform cart on modified results tuple
cart_alg = cart.setup_cart(results, classify, mass_min=0.05)
cart_alg.build_tree()
print(dir(cart_alg))

cart_alg.show_boxes(together=False)
cart_alg.show_tree()
plt.show()