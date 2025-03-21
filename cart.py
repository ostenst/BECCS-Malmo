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
    result = data["regret"]
    classes = result == 0
    return classes

experiments = pd.read_csv("experiments.csv")
outcomes = pd.read_csv("outcomes.csv")
results = (experiments, outcomes)

# # Convert boolean columns in experiments to integers (1 and 0)
# experiments = experiments.astype({col: int for col in experiments.select_dtypes(include=bool).columns})

# # extract results for 1 policy
# logical = experiments["decision"] == "ref"
# new_experiments = experiments[logical]
# new_outcomes = {}
# for key, value in outcomes.items():
#     new_outcomes[key] = value[logical]

# results = (new_experiments, new_outcomes)

# perform cart on modified results tuple
cart_alg = cart.setup_cart(results, classify, mass_min=0.05)
cart_alg.build_tree()
print(dir(cart_alg))
df = cart_alg.boxes_to_dataframe()
cart_alg.show_boxes(together=False)
cart_alg.show_tree()

# Print general information about the DataFrame
print("DataFrame Info:")
print(df.info())  # Column names, data types, non-null values

# Print the first few rows to inspect structure
print("\nFirst 5 rows of the DataFrame:")
print(df.head())
for col in df.columns:
    print(f"\n===== Column: {col} =====")
    print(df[col])  # Print all values in the column
    
plt.show()