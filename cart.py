"""
Created on May 26, 2015

@author: jhkwakkel
"""
import pandas as pd
import matplotlib.pyplot as plt

import ema_workbench.analysis.cart as cart
from ema_workbench import ema_logging, load_results

ema_logging.log_to_stderr(level=ema_logging.INFO)

def filter_by_decision(experiments, outcomes, decision_values):
    """
    Filters the experiments and outcomes based on the categorical value of 'decision' and prints row counts.
    """

    initial_rows = len(experiments)

    # Ensure decision_values is a list
    if not isinstance(decision_values, list):
        decision_values = [decision_values]

    # Generate the correct column names (e.g., "decision_ref", "decision_amine", ...)
    decision_columns = [f"decision_{val}" for val in decision_values]

    # Create a filter mask: Select rows where at least one of the decision columns is 1
    mask = experiments[decision_columns].sum(axis=1) > 0

    # Apply filtering
    filtered_experiments = experiments[mask]
    filtered_outcomes = outcomes[mask]  # Keep outcomes aligned

    filtered_rows = len(filtered_experiments)

    print(f"Initial number of rows: {initial_rows}")
    print(f"Number of rows after filtering: {filtered_rows}")

    return filtered_experiments, filtered_outcomes

def classify(data):
    # Get the regret_ref outcome
    result = data["regret_1"]
    classes = result < 0
    return classes


def filter_by_box(results, df_boxes, box_name, **categorical_filters):
    """
    Filters the experiments and corresponding outcomes based on the numerical limits of a given box.
    
    Parameters:
    - results: tuple of (experiments, outcomes) dataframes
    - df_boxes: DataFrame containing box boundaries
    - box_name: Name of the box to filter by (e.g., "box 1")
    - categorical_filters: Manually specified categorical conditions 
      (e.g., decision=["oxy", "clc"], Auction=True)
    
    Returns:
    - (filtered_experiments, filtered_outcomes): Tuple of filtered DataFrames
    """
    experiments, outcomes = results  
    print("NOTE: the user must manually specify what categorical features to include")
    print(f"Original number of experiments: {len(experiments)}")

    min_cols = df_boxes.loc[:, (box_name, "min")]
    max_cols = df_boxes.loc[:, (box_name, "max")]

    mask = pd.Series(True, index=experiments.index)
    for col in min_cols.index:
        if col in experiments.columns and pd.api.types.is_numeric_dtype(experiments[col]):
            mask &= (experiments[col] >= min_cols[col]) & (experiments[col] <= max_cols[col])

    for cat_col, cat_value in categorical_filters.items():
        if cat_col in experiments.columns:
            if isinstance(cat_value, list):  # If multiple categories are specified
                mask &= experiments[cat_col].isin(cat_value)
            else:  # If it's a single value (string, boolean, etc.)
                mask &= experiments[cat_col] == cat_value

    filtered_experiments = experiments[mask]
    filtered_outcomes = outcomes.loc[mask] 

    print(f"Filtered number of experiments: {len(filtered_experiments)}")

    return filtered_experiments, filtered_outcomes

def filter_by_feature_limits(results, feature_limits):
    """
    Filters experiments and outcomes based on hardcoded feature limits.

    """
    experiments, outcomes = results
    initial_rows = len(experiments)

    mask = pd.Series(True, index=experiments.index)

    for feature, limit in feature_limits.items():
        if isinstance(limit, tuple):  # Numeric range (min, max)
            mask &= (experiments[feature] >= limit[0]) & (experiments[feature] <= limit[1])
        else:  # One-hot encoded categorical filter (1 or 0)
            mask &= (experiments[feature] == limit)

    filtered_experiments = experiments[mask]
    filtered_outcomes = outcomes[mask]  # Keep outcomes aligned

    filtered_rows = len(filtered_experiments)

    print(f"Initial number of rows: {initial_rows}")
    print(f"Number of rows after filtering: {filtered_rows}")

    return filtered_experiments, filtered_outcomes

def count_classifications(filtered_outcomes, classify):

    classifications = classify(filtered_outcomes)
    
    count_true = classifications.sum()  
    count_false = len(classifications) - count_true 

    print(f"Number of 'True' classifications: {count_true}")
    print(f"Number of 'False' classifications: {count_false}")

    return count_true, count_false

if __name__ == "__main__":

    experiments = pd.read_csv("experiments.csv")
    outcomes = pd.read_csv("outcomes.csv")

    # Convert boolean columns to 1/0 and One-hot encode the 'decision' feature. Ensure the new columns are in integer format (1/0 instead of True/False)
    bool_columns = experiments.select_dtypes(include=["bool"]).columns
    experiments[bool_columns] = experiments[bool_columns].astype(int)

    if 'decision' in experiments.columns:
        experiments = pd.get_dummies(experiments, columns=['decision'], drop_first=False)
        decision_columns = [col for col in experiments.columns if col.startswith('decision')] 
        experiments[decision_columns] = experiments[decision_columns].astype(int)

    print(experiments.head())
    print(outcomes.head())

    # Let's filter the data based on the decision of interest
    experiments, outcomes = filter_by_decision(experiments, outcomes, decision_values=["ref"]) # Specify what decision to mine
    results = (experiments, outcomes)

    regret_zero = (outcomes["regret_1"] < 0).sum()
    print(f" - Rows where regret_1 < 0: {regret_zero}")

    cart_alg = cart.setup_cart(results, classify, mass_min=0.05)
    cart_alg.build_tree()
    df = cart_alg.boxes_to_dataframe()
    cart_alg.show_tree()
    # # cart_alg.show_boxes(together=False)
    # for col in df.columns:
    #     print(f"\n===== Column: {col} =====")
    #     print(df[col])  # Print all values in the column
    # print("\nFirst 5 rows of the DataFrame:")
    # print(df.head())

    # Analyze a box by filtering the experiments
    # filtered_experiments, filtered_outcomes = filter_by_box(results, df, "box 12")
    print("\nThe below rows don't do much?")
    feature_limits = {
        "crc": (0, 203), 
        "Auction": 0,       
        # "Bioshortage": 0,
    }
    filtered_experiments, filtered_outcomes = filter_by_feature_limits(results, feature_limits)
    n_zeroregret, n_regret = count_classifications(filtered_outcomes, classify)
    print("density = ", round(n_zeroregret / (n_zeroregret + n_regret)*100), "%")

    plt.show()