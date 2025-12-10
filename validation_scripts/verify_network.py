"""
verify_network.py

Quick checks on simulation output schema + a few sanity diagnostics.
"""

import pandas as pd

PATH = "results/curtailment_simulations.csv"
df = pd.read_csv(PATH)

# Use the correct column names:
df["surplus_mw"] = df["total_available_gen_mw"] - df["total_demand_mw"]

print(df["surplus_mw"].describe())
print("Share deficit months:", (df["surplus_mw"] < 0).mean())
print("Share surplus months:", (df["surplus_mw"] > 0).mean())
