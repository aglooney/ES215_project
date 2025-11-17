"""
Generate subsystem-level generation scenarios for OPF.

This script:
1. Loads your merged generator dataset (should include: ceg, subsystem, gen_val(MW), ...)
2. Aggregates generation by subsystem for a selected date or average period.
3. Saves a scenario file compatible with the curtailment OPF script.

You can modify SCENARIO_DATE or use full-day averages.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Your merged dataset (the one used in your ML model)
DATA_PATH = "data/merged_generation_weather.csv"

# Where to save scenario
OUT_DIR = Path("models/scenarios")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Which date to use for scenario creation (YYYY-MM-DD)
# You can change this to any date in your dataset.
SCENARIO_DATE = "2020-12-10"

# Column names
DATE_COL = "date"
PLANT_ID_COL = "ceg"
SUBSYS_COL = "subsys_name"      # adjust if your dataset uses a different name
GEN_COL_ACTUAL = "gen_val(MW)"
GEN_COL_ML = "gen_ml_pred"      # OPTIONAL: only if you generated ML predictions

# ============================================================
# LOAD & PREPARE DATA
# ============================================================

print("📘 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Ensure date conversion
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# Check subsystem column exists
if SUBSYS_COL not in df.columns:
    raise ValueError(f"Subsystem column '{SUBSYS_COL}' not found. Available columns:\n{df.columns}")

# Filter to the scenario date
df_day = df[df[DATE_COL] == SCENARIO_DATE].copy()
if df_day.empty:
    raise ValueError(f"No rows found for date {SCENARIO_DATE} in dataset.")

print(f"  Rows for scenario date {SCENARIO_DATE}: {len(df_day)}")

# ============================================================
# AGGREGATE GENERATION BY SUBSYSTEM
# ============================================================

print("\n⚡ Aggregating generation...")

agg_dict = {
    GEN_COL_ACTUAL: "sum"
}

if GEN_COL_ML in df_day.columns:
    agg_dict[GEN_COL_ML] = "sum"

df_subsys = (df_day.groupby(SUBSYS_COL).agg(agg_dict).reset_index().rename(columns={
    SUBSYS_COL: "subsystem",
    GEN_COL_ACTUAL: "gen_total_mw",
    GEN_COL_ML: "gen_ml_mw"
})
)

print(df_subsys)

# ============================================================
# SAVE SCENARIO
# ============================================================

scenario_name = f"gen_subsystem_{SCENARIO_DATE}.parquet"
out_path = OUT_DIR / scenario_name

df_subsys.to_parquet(out_path, index=False)

print("\n💾 Saved subsystem scenario file:")
print(out_path)
