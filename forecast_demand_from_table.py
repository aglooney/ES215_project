"""Generate monthly subsystem demand projections for 2025-2028 using table inputs."""
import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Create seasonal demand forecasts using table scenarios.")
parser.add_argument("--historical-path", default="data/demand_data/demand_projection_clean.csv")
parser.add_argument("--scenario", choices=["inferior","referencia","superior"], default="referencia")
parser.add_argument("--hist-start-year", type=int, default=2020)
parser.add_argument("--hist-end-year", type=int, default=2024)
parser.add_argument("--start-year", type=int, default=2025)
parser.add_argument("--end-year", type=int, default=2028)
parser.add_argument("--output", default="data/demand_data/demand_projection_2025_2028.csv")
args = parser.parse_args()

START_YEAR = args.start_year
END_YEAR = args.end_year
OUTPUT = Path(args.output)

scenario_data = {
    "inferior": {
        "NORTE": {"value_2025": 8.2, "growth": 0.029},
        "NORDESTE": {"value_2025": 13.5, "growth": 0.029},
        "SUDESTE": {"value_2025": 46.1, "growth": 0.026},
        "SUL": {"value_2025": 14.1, "growth": 0.030},
    },
    "referencia": {
        "NORTE": {"value_2025": 8.3, "growth": 0.033},
        "NORDESTE": {"value_2025": 13.7, "growth": 0.037},
        "SUDESTE": {"value_2025": 46.4, "growth": 0.031},
        "SUL": {"value_2025": 14.2, "growth": 0.036},
    },
    "superior": {
        "NORTE": {"value_2025": 8.3, "growth": 0.041},
        "NORDESTE": {"value_2025": 13.7, "growth": 0.077},
        "SUDESTE": {"value_2025": 46.9, "growth": 0.048},
        "SUL": {"value_2025": 14.3, "growth": 0.043},
    },
}


def compute_seasonal_factors(path: str, start_year: int, end_year: int) -> dict[str, dict[int, float]]:
    """Return seasonal multipliers (mean=1) for each subsystem."""
    df = pd.read_csv(path)
    df["subsystem"] = df["subsystem"].str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    if df.empty:
        raise ValueError("Historical demand slice is empty.")

    def month_hours(row):
        return calendar.monthrange(int(row["year"]), int(row["month"]))[1] * 24

    agg = df.groupby(["subsystem", "year", "month"])["MWh"].sum().reset_index()
    agg["hours"] = agg.apply(month_hours, axis=1)
    agg["avg_mw"] = agg["MWh"] / agg["hours"]
    agg["annual_avg"] = agg.groupby(["subsystem", "year"])["avg_mw"].transform("mean")
    agg["ratio"] = agg["avg_mw"] / agg["annual_avg"]
    seasonal = (
        agg.groupby(["subsystem", "month"])["ratio"]
        .mean()
        .reset_index()
    )

    seasonal_dict: dict[str, dict[int, float]] = {}
    for (subsys, sub_df) in seasonal.groupby("subsystem"):
        factors = sub_df.set_index("month")["ratio"].to_dict()
        mean_factor = np.mean(list(factors.values()))
        if mean_factor == 0:
            mean_factor = 1.0
        factors = {m: val / mean_factor for m, val in factors.items()}
        seasonal_dict[subsys] = factors
    return seasonal_dict


seasonal_factors = compute_seasonal_factors(
    args.historical_path, args.hist_start_year, args.hist_end_year
)

rows = []
scenario = args.scenario
subsystems = scenario_data[scenario]
for subsystem, info in subsystems.items():
    base_mw = info["value_2025"] * 1000.0
    growth = info["growth"]
    factors = seasonal_factors.get(subsystem, {m: 1.0 for m in range(1, 13)})
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            months_since_start = (year - START_YEAR) * 12 + (month - 1)
            trend = (1.0 + growth) ** (months_since_start / 12.0)
            seasonal = factors.get(month, 1.0)
            demand = base_mw * trend * seasonal
            rows.append({
                "scenario": scenario,
                "subsystem": subsystem,
                "year": year,
                "month": month,
                "demand_mw": demand,
            })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(f"Saved demand projections to {OUTPUT}")
