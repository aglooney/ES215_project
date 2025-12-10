"""Validate DC-OPF curtailment estimates against historical generation-demand gaps."""
import argparse
import calendar
import subprocess
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="Validate OPF curtailment vs historical gaps.")
parser.add_argument("--year", type=int, default=2023, help="Year to validate (single year).")
parser.add_argument("--demand-path", default="data/demand_data/demand_projection_clean.csv")
parser.add_argument("--generation-path", default="data/merged_generation_weather_v2.csv")
parser.add_argument("--results-path", default="results/validation_opf_vs_actual.csv")
parser.add_argument("--opf-script", default="run_curtailment_opf.py")
parser.add_argument("--python-bin", default="./es215_env/bin/python")
parser.add_argument("--min-gen-frac", type=float, default=0.1)
parser.add_argument("--line-loading-percent", type=float, default=100.0)
args = parser.parse_args()


def hours_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1] * 24


def compute_actual_gap(demand_df, gen_df, year: int, month: int) -> float:
    d = demand_df[(demand_df["year"] == year) & (demand_df["month"] == month)]
    g = gen_df[(gen_df["date"].dt.year == year) & (gen_df["date"].dt.month == month)]
    if d.empty or g.empty:
        return float("nan")
    hours = hours_in_month(year, month)
    demand_mw = d.groupby("subsystem")["MWh"].sum().sum() / hours
    gen_mw = g["gen_val(MW)"].sum() / hours
    return gen_mw - demand_mw


demand_df = pd.read_csv(args.demand_path)
demand_df["subsystem"] = demand_df["subsystem"].str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
gen_df = pd.read_csv(args.generation_path, parse_dates=["date"])

year = args.year
records = []
tmp_output = Path("results/_validation_tmp.csv")
for month in range(1, 13):
    subprocess.run(
        [
            args.python_bin,
            args.opf_script,
            "--year",
            str(year),
            "--month",
            str(month),
            "--demand-scaling",
            "1.0",
            "--results-path",
            str(tmp_output),
            "--min-gen-frac",
            str(args.min_gen_frac),
            "--line-loading-percent",
            str(args.line_loading_percent),
        ],
        check=True,
    )
    opf_df = pd.read_csv(tmp_output)
    curtailment_mw = opf_df["curtailed_mw"].sum()
    actual_gap = compute_actual_gap(demand_df, gen_df, year, month)
    records.append(
        {
            "year": year,
            "month": month,
            "opf_curtailment_mw": curtailment_mw,
            "actual_gap_mw": actual_gap,
        }
    )

result_df = pd.DataFrame(records)
result_df["difference_mw"] = result_df["opf_curtailment_mw"] - result_df["actual_gap_mw"].clip(lower=0)
mae = result_df["difference_mw"].abs().mean()
rmse = (result_df["difference_mw"] ** 2).mean() ** 0.5
result_df.to_csv(args.results_path, index=False)
print(f"Saved validation comparison to {args.results_path}")
print(f"MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW")
