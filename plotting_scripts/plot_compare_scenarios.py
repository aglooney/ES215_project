#!/usr/bin/env python3
"""
Plot system-level curtailment means across multiple demand scenarios on one figure.

Looks for summary CSVs produced by summarize_curtailment_sim.py:
  results/<scenario>_summary/curtailment_simulations_<scenario>_monthly_detailed.csv

Default scenarios: inferior, referencia, superior.
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_monthly(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path, parse_dates=["date"])
    return df


def main():
    ap = argparse.ArgumentParser(description="Compare curtailment means across scenarios.")
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=["inferior", "referencia", "superior"],
        help="Scenario names corresponding to summary folders/files.",
    )
    ap.add_argument(
        "--base-dir",
        default="results",
        help="Base results directory containing <scenario>_summary folders.",
    )
    ap.add_argument(
        "--output",
        default="results/curtailment_compare_scenarios.png",
        help="Output plot path.",
    )
    args = ap.parse_args()

    plt.figure(figsize=(11, 5))
    for scen in args.scenarios:
        summary_file = Path(args.base_dir) / f"{scen}_summary" / f"curtailment_simulations_{scen}_monthly_detailed.csv"
        df = load_monthly(summary_file)
        # system-level curtailment in MW → GW
        series = df.groupby("date")["total_curtailment_mw"].mean() / 1000.0
        plt.plot(series.index, series.values, label=f"{scen} (mean curtailment)")

    plt.ylabel("Curtailment (GW)")
    plt.xlabel("Month")
    plt.title("System Curtailment (mean) by Scenario")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
