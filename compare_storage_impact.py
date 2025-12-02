"""Quantify curtailment reduction from a storage scenario."""
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline vs storage curtailment results.")
    parser.add_argument(
        "--baseline",
        default="results/curtailment_simulations.csv",
        help="Baseline CSV from run_forecast_curtailment_sim.py.",
    )
    parser.add_argument(
        "--storage",
        default="results/curtailment_simulations_storage.csv",
        help="Storage CSV from run_forecast_curtailment_sim.py with --storage-* flags.",
    )
    return parser.parse_args()


def load_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "curtailed_mw" not in df.columns:
        raise ValueError(f"'curtailed_mw' missing in {path}")
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01")
    if "trial" in df.columns:
        grouped = (
            df.groupby(["trial", "date"])["curtailed_mw"]
            .sum()
            .reset_index()
            .groupby("date")["curtailed_mw"]
            .mean()
        )
    else:
        grouped = df.groupby("date")["curtailed_mw"].sum()
    return grouped.reset_index().rename(columns={"curtailed_mw": "curtailment_mw"})


def main():
    args = parse_args()
    base_path = Path(args.baseline)
    storage_path = Path(args.storage)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not storage_path.exists():
        raise FileNotFoundError(storage_path)

    baseline = load_stats(base_path)
    storage = load_stats(storage_path)
    merged = baseline.merge(storage, on="date", how="inner", suffixes=("_baseline", "_storage"))
    if merged.empty:
        raise RuntimeError("No overlapping dates between baseline and storage results.")
    merged["reduction_mw"] = merged["curtailment_mw_baseline"] - merged["curtailment_mw_storage"]
    merged["reduction_pct"] = merged["reduction_mw"] / merged["curtailment_mw_baseline"].replace(0, pd.NA) * 100

    overall_baseline = merged["curtailment_mw_baseline"].sum()
    overall_storage = merged["curtailment_mw_storage"].sum()
    total_hours = len(merged)
    print("Curtailment comparison (mean over trials):")
    print(merged[["date", "curtailment_mw_baseline", "curtailment_mw_storage", "reduction_mw", "reduction_pct"]])
    print("\nSummary across all months:")
    print(f"  Baseline total curtailed energy (MW-month sum): {overall_baseline:,.2f}")
    print(f"  Storage total curtailed energy (MW-month sum): {overall_storage:,.2f}")
    print(f"  Absolute reduction: {overall_baseline - overall_storage:,.2f} MW-month")
    pct = (overall_baseline - overall_storage) / overall_baseline * 100 if overall_baseline else float("nan")
    print(f"  Percent reduction: {pct:.2f}%")
    if total_hours:
        avg_hourly = (overall_baseline - overall_storage) / total_hours
        print(f"  Average monthly reduction: {avg_hourly:.2f} MW")


if __name__ == "__main__":
    main()
