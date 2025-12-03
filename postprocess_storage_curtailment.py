"""Post-process baseline curtailment results to emulate storage absorption."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate storage benefit by clipping curtailment ex post.")
    parser.add_argument(
        "--input",
        default="results/curtailment_simulations.csv",
        help="Baseline CSV from run_forecast_curtailment_sim.py.",
    )
    parser.add_argument(
        "--storage-source",
        required=True,
        help="Subsystem whose curtailment can be absorbed by storage (e.g., NORTE).",
    )
    parser.add_argument(
        "--storage-capacity",
        type=float,
        required=True,
        help="MW of curtailment the storage can absorb each month (per trial).",
    )
    parser.add_argument(
        "--storage-variation",
        type=float,
        default=0.2,
        help="Std dev (as fraction of capacity) for month-to-month storage availability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed applied when sampling dynamic storage availability.",
    )
    parser.add_argument(
        "--output-adjusted",
        default=None,
        help="Optional path to save a copy of the dataset with adjusted curtailment.",
        )
    return parser.parse_args()


def aggregate(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = df.groupby(["year", "month", "trial"])[column].sum().reset_index() if "trial" in df.columns else df
    if "trial" in grouped.columns:
        grouped["date"] = pd.to_datetime(
            grouped["year"].astype(int).astype(str)
            + "-"
            + grouped["month"].astype(int).astype(str)
            + "-01"
        )
        stats = grouped.groupby("date")[column].mean().reset_index()
    else:
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
        )
        stats = df.groupby("date")[column].sum().reset_index()
    stats = stats.rename(columns={column: "curtailment_mw"})
    return stats


def main():
    args = parse_args()
    source = args.storage_source.upper().strip()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"subsystem", "curtailed_mw", "year", "month"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Input missing columns: {missing}")

    df["subsystem"] = df["subsystem"].astype(str).str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    df["captured_mw"] = 0.0
    mask = df["subsystem"] == source
    rng = np.random.default_rng(args.seed)
    dynamic_caps = np.maximum(
        rng.normal(args.storage_capacity, args.storage_capacity * args.storage_variation, mask.sum()),
        0.0,
    )
    df.loc[mask, "captured_mw"] = np.minimum(df.loc[mask, "curtailed_mw"].values, dynamic_caps)
    df["curtailed_adjusted"] = df["curtailed_mw"] - df["captured_mw"]

    total_captured = df["captured_mw"].sum()
    baseline_stats = aggregate(df.copy(), "curtailed_mw")
    adjusted_stats = aggregate(df.assign(curtailed_mw=df["curtailed_adjusted"]).copy(), "curtailed_mw")

    merged = baseline_stats.merge(adjusted_stats, on="date", suffixes=("_baseline", "_storage"))
    merged["reduction_mw"] = merged["curtailment_mw_baseline"] - merged["curtailment_mw_storage"]
    merged["reduction_pct"] = (
        merged["reduction_mw"] / merged["curtailment_mw_baseline"].replace(0, pd.NA) * 100
    )

    total_baseline = merged["curtailment_mw_baseline"].sum()
    total_adjusted = merged["curtailment_mw_storage"].sum()
    pct = (total_baseline - total_adjusted) / total_baseline * 100 if total_baseline else float("nan")

    print("Monthly reduction (mean over trials if applicable):")
    print(merged[["date", "curtailment_mw_baseline", "curtailment_mw_storage", "reduction_mw", "reduction_pct"]])
    print("\nSummary:")
    print(f"  Total curtailment (baseline, MW-month sum): {total_baseline:,.2f}")
    print(f"  Total curtailment (storage, MW-month sum): {total_adjusted:,.2f}")
    print(f"  Absolute reduction: {total_baseline - total_adjusted:,.2f} MW-month")
    print(f"  Percent reduction: {pct:.2f}%")
    print(f"  Storage absorbed (sum over all trials/months): {total_captured:,.2f} MW-month")

    if args.output_adjusted:
        out_path = Path(args.output_adjusted)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = df.copy()
        out_df["curtailed_mw"] = out_df["curtailed_adjusted"]
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved adjusted dataset to {out_path}")


if __name__ == "__main__":
    main()
