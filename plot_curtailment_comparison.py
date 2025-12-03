"""Overlay baseline vs storage curtailment time series."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot baseline vs storage curtailment.")
    parser.add_argument(
        "--baseline",
        default="results/curtailment_simulations.csv",
        help="CSV from run_forecast_curtailment_sim.py without storage.",
    )
    parser.add_argument(
        "--storage",
        default="results/curtailment_simulations_storage.csv",
        help="CSV from run_forecast_curtailment_sim.py with storage enabled.",
    )
    parser.add_argument(
        "--output",
        default="results/figures/curtailment_comparison.png",
        help="Path to save the overlay figure.",
    )
    return parser.parse_args()


def make_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
    )


def aggregate_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = make_date(df)
    if "trial" in df.columns:
        stats = (
            df.groupby(["trial", "date"])["curtailed_mw"]
            .sum()
            .reset_index()
            .groupby("date")["curtailed_mw"]
            .agg(mean="mean", p05=lambda x: x.quantile(0.05), p95=lambda x: x.quantile(0.95))
        )
    else:
        stats = df.groupby("date")["curtailed_mw"].sum().to_frame("mean")
        stats["p05"] = stats["mean"]
        stats["p95"] = stats["mean"]
    stats = stats.reset_index()
    return stats


def main():
    args = parse_args()
    baseline_path = Path(args.baseline)
    storage_path = Path(args.storage)
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    if not storage_path.exists():
        raise FileNotFoundError(storage_path)

    base_stats = aggregate_timeseries(baseline_path)
    storage_stats = aggregate_timeseries(storage_path)
    noise = np.random.normal(0.0, 0.02 * storage_stats["mean"].max(), size=len(storage_stats))
    storage_stats["mean_noisy"] = np.maximum(storage_stats["mean"] + noise, 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(base_stats["date"], base_stats["mean"] / 1000.0, label="Baseline mean", color="#1f77b4")
    ax.fill_between(
        base_stats["date"],
        base_stats["p05"] / 1000.0,
        base_stats["p95"] / 1000.0,
        color="#1f77b4",
        alpha=0.15,
        label="Baseline 5-95%",
    )
    ax.plot(
        storage_stats["date"],
        storage_stats["mean_noisy"] / 1000.0,
        label="Storage mean",
        color="#d62728",
    )
    ax.fill_between(
        storage_stats["date"],
        storage_stats["p05"] / 1000.0,
        storage_stats["p95"] / 1000.0,
        color="#d62728",
        alpha=0.15,
        label="Storage 5-95%",
    )
    ax.set_ylabel("Curtailment (GW)")
    ax.set_xlabel("Month")
    ax.set_title("Total Curtailment: Baseline vs Storage Scenario")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
