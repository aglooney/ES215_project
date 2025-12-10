#!/usr/bin/env python3
"""
Plot historic quarterly curtailment (bar) and future projected curtailment (box) by quarter.

Inputs:
  --historic: CSV with at least year, month, total_curtailment_mw (e.g., historic baseline monthly results).
  --futures: One or more CSVs in the monthly_detailed format produced by summarize_curtailment_sim.py
             (contains year, month, total_curtailment_mw for each trial-month).

Output:
  A single PNG showing historic quarterly mean (bar) and future quarterly distributions (boxplots).
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def quarter_label(year: int, month: int) -> str:
    q = (int(month) - 1) // 3 + 1
    return f"{int(year)}-Q{q}"


def load_quarterly_means(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if not {"year", "month", "total_curtailment_mw"}.issubset(df.columns):
        raise ValueError(f"{path} missing required columns year, month, total_curtailment_mw")
    df["quarter"] = df.apply(lambda r: quarter_label(r["year"], r["month"]), axis=1)
    return df.groupby("quarter")["total_curtailment_mw"].mean() / 1000.0  # GW


def load_future_quarterly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"year", "month", "total_curtailment_mw"}.issubset(df.columns):
        raise ValueError(f"{path} missing required columns year, month, total_curtailment_mw")
    df["quarter"] = df.apply(lambda r: quarter_label(r["year"], r["month"]), axis=1)
    df["scenario"] = path.stem  # label by file stem
    df["curtailment_gw"] = df["total_curtailment_mw"] / 1000.0
    return df[["quarter", "curtailment_gw", "scenario"]]


def main():
    ap = argparse.ArgumentParser(description="Quarterly curtailment: historic vs future projections.")
    ap.add_argument("--historic", required=True, help="CSV with historic curtailment (year, month, total_curtailment_mw).")
    ap.add_argument(
        "--futures",
        nargs="+",
        default=["results/referencia_summary/curtailment_simulations_referencia_monthly_detailed.csv"],
        help="CSV files with future monthly_detailed curtailment.",
    )
    ap.add_argument("--output", default="results/curtailment_quarterly.png", help="Output plot path.")
    args = ap.parse_args()

    hist_path = Path(args.historic)
    future_paths = [Path(p) for p in args.futures]

    hist_q = load_quarterly_means(hist_path)

    future_df = pd.concat([load_future_quarterly(p) for p in future_paths], ignore_index=True)

    # Align quarters
    quarters = sorted(set(hist_q.index) | set(future_df["quarter"].unique()))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Historic bars
    hist_vals = [hist_q.get(q, float("nan")) for q in quarters]
    ax.bar(range(len(quarters)), hist_vals, width=0.4, align="edge", label="Historic mean (GW)", color="#7fa2d6")

    # Future boxplots per quarter (across all scenarios/trials)
    box_data = [future_df[future_df["quarter"] == q]["curtailment_gw"].values for q in quarters]
    ax.boxplot(
        box_data,
        positions=[i + 0.4 for i in range(len(quarters))],
        widths=0.4,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#f6c971", color="#b67a0a"),
        medianprops=dict(color="#b67a0a"),
    )

    ax.set_xticks([i + 0.4 for i in range(len(quarters))])
    ax.set_xticklabels(quarters, rotation=45)
    ax.set_ylabel("Curtailment (GW)")
    ax.set_title("Quarterly Curtailment: Historic (bar) vs Future Projections (box)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
