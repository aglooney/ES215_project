#!/usr/bin/env python3
"""
Summarize sensitivity runs (line loading, slack, scenarios) into a compact table and plots.

Looks for files named:
  results/sensitivity/curtailment_sim_<scenario>_L<line>_slack<0|1>.csv

Outputs:
  - results/sensitivity/sensitivity_summary.csv
  - results/sensitivity/sensitivity_ens.png
  - results/sensitivity/sensitivity_curtail.png
"""
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

SENS_DIR = Path("results/sensitivity")
OUT_SUMMARY = SENS_DIR / "sensitivity_summary.csv"
OUT_ENS_PLOT = SENS_DIR / "sensitivity_ens.png"
OUT_CUR_PLOT = SENS_DIR / "sensitivity_curtail.png"
EPS = 1e-6


def parse_tag(path: Path):
    m = re.search(r"curtailment_sim_(.+)_L(\d+)_slack(\d+)", path.stem)
    if not m:
        return None
    scen, line, slack = m.group(1), int(m.group(2)), int(m.group(3))
    return scen, line, slack


def main():
    rows = []
    for csv in sorted(SENS_DIR.glob("curtailment_sim_*_L*_slack*.csv")):
        tag = parse_tag(csv)
        if tag is None:
            continue
        scenario, line_loading, slack = tag
        df = pd.read_csv(csv)
        # monthly means
        monthly = df.groupby(["year", "month"]).agg(
            ens_mw=("ens_mw", "mean"),
            curtail_mw=("curtailed_mw", "mean"),
        ).reset_index()
        ens_gw = monthly["ens_mw"].mean() / 1000.0
        curtail_gw = monthly["curtail_mw"].mean() / 1000.0
        ens_month_share = float((monthly["ens_mw"] > EPS).mean())
        curtail_month_share = float((monthly["curtail_mw"] > EPS).mean())
        rows.append(
            {
                "scenario": scenario,
                "line_loading": line_loading,
                "slack": slack,
                "ens_gw": ens_gw,
                "curtail_gw": curtail_gw,
                "ens_month_share": ens_month_share,
                "curtail_month_share": curtail_month_share,
            }
        )

    if not rows:
        raise RuntimeError(f"No sensitivity files found in {SENS_DIR}")

    summary = pd.DataFrame(rows).sort_values(["scenario", "slack", "line_loading"])
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved summary table to {OUT_SUMMARY}")
    print(summary.to_string(index=False))

    # Plots: one row per scenario, bars for line_loading with/without slack
    scenarios = summary["scenario"].unique().tolist()
    nrows = len(scenarios)
    slack_labels = {0: "no slack", 1: "slack"}

    def make_plot(metric: str, ylabel: str, out_path: Path):
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 3 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        for ax, scen in zip(axes, scenarios):
            sub = summary[summary["scenario"] == scen]
            for slack_val, group in sub.groupby("slack"):
                ax.plot(
                    group["line_loading"],
                    group[metric],
                    marker="o",
                    label=slack_labels.get(slack_val, f"slack={slack_val}"),
                )
            ax.set_title(f"{scen}")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Line loading percent")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    make_plot("ens_gw", "ENS (GW)", OUT_ENS_PLOT)
    make_plot("curtail_gw", "Curtailment (GW)", OUT_CUR_PLOT)


if __name__ == "__main__":
    main()
