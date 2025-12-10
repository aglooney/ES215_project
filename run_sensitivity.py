#!/usr/bin/env python3
"""
Run a small sensitivity sweep on curtailment simulations.

Sweeps over:
  - line_loading_percent values
  - allow_slack_imports on/off

Uses a single demand/gen input (defaults to referencia forecast + relaxed net).
Writes results and summaries for each combination.
"""
import itertools
import subprocess
from pathlib import Path

# Base inputs
NET_JSON = "models/pandapower_snapshots/brazil_network_zonal_5bus_relaxed.json"
GEN = "results/generation_forecast_2025_2028.csv"
DEMAND_CLEAN = "data/demand_data/demand_projection_clean.csv"
SCENARIOS = ["inferior", "referencia", "superior"]

# Sweep values
LINE_LOADINGS = [100.0, 150.0, 200.0]
ALLOW_SLACK_OPTS = [0, 1]

# Trials/seed settings
N_TRIALS = 100
SEED = 0
HIST_START = 2020
HIST_END = 2024
START_YEAR = 2025
END_YEAR = 2028


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    out_base = Path("results/sensitivity")
    out_base.mkdir(parents=True, exist_ok=True)

    for scenario, line_loading, allow_slack in itertools.product(
        SCENARIOS, LINE_LOADINGS, ALLOW_SLACK_OPTS
    ):
        demand_path = Path(f"data/demand_data/demand_projection_{scenario}_2025_2028.csv")
        if not demand_path.exists():
            run(
                [
                    "python3",
                    "forecast_demand_from_table.py",
                    "--historical-path",
                    DEMAND_CLEAN,
                    "--hist-start-year",
                    str(HIST_START),
                    "--hist-end-year",
                    str(HIST_END),
                    "--start-year",
                    str(START_YEAR),
                    "--end-year",
                    str(END_YEAR),
                    "--scenario",
                    scenario,
                    "--output",
                    str(demand_path),
                ]
            )

        tag = f"{scenario}_L{int(line_loading)}_slack{allow_slack}"
        sim_out = out_base / f"curtailment_sim_{tag}.csv"
        sim_fail = out_base / f"curtailment_sim_{tag}_opf_failures.csv"
        summary_dir = out_base / f"{tag}_summary"

        slack_flag = ["--allow-slack-imports"] if allow_slack else []

        # Simulation
        cmd_sim = [
            "python3",
            "run_forecast_curtailment_sim.py",
            "--net-json",
            NET_JSON,
            "--demand",
            str(demand_path),
            "--gen",
            GEN,
            "--n-trials",
            str(N_TRIALS),
            "--seed",
            str(SEED),
            "--line-loading-percent",
            str(line_loading),
            "--demand-scenario",
            scenario,
            "--out",
            str(sim_out),
            "--opf-failures-out",
            str(sim_fail),
        ] + slack_flag
        run(cmd_sim)

        # Summary
        cmd_sum = [
            "python3",
            "summarize_curtailment_sim.py",
            "--input",
            str(sim_out),
            "--out-dir",
            str(summary_dir),
            "--fig-dir",
            str(summary_dir / "figures"),
        ]
        run(cmd_sum)

    print("Sensitivity sweep done.")


if __name__ == "__main__":
    main()
