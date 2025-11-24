import argparse
import subprocess
from pathlib import Path

import pandas as pd


def run_scenario(python_bin, seed, std, base_args, results_dir):
    output_path = results_dir / f"curtailment_seed_{seed}.csv"
    cmd = [
        python_bin,
        "run_curtailment_opf.py",
        "--gen-scale-std",
        str(std),
        "--seed",
        str(seed),
        "--results-path",
        str(output_path),
    ]
    cmd.extend(base_args)
    subprocess.run(cmd, check=True)
    df = pd.read_csv(output_path)
    df["seed"] = seed
    return df


def main():
    parser = argparse.ArgumentParser(description="Batch uncertainty runs for DC-OPF.")
    parser.add_argument("--python", default="python", help="Python executable to use.")
    parser.add_argument("--runs", type=int, default=10, help="Number of scenarios.")
    parser.add_argument("--std", type=float, default=0.1, help="Std dev for generation scaling.")
    parser.add_argument("--start-seed", type=int, default=0, help="Seed offset.")
    parser.add_argument("--results-dir", default="results/uncertainty", help="Directory for per-run CSVs.")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional args passed to run_curtailment_opf.py (e.g., --year 2021 --month 7).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    aggregated = []
    base_args = args.extra_args or []
    for i in range(args.runs):
        seed = args.start_seed + i
        df = run_scenario(args.python, seed, args.std, base_args, results_dir)
        aggregated.append(df)

    combined = pd.concat(aggregated, ignore_index=True)
    combined.to_csv(results_dir / "curtailment_uncertainty.csv", index=False)
    print(f"Saved individual runs to {results_dir}")
    print(f"Aggregated results: {results_dir / 'curtailment_uncertainty.csv'}")


if __name__ == "__main__":
    main()
