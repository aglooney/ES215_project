import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def normalize_subsystem(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE", regex=False)
    )


def load_demand(path, start_month, end_month, include_paraguai, scaling):
    df = pd.read_csv(path)
    required = {"state", "subsystem", "year", "month", "MWh"}
    if not required.issubset(df.columns):
        raise ValueError(f"Demand CSV must contain: {required}")

    df["subsystem"] = normalize_subsystem(df["subsystem"])
    month_stamps = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    mask = (month_stamps >= start_month) & (month_stamps <= end_month)
    df = df.loc[mask]
    df["month_start"] = month_stamps.loc[mask]

    if not include_paraguai:
        df = df[df["subsystem"] != "PARAGUAI"]

    monthly = (
        df.groupby(["subsystem", "year", "month"], as_index=False)["MWh"].sum()
    )

    monthly["demand_avg_mw"] = monthly.apply(
        lambda r: r["MWh"]
        / (pd.Period(year=int(r["year"]), month=int(r["month"]), freq="M").days_in_month * 24),
        axis=1,
    )
    monthly["demand_scaled_mw"] = monthly["demand_avg_mw"] * scaling
    monthly["month_start"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )
    return monthly


def load_generation(path, start_month, end_month, include_paraguai):
    df = pd.read_csv(path, parse_dates=["date"])
    if "subsys_name" not in df.columns or "gen_val(MW)" not in df.columns:
        raise ValueError("Generation CSV must have 'subsys_name' and 'gen_val(MW)' columns.")

    df["subsystem"] = normalize_subsystem(df["subsys_name"])
    if not include_paraguai:
        df = df[df["subsystem"] != "PARAGUAI"]

    month_start = df["date"].values.astype("datetime64[M]")
    df = df.assign(month_start=pd.to_datetime(month_start))

    mask = (df["month_start"] >= start_month) & (df["month_start"] <= end_month)
    df = df.loc[mask]
    df["year"] = df["month_start"].dt.year
    df["month"] = df["month_start"].dt.month

    monthly = (
        df.groupby(["subsystem", "year", "month"], as_index=False)["gen_val(MW)"].sum()
    )
    monthly = monthly.rename(columns={"gen_val(MW)": "gen_mwh"})

    monthly["hours_in_month"] = monthly.apply(
        lambda r: pd.Period(year=int(r["year"]), month=int(r["month"]), freq="M").days_in_month * 24,
        axis=1,
    )
    monthly["gen_avg_mw"] = monthly["gen_mwh"] / monthly["hours_in_month"]
    monthly["month_start"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )
    return monthly


def make_summary(demand_df, gen_df):
    merged = pd.merge(
        demand_df,
        gen_df,
        on=["subsystem", "year", "month", "month_start"],
        how="outer",
        suffixes=("_demand", "_gen"),
    )
    fill_vals = {
        "MWh": 0.0,
        "demand_avg_mw": 0.0,
        "demand_scaled_mw": 0.0,
        "gen_mwh": 0.0,
        "hours_in_month": 0.0,
        "gen_avg_mw": 0.0,
    }
    merged = merged.fillna(fill_vals)
    merged["avg_gap_mw"] = merged["gen_avg_mw"] - merged["demand_avg_mw"]
    merged["scaled_gap_mw"] = merged["gen_avg_mw"] - merged["demand_scaled_mw"]
    merged = merged.sort_values(["month_start", "subsystem"]).reset_index(drop=True)

    total = (
        merged.groupby("month_start", as_index=False)[
            ["demand_avg_mw", "gen_avg_mw", "avg_gap_mw", "scaled_gap_mw"]
        ].sum()
    )
    total["subsystem"] = "TOTAL"

    summary = pd.concat([merged, total], ignore_index=True)
    summary = summary.sort_values(["month_start", "subsystem"]).reset_index(drop=True)

    summary["year_total"] = summary["month_start"].dt.year

    annual = (
        summary.groupby(
            ["year_total", "subsystem"],
            as_index=False,
        )[
            ["demand_avg_mw", "gen_avg_mw", "avg_gap_mw", "scaled_gap_mw"]
        ].mean()
    )
    annual = annual.rename(
        columns={
            "demand_avg_mw": "annual_demand_avg_mw",
            "gen_avg_mw": "annual_gen_avg_mw",
            "avg_gap_mw": "annual_avg_gap_mw",
            "scaled_gap_mw": "annual_scaled_gap_mw",
        }
    )
    return summary, annual


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize monthly generation vs demand gaps per subsystem."
    )
    parser.add_argument(
        "--demand-path",
        type=Path,
        default="data/demand_data/demand_projection_2025_2028.csv",
        help="Monthly demand CSV.",
    )
    parser.add_argument(
        "--gen-path",
        type=Path,
        default="results/generation_forecast_2025_2028.csv",
        help="Daily generation CSV.",
    )
    parser.add_argument(
        "--start",
        type=pd.Timestamp,
        required=True,
        help="Start month (YYYY-MM).",
    )
    parser.add_argument(
        "--end",
        type=pd.Timestamp,
        required=True,
        help="End month (YYYY-MM).",
    )
    parser.add_argument(
        "--demand-scaling",
        type=float,
        default=1.0,
        help="Multiplier to approximate peak demand.",
    )
    parser.add_argument(
        "--exclude-paraguai",
        action="store_true",
        help="If set, drop PARAGUAI subsystem rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        help="Optional path for a total-gap plot (PNG).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_month = pd.to_datetime(args.start).to_period("M").to_timestamp()
    end_month = pd.to_datetime(args.end).to_period("M").to_timestamp()
    if end_month < start_month:
        raise ValueError("end must be after or equal to start.")

    include_paraguai = not args.exclude_paraguai

    demand = load_demand(
        args.demand_path,
        start_month,
        end_month,
        include_paraguai=include_paraguai,
        scaling=args.demand_scaling,
    )
    generation = load_generation(
        args.gen_path,
        start_month,
        end_month,
        include_paraguai=include_paraguai,
    )
    summary, annual = make_summary(demand, generation)

    print("Monthly generation vs demand gaps (MW)")
    print(
        summary[
            [
                "month_start",
                "subsystem",
                "demand_avg_mw",
                "gen_avg_mw",
                "avg_gap_mw",
                "scaled_gap_mw",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:,.2f}")
    )

    print("\nAnnual average gaps (MW)")
    print(
        annual[
            [
                "year_total",
                "subsystem",
                "annual_demand_avg_mw",
                "annual_gen_avg_mw",
                "annual_avg_gap_mw",
                "annual_scaled_gap_mw",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:,.2f}")
    )

    if args.output:
        summary.to_csv(args.output, index=False)
        annual.to_csv(args.output.with_suffix(".annual.csv"), index=False)
        print(f"\nSaved monthly detail to {args.output}")
        print(f"Saved annual summary to {args.output.with_suffix('.annual.csv')}")

    if args.plot_path:
        total_gap = summary[summary["subsystem"] == "TOTAL"].sort_values("month_start")
        plt.figure(figsize=(10, 5))
        plt.plot(
            total_gap["month_start"],
            total_gap["avg_gap_mw"],
            marker="o",
            label="Gen - Demand gap",
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Month")
        plt.ylabel("Average gap (MW)")
        plt.title("Total monthly generation-demand gap")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.savefig(args.plot_path)
        print(f"Saved gap plot to {args.plot_path}")


if __name__ == "__main__":
    main()
