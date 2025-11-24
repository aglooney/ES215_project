import argparse
import calendar
from pathlib import Path

import pandas as pd


def normalize_subsystem(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE", regex=False)
    )


def load_demand(path, start_date, end_date, scaling):
    df = pd.read_csv(path)
    required = {"state", "subsystem", "year", "month", "MWh"}
    if not required.issubset(df.columns):
        raise ValueError(f"Demand file must contain columns: {required}")

    df["subsystem"] = normalize_subsystem(df["subsystem"])
    df["month_start"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1)
    )

    start_month = start_date.to_period("M").to_timestamp()
    end_month = end_date.to_period("M").to_timestamp()
    mask = (df["month_start"] >= start_month) & (df["month_start"] <= end_month)
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(
            f"No demand rows between {start_date.date()} and {end_date.date()}."
        )

    monthly = df.groupby(["subsystem", "year", "month"], as_index=False)["MWh"].sum()
    monthly["hours_in_month"] = monthly.apply(
        lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1] * 24,
        axis=1,
    )

    agg = monthly.groupby("subsystem", as_index=False).agg(
        {"MWh": "sum", "hours_in_month": "sum"}
    )
    agg["demand_avg_mw"] = agg["MWh"] / agg["hours_in_month"]
    agg["demand_peak_mw"] = agg["demand_avg_mw"] * scaling
    return agg


def load_generation(path, start_date, end_date):
    df = pd.read_csv(path, parse_dates=["date"])
    if "subsys_name" not in df.columns or "gen_val(MW)" not in df.columns:
        raise ValueError("Generation CSV must contain 'subsys_name' and 'gen_val(MW)'.")

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(
            f"No generation rows between {start_date.date()} and {end_date.date()}."
        )

    df["subsystem"] = normalize_subsystem(df["subsys_name"])

    energy = df.groupby("subsystem", as_index=False)["gen_val(MW)"].sum()
    energy = energy.rename(columns={"gen_val(MW)": "gen_mwh"})

    day_counts = (
        df.groupby("subsystem")["date"].nunique().reset_index(name="days_count")
    )

    agg = energy.merge(day_counts, on="subsystem", how="left")
    agg["hours_in_window"] = agg["days_count"] * 24
    agg["gen_avg_mw"] = agg.apply(
        lambda r: r["gen_mwh"] / r["hours_in_window"] if r["hours_in_window"] > 0 else 0,
        axis=1,
    )
    return agg


def compare_levels(demand_df, gen_df):
    demand_df = demand_df[demand_df["subsystem"] != "PARAGUAI"]
    gen_df = gen_df[gen_df["subsystem"] != "PARAGUAI"]
    combo = pd.merge(demand_df, gen_df, on="subsystem", how="outer")
    combo = combo.fillna(
        {
            "MWh": 0.0,
            "hours_in_month": 0.0,
            "demand_avg_mw": 0.0,
            "demand_peak_mw": 0.0,
            "gen_mwh": 0.0,
            "days_count": 0.0,
            "hours_in_window": 0.0,
            "gen_avg_mw": 0.0,
        }
    )
    combo["surplus_mw"] = combo["gen_avg_mw"] - combo["demand_peak_mw"]
    return combo.sort_values("subsystem").reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare subsystem-level generation from merged_generation_weather.csv with monthly demand."
    )
    parser.add_argument(
        "--demand-path",
        default=Path("data/demand_data/demand_projection_clean.csv"),
        type=Path,
        help="CSV with columns state, subsystem, year, month, MWh.",
    )
    parser.add_argument(
        "--gen-path",
        default=Path("data/merged_generation_weather.csv"),
        type=Path,
        help="Merged generation CSV containing subsys_name and gen_val(MW).",
    )
    parser.add_argument(
        "--start-date",
        type=pd.Timestamp,
        required=True,
        help="Start date (YYYY-MM-DD) for the comparison window.",
    )
    parser.add_argument(
        "--end-date",
        type=pd.Timestamp,
        required=True,
        help="End date (YYYY-MM-DD) for the comparison window.",
    )
    parser.add_argument(
        "--demand-scaling",
        type=float,
        default=1.3,
        help="Multiplier applied to average hourly demand to approximate peak.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to store the comparison table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.end_date < args.start_date:
        raise ValueError("end-date must be on or after start-date.")

    demand = load_demand(args.demand_path, args.start_date, args.end_date, args.demand_scaling)
    generation = load_generation(args.gen_path, args.start_date, args.end_date)
    comparison = compare_levels(demand, generation)

    print("\nSubsystem generation vs demand")
    print(f"Date range: {args.start_date.date()} → {args.end_date.date()}")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    if args.output:
        comparison.to_csv(args.output, index=False)
        print(f"\nSaved comparison to {args.output}")


if __name__ == "__main__":
    main()
