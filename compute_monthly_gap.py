
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_subsystem(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    )


def load_demand(demand_path):
    df = pd.read_csv(demand_path)
    required = {"state", "subsystem", "year", "month", "MWh"}
    if not required.issubset(df.columns):
        raise ValueError(f"Demand file missing columns: {required}")
    df["subsystem"] = normalize_subsystem(df["subsystem"])
    monthly = df.groupby(["year", "month", "subsystem"], as_index=False)["MWh"].sum()
    monthly["hours"] = monthly.apply(
        lambda r: pd.Period(year=int(r["year"]), month=int(r["month"]), freq="M").days_in_month * 24,
        axis=1,
    )
    monthly["demand_avg_mw"] = monthly["MWh"] / monthly["hours"]
    return monthly[["year", "month", "subsystem", "demand_avg_mw"]]


def load_generation(gen_path):
    df = pd.read_csv(gen_path)

    if "date" in df.columns and "gen_val(MW)" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        subsystem_col = "subsys_name"
        energy = pd.to_numeric(df["gen_val(MW)"], errors="coerce").fillna(0.0)
    elif "din_instante" in df.columns and "val_geracao" in df.columns:
        df["date"] = pd.to_datetime(df["din_instante"])
        subsystem_col = "nom_subsistema"
        energy = pd.to_numeric(df["val_geracao"], errors="coerce").fillna(0.0)
    else:
        raise ValueError(
            "Generation file must contain either ['date','gen_val(MW)'] "
            "or ['din_instante','val_geracao'] columns."
        )

    df["subsystem"] = normalize_subsystem(df[subsystem_col])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["energy_mwh"] = energy

    monthly = df.groupby(["year", "month", "subsystem"], as_index=False)["energy_mwh"].sum()
    monthly["hours"] = monthly.apply(
        lambda r: pd.Period(year=int(r["year"]), month=int(r["month"]), freq="M").days_in_month * 24,
        axis=1,
    )
    monthly["gen_avg_mw"] = monthly["energy_mwh"] / monthly["hours"]
    return monthly[["year", "month", "subsystem", "gen_avg_mw"]]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze monthly generation vs demand, export table/stats/plots."
    )
    parser.add_argument(
        "--demand-path",
        default="data/demand_data/demand_projection_clean.csv",
        type=Path,
        help="Cleaned demand CSV.",
    )
    parser.add_argument(
        "--generation-path",
        default="data/merged_generation_weather_v2.csv",
        type=Path,
        help="Merged generation-weather CSV.",
    )
    parser.add_argument(
        "--start-date",
        type=pd.Timestamp,
        default=pd.Timestamp("2020-01-01"),
        help="Minimum month (YYYY-MM) inclusive.",
    )
    parser.add_argument(
        "--end-date",
        type=pd.Timestamp,
        default=pd.Timestamp("2025-06-01"),
        help="Maximum month (YYYY-MM) inclusive.",
    )
    parser.add_argument(
        "--output",
        default="results/monthly_generation_gap.csv",
        type=Path,
        help="CSV path for the gap table.",
    )
    parser.add_argument(
        "--stats-path",
        default="results/gap_stats.csv",
        type=Path,
        help="Where to save summary statistics.",
    )
    parser.add_argument(
        "--plot-path",
        default="results/gap_distribution.png",
        type=Path,
        help="Path for the histogram plot.",
    )
    parser.add_argument(
        "--series-plot-path",
        default="results/time_series_plot.png",
        type=Path,
        help="Optional path for a time-series plot of total gaps.",
    )
    parser.add_argument(
        "--show-trend",
        action="store_true",
        default=True,
        help="Add a linear trendline with slope annotation to the time-series plot.",
    )
    args = parser.parse_args()

    demand = load_demand(args.demand_path)
    generation = load_generation(args.generation_path)

    merged = pd.merge(
        demand,
        generation,
        on=["year", "month", "subsystem"],
        how="outer",
        suffixes=("_demand", "_gen"),
    )
    merged["demand_avg_mw"] = merged["demand_avg_mw"].fillna(0.0)
    merged["gen_avg_mw"] = merged["gen_avg_mw"].fillna(0.0)
    merged["gap_mw"] = merged["gen_avg_mw"] - merged["demand_avg_mw"]
    merged = merged.sort_values(["year", "month", "subsystem"])

    start_year = args.start_date.year
    start_month = args.start_date.month
    end_year = args.end_date.year
    end_month = args.end_date.month

    mask = (
        (merged["year"] > start_year)
        | ((merged["year"] == start_year) & (merged["month"] >= start_month))
    ) & (
        (merged["year"] < end_year)
        | ((merged["year"] == end_year) & (merged["month"] <= end_month))
    )
    merged = merged[mask]

    merged.to_csv(args.output, index=False)
    print(f"Saved monthly gaps to {args.output}")

    totals = (
        merged.groupby(["year", "month"])["gap_mw"]
        .sum()
        .reset_index(name="total_gap_mw")
    )

    if args.stats_path:
        summary = totals["total_gap_mw"].agg(["mean", "std"]).to_frame().T
        summary.columns = ["normal_mean", "normal_std"]
        summary.to_csv(args.stats_path, index=False)
        print(f"Saved summary stats to {args.stats_path}")

    if args.plot_path:
        plt.figure(figsize=(10, 6))
        plt.hist(
            totals["total_gap_mw"],
            bins=20,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        plt.xlabel("Total generation-demand gap (MW)")
        plt.ylabel("Density")
        plt.title(
            f"Total grid gap distribution ({args.start_date.date()} to {args.end_date.date()})"
        )
        plt.tight_layout()
        plt.savefig(args.plot_path)
        print(f"Saved histogram to {args.plot_path}")

    if args.series_plot_path:
        plt.figure(figsize=(10, 5))
        dates = pd.to_datetime(
            totals["year"].astype(int).astype(str)
            + "-"
            + totals["month"].astype(int).astype(str)
            + "-01"
        )
        plt.plot(
            dates,
            totals["total_gap_mw"],
            marker="o",
            label="Monthly gap",
        )

        if args.show_trend and len(totals) > 1:
            x = (dates - dates.min()).dt.days.values
            y = totals["total_gap_mw"].values
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope * x + intercept
            plt.plot(dates, trend, linestyle="--", color="red", label="Trend")
            slope_per_month = slope * 30.4375  # approx days per month
            y_offset = (trend.max() - trend.min()) * 0.1 if trend.max() != trend.min() else 1000
            plt.text(
                dates.iloc[len(dates) // 2],
                trend.max() + y_offset,
                f"Slope: {slope_per_month:.1f} MW/mo",
                color="darkgreen",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="darkgreen", alpha=0.8),
            )

        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Month")
        plt.ylabel("Total gap (MW)")
        plt.title(
            f"Total monthly generation-demand gap ({args.start_date.date()} to {args.end_date.date()})"
        )
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.series_plot_path)
        print(f"Saved time-series plot to {args.series_plot_path}")


if __name__ == "__main__":
    main()
