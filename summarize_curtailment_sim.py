"""
summarize_curtailment_detailed.py

Detailed summaries + plots for curtailment + ENS simulation outputs.

Input:
  results/curtailment_simulations.csv (row-level, one row per subsystem per month per trial)

All power quantities in the input are assumed to be in MW.
Plots convert to GW by dividing by 1000.

Outputs:
  *_monthly_detailed.csv
  *_subsystem_monthly.csv
  *_subsystem_stats.csv
  *_regime_counts.csv
  *_binding_line_impacts.csv
  *_top_events_ens.csv, *_top_events_curtailment.csv, *_top_events_both.csv
  Figures in --fig-dir:
    - system_timeseries_ens_curtail_stranded.png
    - system_scatter_ens_vs_surplus.png
    - heatmap_mean_curtailment.png
    - boxplot_curtailment_by_subsystem.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_date(year, month):
    return pd.to_datetime(
        year.astype(int).astype(str) + "-" + month.astype(int).astype(str) + "-01"
    )


def first_nonempty_string(series):
    for x in series:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return ""


def explode_binding_lines(df_monthly, col="binding_line_names"):
    if col not in df_monthly.columns:
        return pd.DataFrame(columns=["trial", "date", "binding_line"])
    tmp = df_monthly[["trial", "date", col]].copy()
    tmp[col] = tmp[col].fillna("").astype(str)
    tmp["binding_line"] = tmp[col].str.split(";")
    tmp = tmp.drop(columns=[col]).explode("binding_line")
    tmp["binding_line"] = tmp["binding_line"].astype(str).str.strip()
    tmp = tmp[tmp["binding_line"] != ""].copy()
    return tmp


def q(x, p):
    return x.quantile(p)


def main():
    parser = argparse.ArgumentParser(
        description="Create detailed summaries from curtailment simulation outputs."
    )
    parser.add_argument("--input", default="results/curtailment_simulations.csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument(
        "--include-opf-failures", action="store_true",
        help="If set, keep OPF failures instead of dropping them."
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path(args.fig_dir) if args.fig_dir else (out_dir / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Keep only converged OPFs unless explicitly told otherwise
    if "opf_converged" in df.columns and not args.include_opf_failures:
        df = df[df["opf_converged"] == True].copy()

    df = df[df["subsystem"].notna()].copy()
    if df.empty:
        raise RuntimeError("No data left after filtering.")

    df["date"] = make_date(df["year"], df["month"])
    eps = float(args.eps)

    # ------------------------------------------------------------------
    # Ensure expected columns exist (backward compatible with older runs)
    # All of these are MW if present.
    # ------------------------------------------------------------------
    must_cols = [
        "ens_mw",
        "slack_import_mw",
        "deficit_mw",
        "total_demand_mw",
        "total_available_gen_mw",
        "binding_lines_count",
        "binding_line_names",
        "binding_line_loading_pct_max",
        "demand_mw_subsystem",
        "available_gen_mw_subsystem",
        "ens_mw_subsystem",
        "stranded_curtailment_mw",
        "total_zone_surplus_mw",
        "total_zone_deficit_mw",
    ]
    for c in must_cols:
        if c not in df.columns:
            df[c] = np.nan

    # ------------------------------------------------------------------
    # 1) MONTHLY SYSTEM TABLE (all MW)
    # ------------------------------------------------------------------
    monthly = df.groupby(["trial", "year", "month", "date"]).agg(
        total_curtailment_mw=("curtailed_mw", "sum"),
        total_pmax_mw=("pmax_mw", "sum"),
        total_dispatch_mw=("dispatch_mw", "sum"),
        ens_mw=("ens_mw", "max"),
        slack_import_mw=("slack_import_mw", "max"),
        deficit_mw=("deficit_mw", "max"),
        total_demand_mw=("total_demand_mw", "max"),
        total_available_gen_mw=("total_available_gen_mw", "max"),
        binding_lines_count=("binding_lines_count", "max"),
        binding_line_loading_pct_max=("binding_line_loading_pct_max", "max"),
        binding_line_names=("binding_line_names", first_nonempty_string),
    ).reset_index()

    # Diagnostics: unused capacity and gross system balance (MW)
    monthly["unused_cap_mw"] = np.maximum(
        monthly["total_pmax_mw"] - monthly["total_dispatch_mw"], 0.0
    )
    monthly["gen_minus_demand_mw"] = (
        monthly["total_available_gen_mw"] - monthly["total_demand_mw"]
    )
    monthly["is_surplus"] = monthly["gen_minus_demand_mw"] > eps
    monthly["is_deficit"] = monthly["gen_minus_demand_mw"] < -eps

    monthly["has_curtailment"] = monthly["total_curtailment_mw"] > eps
    monthly["has_ens"] = monthly["ens_mw"] > eps
    monthly["is_congested"] = monthly["binding_lines_count"].fillna(0) > 0

    # Fractions (unitless)
    monthly["curtailment_pct_system"] = np.where(
        monthly["total_pmax_mw"] > eps,
        monthly["total_curtailment_mw"] / monthly["total_pmax_mw"],
        0.0,
    )
    monthly["ens_pct_demand"] = np.where(
        monthly["total_demand_mw"] > eps,
        monthly["ens_mw"] / monthly["total_demand_mw"],
        0.0,
    )
    monthly["slack_import_pct_demand"] = np.where(
        monthly["total_demand_mw"] > eps,
        monthly["slack_import_mw"] / monthly["total_demand_mw"],
        0.0,
    )

    def regime(row):
        if row["has_ens"] and row["has_curtailment"]:
            return "both_ens_and_curtailment"
        if row["has_ens"]:
            return "ens_only"
        if row["has_curtailment"]:
            return "curtailment_only"
        return "neither_ens_nor_curtailment"

    monthly["regime"] = monthly.apply(regime, axis=1)

    # ------------------------------------------------------------------
    # 2) SUBSYSTEM MONTHLY TABLE (all MW)
    # ------------------------------------------------------------------
    subsys_monthly = df.groupby(
        ["trial", "year", "month", "date", "subsystem"]
    ).agg(
        pmax_mw=("pmax_mw", "sum"),
        dispatch_mw=("dispatch_mw", "sum"),
        curtailed_mw=("curtailed_mw", "sum"),
        demand_mw_subsystem=("demand_mw_subsystem", "max"),
        available_gen_mw_subsystem=("available_gen_mw_subsystem", "max"),
        ens_mw_subsystem=("ens_mw_subsystem", "max"),
    ).reset_index()

    # Curtailment fraction per subsystem
    subsys_monthly["curtailment_pct"] = np.where(
        subsys_monthly["pmax_mw"] > eps,
        subsys_monthly["curtailed_mw"] / subsys_monthly["pmax_mw"],
        0.0,
    )

    # ------------------------------------------------------------------
    # 3) STRANDED VS SCARCITY FLAGS (monthly, zone-based)
    #
    # New simulator may not provide demand_mw_subsystem / available_gen_mw_subsystem.
    # We only compute zone-balances if BOTH are present and non-NaN somewhere.
    # Otherwise we disable stranded metrics (set to 0) to avoid bogus conclusions.
    # ------------------------------------------------------------------
    has_zone_demand = subsys_monthly["demand_mw_subsystem"].notna().any()
    has_zone_avail = subsys_monthly["available_gen_mw_subsystem"].notna().any()
    use_zone_balance = bool(has_zone_demand and has_zone_avail)

    if use_zone_balance:
        # Zone net balance = available - demand (MW)
        subsys_monthly["net_balance_mw"] = (
            subsys_monthly["available_gen_mw_subsystem"]
            - subsys_monthly["demand_mw_subsystem"]
        )

        subsys_monthly["curtailment_in_surplus_zone_mw"] = np.where(
            subsys_monthly["net_balance_mw"] > eps,
            subsys_monthly["curtailed_mw"],
            0.0,
        )

        stranded = subsys_monthly.groupby(["trial", "date"]).agg(
            stranded_curtailment_mw=(
                "curtailment_in_surplus_zone_mw",
                "sum",
            ),
            total_zone_surplus_mw=(
                "net_balance_mw",
                lambda x: float(x[x > 0].sum()) if len(x) else 0.0,
            ),
            total_zone_deficit_mw=(
                "net_balance_mw",
                lambda x: float((-x[x < 0]).sum()) if len(x) else 0.0,
            ),
        ).reset_index()
    else:
        # No per-zone demand/avail in the results file: stranded metrics disabled.
        subsys_monthly["net_balance_mw"] = 0.0
        subsys_monthly["curtailment_in_surplus_zone_mw"] = 0.0
        stranded = monthly[["trial", "date"]].copy()
        stranded["stranded_curtailment_mw"] = 0.0
        stranded["total_zone_surplus_mw"] = 0.0
        stranded["total_zone_deficit_mw"] = 0.0
        print(
            "Note: demand_mw_subsystem / available_gen_mw_subsystem "
            "not found or all-NaN in results; stranded-curtailment "
            "and zone surplus/deficit metrics are set to 0."
        )

    monthly = monthly.merge(stranded, on=["trial", "date"], how="left").fillna(0.0)

    monthly["ens_is_deliverability_driven"] = (
        monthly["has_ens"] & (monthly["total_zone_surplus_mw"] > eps)
    )
    monthly["ens_is_scarcity_driven"] = (
        monthly["has_ens"] & (monthly["total_zone_surplus_mw"] <= eps)
    )
    monthly["ens_with_unused_cap"] = monthly["has_ens"] & (
        monthly["unused_cap_mw"] > eps
    )

    # Save monthly_detailed
    monthly_path = out_dir / f"{in_path.stem}_monthly_detailed.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"Saved: {monthly_path}")

    # Save subsystem_monthly
    subsys_monthly_path = out_dir / f"{in_path.stem}_subsystem_monthly.csv"
    subsys_monthly.to_csv(subsys_monthly_path, index=False)
    print(f"Saved: {subsys_monthly_path}")

    # ------------------------------------------------------------------
    # 4) SUBSYSTEM STATS ACROSS TRIALS (MW)
    # ------------------------------------------------------------------
    subsys_stats = subsys_monthly.groupby(
        ["year", "month", "subsystem"]
    ).agg(
        mean_curtailed_mw=("curtailed_mw", "mean"),
        p05_curtailed_mw=("curtailed_mw", lambda x: q(x, 0.05)),
        p95_curtailed_mw=("curtailed_mw", lambda x: q(x, 0.95)),
        mean_curtailment_pct=("curtailment_pct", "mean"),
        p05_curtailment_pct=("curtailment_pct", lambda x: q(x, 0.05)),
        p95_curtailment_pct=("curtailment_pct", lambda x: q(x, 0.95)),
        mean_ens_mw=("ens_mw_subsystem", "mean"),
        p95_ens_mw=("ens_mw_subsystem", lambda x: q(x, 0.95)),
    ).reset_index()
    subsys_stats["date"] = make_date(subsys_stats["year"], subsys_stats["month"])

    subsys_stats_path = out_dir / f"{in_path.stem}_subsystem_stats.csv"
    subsys_stats.to_csv(subsys_stats_path, index=False)
    print(f"Saved: {subsys_stats_path}")

    # ------------------------------------------------------------------
    # 5) REGIME COUNTS
    # ------------------------------------------------------------------
    regime_counts = monthly.groupby(
        [
            "regime",
            "is_congested",
            "ens_is_deliverability_driven",
            "ens_is_scarcity_driven",
        ]
    ).size().reset_index(name="count")
    regime_counts["fraction"] = regime_counts["count"] / len(monthly)

    regime_counts_path = out_dir / f"{in_path.stem}_regime_counts.csv"
    regime_counts.to_csv(regime_counts_path, index=False)
    print(f"Saved: {regime_counts_path}")

    # ------------------------------------------------------------------
    # 6) BINDING LINE IMPACTS
    # ------------------------------------------------------------------
    bl_long = explode_binding_lines(monthly, col="binding_line_names")
    if not bl_long.empty:
        bl_long = bl_long.merge(
            monthly[
                [
                    "trial",
                    "date",
                    "has_ens",
                    "has_curtailment",
                    "ens_mw",
                    "total_curtailment_mw",
                    "regime",
                ]
            ],
            on=["trial", "date"],
            how="left",
        )

        n_all = len(monthly)
        n_ens = max(int(monthly["has_ens"].sum()), 1)
        n_cur = max(int(monthly["has_curtailment"].sum()), 1)
        n_both = max(
            int(((monthly["has_ens"]) & (monthly["has_curtailment"])).sum()), 1
        )

        f_all = bl_long["binding_line"].value_counts() / n_all
        f_ens = (
            bl_long[bl_long["has_ens"]]["binding_line"].value_counts() / n_ens
        )
        f_cur = (
            bl_long[bl_long["has_curtailment"]]["binding_line"].value_counts()
            / n_cur
        )
        f_both = (
            bl_long[(bl_long["has_ens"]) & (bl_long["has_curtailment"])]["binding_line"]
            .value_counts()
            / n_both
        )

        means_when_binding = bl_long.groupby("binding_line").agg(
            count=("binding_line", "size"),
            mean_ens_when_binding=("ens_mw", "mean"),
            mean_curtailment_when_binding=("total_curtailment_mw", "mean"),
        )

        impacts = means_when_binding.copy()
        impacts["freq_overall"] = f_all
        impacts["freq_in_ens_months"] = f_ens
        impacts["freq_in_curtailment_months"] = f_cur
        impacts["freq_in_both_months"] = f_both

        impacts = (
            impacts.reset_index()
            .fillna(0.0)
            .sort_values("freq_overall", ascending=False)
        )

        impacts_path = out_dir / f"{in_path.stem}_binding_line_impacts.csv"
        impacts.to_csv(impacts_path, index=False)
        print(f"Saved: {impacts_path}")
    else:
        print("No binding_line_names found; skipping binding line impacts.")

    # ------------------------------------------------------------------
    # 7) TOP EVENTS TABLES (still MW internally)
    # ------------------------------------------------------------------
    cols = [
        "trial",
        "year",
        "month",
        "date",
        "regime",
        "is_congested",
        "is_surplus",
        "is_deficit",
        "total_demand_mw",
        "total_available_gen_mw",
        "gen_minus_demand_mw",
        "slack_import_mw",
        "deficit_mw",
        "ens_mw",
        "total_curtailment_mw",
        "stranded_curtailment_mw",
        "curtailment_pct_system",
        "ens_pct_demand",
        "binding_lines_count",
        "binding_line_loading_pct_max",
        "binding_line_names",
        "ens_is_deliverability_driven",
        "ens_is_scarcity_driven",
        "ens_with_unused_cap",
    ]

    # Some columns may be missing if input was older; filter safely
    cols = [c for c in cols if c in monthly.columns]

    top_ens = monthly.sort_values("ens_mw", ascending=False).head(50)[cols]
    top_cur = monthly.sort_values("total_curtailment_mw", ascending=False).head(50)[cols]
    top_both = monthly[
        monthly["regime"] == "both_ens_and_curtailment"
    ].sort_values(
        ["ens_mw", "total_curtailment_mw"], ascending=False
    ).head(50)[cols]

    (out_dir / f"{in_path.stem}_top_events_ens.csv").write_text(
        top_ens.to_csv(index=False)
    )
    (out_dir / f"{in_path.stem}_top_events_curtailment.csv").write_text(
        top_cur.to_csv(index=False)
    )
    (out_dir / f"{in_path.stem}_top_events_both.csv").write_text(
        top_both.to_csv(index=False)
    )

    print(f"Saved: {out_dir / f'{in_path.stem}_top_events_ens.csv'}")
    print(f"Saved: {out_dir / f'{in_path.stem}_top_events_curtailment.csv'}")
    print(f"Saved: {out_dir / f'{in_path.stem}_top_events_both.csv'}")

    # ------------------------------------------------------------------
    # 8) PLOTS – CONVERT MW → GW ONLY HERE
    # ------------------------------------------------------------------
    sys = monthly.groupby(["trial", "date"]).agg(
        ens_mw=("ens_mw", "max"),
        curtail_mw=("total_curtailment_mw", "max"),
        stranded_mw=("stranded_curtailment_mw", "max"),
        surplus_mw=("total_zone_surplus_mw", "max"),
    ).reset_index()

    stats = sys.groupby("date").agg(
        cur_mean=("curtail_mw", "mean"),
        cur_p05=("curtail_mw", lambda x: x.quantile(0.05)),
        cur_p95=("curtail_mw", lambda x: x.quantile(0.95)),
    ).reset_index()

    # System Curtailment only (GW)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        stats["date"],
        stats["cur_mean"] / 1000.0,
        label="Curtailment (mean)",
    )
    ax.fill_between(
        stats["date"],
        stats["cur_p05"] / 1000.0,
        stats["cur_p95"] / 1000.0,
        alpha=0.2,
    )

    ax.set_ylabel("GW (MW / 1000)")
    ax.set_xlabel("Month")
    ax.set_title("System Curtailment (mean ± 5–95%)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    p = fig_dir / "system_timeseries_ens_curtail_stranded.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print(f"Saved {p}")

    # Scatter: ENS vs total zone surplus (GW)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(
        sys["surplus_mw"] / 1000.0,
        sys["ens_mw"] / 1000.0,
        s=8,
        alpha=0.35,
    )
    ax.set_xlabel("Total zone surplus (GW)")
    ax.set_ylabel("ENS (GW)")
    ax.set_title("ENS vs system surplus (trial-month points)")
    fig.tight_layout()
    p = fig_dir / "system_scatter_ens_vs_surplus.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print(f"Saved {p}")

    # Heatmap of mean curtailment by subsystem vs time (GW)
    agg = subsys_monthly.groupby(["subsystem", "date"]).agg(
        mean_curtailed_mw=("curtailed_mw", "mean")
    ).reset_index()
    pivot = agg.pivot_table(
        index="subsystem",
        columns="date",
        values="mean_curtailed_mw",
        fill_value=0.0,
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values / 1000.0, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in pivot.columns], rotation=90
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Subsystem")
    ax.set_title("Mean Curtailment Heatmap (GW)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("GW curtailed")
    fig.tight_layout()
    p = fig_dir / "heatmap_mean_curtailment.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print(f"Saved {p}")

    # Boxplot per subsystem curtailment distribution (GW)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = [
        subsys_monthly[subsys_monthly["subsystem"] == s]["curtailed_mw"]
        / 1000.0
        for s in pivot.index
    ]
    ax.boxplot(data, labels=pivot.index, showfliers=False)
    ax.set_xlabel("Subsystem")
    ax.set_ylabel("Curtailment (GW)")
    ax.set_title("Curtailment Distribution per Subsystem")
    fig.tight_layout()
    p = fig_dir / "boxplot_curtailment_by_subsystem.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print(f"Saved {p}")

    # ------------------------------------------------------------------
    # 9) DASHBOARD
    # ------------------------------------------------------------------
    print("\n=== DASHBOARD ===")
    print(f"Months (trial-months): {len(monthly)}")
    print("Regime counts:")
    print(monthly["regime"].value_counts())
    print("\nShare congested:", float(monthly["is_congested"].mean()))
    print("Share with ENS:", float(monthly["has_ens"].mean()))
    print("Share with curtailment:", float(monthly["has_curtailment"].mean()))
    print(
        "Share with BOTH:",
        float((monthly["regime"] == "both_ens_and_curtailment").mean()),
    )
    print(
        "Share ENS deliverability-driven:",
        float(monthly["ens_is_deliverability_driven"].mean()),
    )
    print(
        "Share ENS scarcity-driven:",
        float(monthly["ens_is_scarcity_driven"].mean()),
    )
    print("=================\n")


if __name__ == "__main__":
    main()
