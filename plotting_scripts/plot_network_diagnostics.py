"""
plot_network_diagnostics.py

Reads the *monthly_detailed* output (from summarize_curtailment_detailed.py)
and produces a small set of figures + a corridor-limit table to diagnose
whether the zonal network (corridor aggregation) is causing trapped surplus
(ENS + curtailment + slack imports simultaneously).

Usage:
  python plot_network_diagnostics.py \
      --monthly results/curtailment_simulations_monthly_detailed.csv \
      --out-dir results/diagnostic_figures \
      --net-json models/pandapower_snapshots/brazil_network_zonal_5bus_from_tx.json

Outputs:
  - system_*.png (time series w/ uncertainty bands)
  - ens_vs_unused_scatter.png
  - binding_corridor_frequency.png
  - corridor_limits_from_net.csv (computed MW limits per corridor from the net json)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQRT3 = 1.7320508075688772


def clip_small(x: pd.Series, eps: float) -> pd.Series:
    """Clip tiny numerical noise to 0; keep sign for larger values."""
    x = pd.to_numeric(x, errors="coerce")
    return x.where(x.abs() > eps, 0.0)


def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01",
            errors="coerce",
        )
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def summarize_over_trials(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    For a monthly table with trial dimension: compute mean/p05/p95 over trials per date.
    If there's no 'trial' column, returns a single series (mean==p05==p95).
    """
    if "trial" in df.columns:
        out = (
            df.groupby("date")[value_col]
            .agg(
                mean="mean",
                p05=lambda s: s.quantile(0.05),
                p95=lambda s: s.quantile(0.95),
            )
            .reset_index()
        )
    else:
        out = df[["date", value_col]].copy()
        out = out.rename(columns={value_col: "mean"})
        out["p05"] = out["mean"]
        out["p95"] = out["mean"]
    return out.sort_values("date")


def make_band_plot(stats: pd.DataFrame, ylab: str, title: str, out_path: Path, scale=1.0):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stats["date"], stats["mean"] / scale, label="Mean")
    ax.fill_between(stats["date"], stats["p05"] / scale, stats["p95"] / scale, alpha=0.2, label="5–95%")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def explode_binding_lines(monthly: pd.DataFrame, col="binding_line_names") -> pd.DataFrame:
    if col not in monthly.columns:
        return pd.DataFrame(columns=["trial", "date", "binding_line"])
    tmp = monthly[["trial", "date", col]].copy() if "trial" in monthly.columns else monthly[["date", col]].copy()
    tmp[col] = tmp[col].fillna("").astype(str)
    tmp["binding_line"] = tmp[col].str.split(";")
    tmp = tmp.drop(columns=[col]).explode("binding_line")
    tmp["binding_line"] = tmp["binding_line"].astype(str).str.strip()
    tmp = tmp[tmp["binding_line"] != ""].copy()
    return tmp


def binding_frequency_plot(monthly: pd.DataFrame, out_path: Path, eps: float, top_k: int = 12):
    """
    Bar chart: how often each corridor binds (overall), plus conditional frequency when ENS>0 and when unused_cap>0.
    """
    if "binding_line_names" not in monthly.columns:
        print("No binding_line_names column; skipping binding frequency plot.")
        return

    base_cols = ["date"]
    if "trial" in monthly.columns:
        base_cols.append("trial")

    # Outcomes / conditioning flags
    m = monthly.copy()
    m["has_ens"] = clip_small(m.get("ens_mw", 0.0), eps) > eps
    m["has_unused"] = clip_small(m.get("unused_cap_mw", 0.0), eps) > eps
    m["ens_and_unused"] = m["has_ens"] & m["has_unused"]

    bl = explode_binding_lines(m, "binding_line_names")
    if bl.empty:
        print("binding_line_names is empty everywhere; skipping binding frequency plot.")
        return

    # attach conditioning flags
    key = base_cols
    bl = bl.merge(m[key + ["has_ens", "ens_and_unused"]], on=key, how="left")

    # denominators: number of (trial,date) months in each conditioning set
    if "trial" in m.columns:
        n_all = len(m)
        n_ens = max(int(m["has_ens"].sum()), 1)
        n_eu = max(int(m["ens_and_unused"].sum()), 1)
    else:
        n_all = len(m)
        n_ens = max(int(m["has_ens"].sum()), 1)
        n_eu = max(int(m["ens_and_unused"].sum()), 1)

    f_all = bl["binding_line"].value_counts() / n_all
    f_ens = bl.loc[bl["has_ens"] == True, "binding_line"].value_counts() / n_ens
    f_eu = bl.loc[bl["ens_and_unused"] == True, "binding_line"].value_counts() / n_eu

    top = f_all.sort_values(ascending=False).head(top_k).index.tolist()
    plot_df = pd.DataFrame({
        "binding_line": top,
        "freq_overall": [float(f_all.get(k, 0.0)) for k in top],
        "freq_in_ens_months": [float(f_ens.get(k, 0.0)) for k in top],
        "freq_in_ens_and_unused_months": [float(f_eu.get(k, 0.0)) for k in top],
    })

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(plot_df))
    w = 0.28
    ax.bar(x - w, plot_df["freq_overall"], width=w, label="Overall")
    ax.bar(x,      plot_df["freq_in_ens_months"], width=w, label="Given ENS")
    ax.bar(x + w,  plot_df["freq_in_ens_and_unused_months"], width=w, label="Given ENS & Unused Cap")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["binding_line"], rotation=45, ha="right")
    ax.set_ylabel("Frequency (fraction of months)")
    ax.set_title("Binding corridor frequency (overall and conditional)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def ens_vs_unused_scatter(monthly: pd.DataFrame, out_path: Path, eps: float):
    m = monthly.copy()
    m["ens_mw"] = clip_small(m.get("ens_mw", 0.0), eps)
    m["unused_cap_mw"] = clip_small(m.get("unused_cap_mw", 0.0), eps)

    # Flag the “dominant” corridor if it exists
    dominant = None
    if "binding_line_names" in m.columns:
        # take most common nonempty binding_line_names string (not perfect, but useful)
        nonempty = m["binding_line_names"].fillna("").astype(str)
        nonempty = nonempty[nonempty.str.strip() != ""]
        if len(nonempty) > 0:
            dominant = nonempty.value_counts().index[0]

    fig, ax = plt.subplots(figsize=(7, 6))

    if dominant is None:
        ax.scatter(m["unused_cap_mw"], m["ens_mw"], alpha=0.6)
        ax.set_title("ENS vs Unused Capacity")
    else:
        mask_dom = m["binding_line_names"].fillna("").astype(str).str.contains(dominant, regex=False)
        ax.scatter(m.loc[~mask_dom, "unused_cap_mw"], m.loc[~mask_dom, "ens_mw"], alpha=0.5, label="Other / none")
        ax.scatter(m.loc[mask_dom, "unused_cap_mw"],  m.loc[mask_dom, "ens_mw"],  alpha=0.7, label=f"Contains: {dominant}")
        ax.legend()
        ax.set_title("ENS vs Unused Capacity (highlighting most-common binding corridor set)")

    # Reference lines
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)

    ax.set_xlabel("Unused capacity (MW) = total_available_gen_mw - total_dispatch_mw")
    ax.set_ylabel("ENS (MW)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def corridor_limits_from_zonal_net(net_json: Path, out_csv: Path):
    """
    Reads a pandapower zonal net JSON and computes approximate MW limit per corridor line:
      limit_mw ≈ sqrt(3) * vn_kv * max_i_ka   (pf~1)
    Also exports x_ohm_total (x_ohm_per_km * length_km).
    """
    try:
        import pandapower as pp
    except Exception as e:
        print(f"Could not import pandapower; skipping corridor limit extraction. ({e})")
        return

    net = pp.from_json(str(net_json))
    if net.line.empty:
        print("Net has no lines; skipping corridor limit extraction.")
        return

    # map bus idx -> vn_kv
    vn = net.bus["vn_kv"].to_dict()

    rows = []
    for i, r in net.line.iterrows():
        fr = int(r["from_bus"])
        to = int(r["to_bus"])
        name = str(r.get("name", f"line_{i}"))
        length = float(r.get("length_km", 1.0))
        x_per_km = float(r.get("x_ohm_per_km", np.nan))
        x_total = x_per_km * length if np.isfinite(x_per_km) else np.nan
        max_i_ka = float(r.get("max_i_ka", np.nan))

        # choose voltage from from_bus (should be same in zonal net)
        v_kv = float(vn.get(fr, np.nan))
        limit_mw = SQRT3 * v_kv * max_i_ka if (np.isfinite(v_kv) and np.isfinite(max_i_ka)) else np.nan

        rows.append({
            "line_index": i,
            "name": name,
            "from_bus": fr,
            "to_bus": to,
            "vn_kv_from": v_kv,
            "length_km": length,
            "max_i_ka": max_i_ka,
            "limit_mw_approx": limit_mw,
            "x_ohm_total": x_total,
            "x_ohm_per_km": x_per_km,
        })

    out = pd.DataFrame(rows).sort_values("limit_mw_approx", ascending=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved corridor limits from net JSON: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly", required=True, help="Path to *_monthly_detailed.csv")
    ap.add_argument("--out-dir", required=True, help="Where to save figures / csv")
    ap.add_argument("--net-json", default=None, help="Optional: zonal net JSON to extract corridor limits")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    monthly_path = Path(args.monthly)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(monthly_path)
    df = ensure_date(df)

    eps = float(args.eps)

    # Clip numerical noise in key fields
    for c in ["ens_mw", "slack_import_mw", "deficit_mw", "total_curtailment_mw", "unused_cap_mw"]:
        if c in df.columns:
            df[c] = clip_small(df[c], eps)

    # If unused_cap_mw isn't present, compute it if possible
    if "unused_cap_mw" not in df.columns:
        if ("total_available_gen_mw" in df.columns) and ("total_dispatch_mw" in df.columns):
            df["unused_cap_mw"] = clip_small(df["total_available_gen_mw"] - df["total_dispatch_mw"], eps)
        else:
            df["unused_cap_mw"] = 0.0

    # --- Time series plots (system level) ---
    if "total_curtailment_mw" in df.columns:
        cur = summarize_over_trials(df, "total_curtailment_mw")
        make_band_plot(
            cur,
            ylab="Curtailment (GW)",
            title="System Curtailment (mean ± 5–95%)",
            out_path=out_dir / "system_curtailment_timeseries.png",
            scale=1000.0,
        )

    if "ens_mw" in df.columns:
        ens = summarize_over_trials(df, "ens_mw")
        make_band_plot(
            ens,
            ylab="ENS (GW)",
            title="System ENS (mean ± 5–95%)",
            out_path=out_dir / "system_ens_timeseries.png",
            scale=1000.0,
        )

    if "slack_import_mw" in df.columns:
        slk = summarize_over_trials(df, "slack_import_mw")
        make_band_plot(
            slk,
            ylab="Slack imports (GW)",
            title="Slack Imports (mean ± 5–95%)",
            out_path=out_dir / "system_slack_imports_timeseries.png",
            scale=1000.0,
        )

    if "unused_cap_mw" in df.columns:
        unc = summarize_over_trials(df, "unused_cap_mw")
        make_band_plot(
            unc,
            ylab="Unused capacity (GW)",
            title="Unused Capacity = Available Gen − Dispatch (mean ± 5–95%)",
            out_path=out_dir / "system_unused_capacity_timeseries.png",
            scale=1000.0,
        )

    # --- Scatter that diagnoses “trapped surplus” ---
    ens_vs_unused_scatter(df, out_dir / "ens_vs_unused_scatter.png", eps)

    # --- Binding corridor frequency plot ---
    binding_frequency_plot(df, out_dir / "binding_corridor_frequency.png", eps, top_k=12)

    # --- Optional corridor limit extraction from zonal net JSON ---
    if args.net_json:
        net_json = Path(args.net_json)
        if net_json.exists():
            corridor_limits_from_zonal_net(net_json, out_dir / "corridor_limits_from_net.csv")
        else:
            print(f"Net JSON not found: {net_json}")

    print(f"Saved diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
