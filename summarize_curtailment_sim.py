"""Create per-subsystem statistics + visualizations from simulation outputs."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Summarize curtailment simulation results.")
parser.add_argument("--input", default='results/curtailment_simulations.csv', help="Simulation CSV from run_forecast_curtailment_sim.py")
parser.add_argument("--fig-dir", default="results/figures", help="Directory to store generated figures.")
args = parser.parse_args()

path = Path(args.input)
if not path.exists():
    raise FileNotFoundError(path)

df = pd.read_csv(path)

# ------------------------------------------------------------------
# Per-subsystem monthly statistics (mean + tail percentiles)
# ------------------------------------------------------------------
agg = df.groupby(["subsystem", "year", "month"]).agg(
    mean_curtailed_mw=("curtailed_mw", "mean"),
    p95_curtailed_mw=("curtailed_mw", lambda x: x.quantile(0.95)),
    p05_curtailed_mw=("curtailed_mw", lambda x: x.quantile(0.05)),
).reset_index()

out_path = path.with_name(path.stem + "_subsystem_summary.csv")
agg.to_csv(out_path, index=False)
print(f"Saved subsystem summary to {out_path}")

# ------------------------------------------------------------------
# Create helpful visualizations
# ------------------------------------------------------------------
fig_dir = Path(args.fig_dir)
fig_dir.mkdir(parents=True, exist_ok=True)

def make_date(series_year, series_month):
    return pd.to_datetime(series_year.astype(int).astype(str) + "-" + series_month.astype(int).astype(str) + "-01")

df["date"] = make_date(df["year"], df["month"])
agg["date"] = make_date(agg["year"], agg["month"])

# 1) Total curtailment time series with uncertainty band
monthly_total = (
    df.groupby(["trial", "date"])["curtailed_mw"].sum().reset_index()
    if "trial" in df.columns
    else df.groupby("date")["curtailed_mw"].sum().reset_index()
)

if "trial" in monthly_total.columns:
    stats = monthly_total.groupby("date")["curtailed_mw"].agg(
        mean="mean", p05=lambda x: x.quantile(0.05), p95=lambda x: x.quantile(0.95)
    ).reset_index()
else:
    stats = monthly_total.rename(columns={"curtailed_mw": "mean"})
    stats["p05"] = stats["mean"]
    stats["p95"] = stats["mean"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stats["date"], stats["mean"] / 1000.0, label="Mean curtailment")
ax.fill_between(
    stats["date"],
    stats["p05"] / 1000.0,
    stats["p95"] / 1000.0,
    alpha=0.2,
    label="5th-95th percentile",
)
ax.set_ylabel("Curtailment (GW)")
ax.set_xlabel("Month")
ax.set_title("Monthly Total Curtailment (mean ± 5-95%)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
ts_path = fig_dir / "curtailment_time_series.png"
fig.savefig(ts_path, dpi=200)
plt.close(fig)
print(f"Saved {ts_path}")

# 2) Heatmap of mean curtailment by subsystem vs time
pivot = agg.pivot_table(index="subsystem", columns="date", values="mean_curtailed_mw", fill_value=0.0)
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(pivot.values / 1000.0, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([d.strftime("%Y-%m") for d in pivot.columns], rotation=90)
ax.set_xlabel("Month")
ax.set_ylabel("Subsystem")
ax.set_title("Mean Curtailment Heatmap (GW)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("GW curtailed")
fig.tight_layout()
heat_path = fig_dir / "curtailment_heatmap.png"
fig.savefig(heat_path, dpi=200)
plt.close(fig)
print(f"Saved {heat_path}")

# 3) Boxplot per subsystem of curtailment distribution
fig, ax = plt.subplots(figsize=(8, 5))
data = [df[df["subsystem"] == subsys]["curtailed_mw"] / 1000.0 for subsys in pivot.index]
ax.boxplot(data, labels=pivot.index, showfliers=False)
ax.set_xlabel("Subsystem")
ax.set_ylabel("Curtailment (GW)")
ax.set_title("Curtailment Distribution per Subsystem")
fig.tight_layout()
box_path = fig_dir / "curtailment_boxplot.png"
fig.savefig(box_path, dpi=200)
plt.close(fig)
print(f"Saved {box_path}")
