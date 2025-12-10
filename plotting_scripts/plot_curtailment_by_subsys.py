import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "results/curtailment_simulations.csv"
OUT_PNG = "results/figures/curtailment_by_subsystem_stacked.png"

df = pd.read_csv(IN_CSV)

# monthly totals per trial (sum across subsystems)
df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01")

sub_month = df.groupby(["trial", "date", "subsystem"])["curtailed_mw"].sum().reset_index()

# average across trials for each month/subsystem
mean_month = sub_month.groupby(["date", "subsystem"])["curtailed_mw"].mean().reset_index()

pivot = mean_month.pivot(index="date", columns="subsystem", values="curtailed_mw").fillna(0.0) / 1000.0  # GW

pivot = pivot.sort_index()

plt.figure(figsize=(11, 5))
pivot.plot(kind="area", stacked=True)
plt.ylabel("Curtailment (GW)")
plt.xlabel("Month")
plt.title("Curtailment by subsystem (mean across trials)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print("Saved:", OUT_PNG)
