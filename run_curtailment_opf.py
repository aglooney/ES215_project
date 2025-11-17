"""
Curtailment-focused DC-OPF using:
- Monthly subsystem demand (from demand_projection.xlsx)
- ML-predicted or actual generation (from prior scenario file)
- Network from build_network_with_generation.py

This script:
1. Loads demand and converts monthly MWh -> MW for a "peak hour".
2. Attaches loads to the subsystem hub buses.
3. Applies generation maxima from your scenario.
4. Runs DC-OPF (linear optimal power flow).
5. Computes curtailment per subsystem.

Modify SCENARIO_DATE to choose which gen scenario to use.
"""

import pandas as pd
import numpy as np
import pandapower as pp
from pandapower.pypower import idx_gen
from pandapower.networks import case9
import calendar

# ============================================================
# CONFIGURATION
# ============================================================

DEMAND_PATH  = "data/demand_data/demand_projection_clean.csv"       # The file you uploaded
SCENARIO_GEN = "models/scenarios/gen_subsystem_2020-12-10.parquet"  # adjust as needed

NETWORK_PATH = "models/pandapower_snapshots/brazil_network_with_subsys_generation.json"

# Demand parameters
DEMAND_SCALING = 1.3   # peak-hour multiplier
SUBSYS_COL = "subsystem"

# Generation column (actual or ML)
GEN_COL = "gen_total_mw"     # could switch to "gen_ml_mw"

# ============================================================
# LOAD NETWORK
# ============================================================

print("🔌 Loading network...")
net = pp.from_json(NETWORK_PATH)
print("  Loaded network with:")
print(f"  - {len(net.bus)} buses")
print(f"  - {len(net.line)} lines")
print(f"  - {len(net.sgen)} generators\n")


# ============================================================
# LOAD GENERATION SCENARIO
# ============================================================

print("⚡ Loading generation scenario...")
gen_df = pd.read_parquet(SCENARIO_GEN)

print(gen_df.columns)
print(gen_df.head())

# Expected columns: ["subsystem", "gen_total_mw", ...]
if SUBSYS_COL not in gen_df.columns or GEN_COL not in gen_df.columns:
    raise ValueError("Generation scenario file missing required columns.")

subsys_gen = gen_df.set_index(SUBSYS_COL)[GEN_COL].to_dict()

print("  Generation (MW) by subsystem:")
print(subsys_gen)
print()


# ============================================================
# LOAD MONTHLY DEMAND & CONVERT TO MW  (NUMERIC MONTH VERSION)
# ============================================================

print("📘 Loading monthly demand...")
df_demand = pd.read_csv(DEMAND_PATH)



required_cols = {"state", "subsystem", "year", "month", "MWh"}
if not required_cols.issubset(df_demand.columns):
    raise ValueError(f"Demand file must contain: {required_cols}")

# Ensure subsystem format matches scenario
df_demand["subsystem"] = df_demand["subsystem"].str.upper().str.strip()

df_demand["subsystem"] = (
    df_demand["subsystem"]
    .str.upper()
    .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
)
# Convert from kWh → MWh if needed (kWh data should be 1000× larger)
if df_demand["MWh"].max() > 1e9:
    print("⚠️ Detected kWh, converting to MWh...")
    df_demand["MWh"] = df_demand["MWh"] / 1000.0



print("DF DEMAND HEAD:")
print(df_demand.head())

print("Unique years:", df_demand["year"].unique())
print("Unique months:", df_demand["month"].unique())
print("Unique subsystems:", df_demand["subsystem"].unique())


# ----------------------------
# Choose which year/month to run
# ----------------------------
YEAR = 2020
MONTH = 12   # numeric: 12 = December

df_month = df_demand[(df_demand["year"] == YEAR) &
                     (df_demand["month"] == MONTH)]

if df_month.empty:
    raise ValueError(f"No demand found for YEAR={YEAR}, MONTH={MONTH}.")

# ----------------------------
# Aggregate MWh by subsystem
# ----------------------------
df_subsys = df_month.groupby("subsystem")["MWh"].sum().reset_index()

# ----------------------------
# Convert monthly MWh → MW
# ----------------------------
import calendar
days = calendar.monthrange(YEAR, MONTH)[1]
hours = days * 24

df_subsys["P_avg"] = df_subsys["MWh"] / hours

DEMAND_SCALING = 1.3     # peak-hour approximation
df_subsys["P_peak"] = df_subsys["P_avg"] * DEMAND_SCALING

subsys_demand_mw = dict(zip(df_subsys["subsystem"], df_subsys["P_peak"]))

if not subsys_demand_mw:
    raise ValueError("Demand aggregation produced no subsystems; check YEAR/MONTH selection.")

print("  Demand MW by subsystem (peak-hour approximation):")
print(subsys_demand_mw)
print()




# ============================================================
# ATTACH DEMAND TO HUB BUSES
# ============================================================

print("🔗 Attaching subsystem demand to network...")

# Build mapping from subsystem name to hub bus using sgens
hub_lookup = {}
for _, sgen_row in net.sgen.iterrows():
    name = str(sgen_row["name"])  # e.g. "GEN_NORTE"
    if name.startswith("GEN_"):
        subsys = name.replace("GEN_", "").upper()
        hub_lookup[subsys] = sgen_row["bus"]

print("  Hub buses:", hub_lookup, "\n")

# Add loads at hub buses
for subsys, p_mw in subsys_demand_mw.items():
    if subsys not in hub_lookup:
        print(f"  ⚠ Subsystem {subsys} missing hub bus, skipping load.")
        continue
    bus = hub_lookup[subsys]
    pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=0.0, name=f"LOAD_{subsys}")

print(f"  Added {len(net.load)} loads.\n")



# ============================================================
# SET GENERATOR MAX OUTPUTS (CURTAILMENT POSSIBILITY)
# ============================================================

print("🎯 Setting generator limits...")

for i, row in net.sgen.iterrows():
    subsys = row["name"].replace("GEN_", "")
    if subsys in subsys_gen:
        pmax = subsys_gen[subsys]
        net.sgen.at[i, "max_p_mw"] = pmax
        net.sgen.at[i, "min_p_mw"] = 0.0
        net.sgen.at[i, "in_service"] = True
    else:
        print(f"  ⚠ Subsystem {subsys} has no gen scenario entry; forcing pmax=0.")
        net.sgen.at[i, "max_p_mw"] = 0.0

print("  Generator limits applied.\n")

# Identify SUDESTE hub bus
sudeste_row = net.sgen[net.sgen["name"] == "GEN_SUDESTE"].iloc[0]
slack_bus = sudeste_row["bus"]

# Add slack (reference bus)
slack_idx = pp.create_ext_grid(
    net,
    bus=slack_bus,
    vm_pu=1.0,
    va_degree=0.0,
    name="SLACK_SUDESTE"
)

print(f"Added SLACK at bus {slack_bus}")

print("HUB LOOKUP KEYS:", sorted(hub_lookup.keys()))
print("DEMAND KEYS:", sorted(subsys_demand_mw.keys()))


# ============================================================
# ECONOMIC DISPATCH SETUP (COSTS)
# Ensure OPF prefers subsystem generators over slack power.
# ============================================================

if not net.poly_cost.empty:
    net.poly_cost.drop(net.poly_cost.index, inplace=True)

SLACK_COST = 1000.0  # make imports expensive
GEN_COST = 1.0       # treat subsystem generation as cheap/neutral

pp.create_poly_cost(
    net,
    element=slack_idx,
    et="ext_grid",
    cp1_eur_per_mw=SLACK_COST,
    cp0_eur=0.0
)

for idx in net.sgen.index:
    pp.create_poly_cost(
        net,
        element=idx,
        et="sgen",
        cp1_eur_per_mw=GEN_COST,
        cp0_eur=0.0
    )


# ============================================================
# RUN DC-OPF
# ============================================================

def ensure_scaling(df):
    if "scaling" not in df.columns:
        df["scaling"] = 1.0

# Apply to all major element types
ensure_scaling(net.load)
ensure_scaling(net.sgen)
ensure_scaling(net.gen)
ensure_scaling(net.shunt)
ensure_scaling(net.storage)



print("⚙️ Running DC–OPF...")

try:
    pp.runopp(net, calculate_voltage_angles=True)
    print("  DC–OPF solved successfully.\n")
except Exception as e:
    print("❌ OPF failed.")
    raise e


# ============================================================
# CURTAILMENT CALCULATION
# ============================================================

print("📉 Curtailment results:")

curtailment_results = []

for i, row in net.sgen.iterrows():
    subsys = row["name"].replace("GEN_", "")
    pmax = row["max_p_mw"]
    pdispatch = net.res_sgen.at[i, "p_mw"]
    curtailed = pmax - pdispatch

    curtailment_results.append({
        "subsystem": subsys,
        "pmax_mw": pmax,
        "dispatch_mw": pdispatch,
        "curtailed_mw": curtailed,
        "curtailment_pct": curtailed / pmax if pmax > 0 else 0
    })

df_curta = pd.DataFrame(curtailment_results)
print(df_curta)
print("\n")

# Save results
df_curta.to_csv("results/curtailment_results.csv", index=False)
print("💾 Saved: results/curtailment_results.csv")
