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

import argparse
import pandas as pd
import numpy as np
import pandapower as pp
from pandapower.pypower import idx_gen
from pandapower.networks import case9
import calendar

# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="DC OPF with subsystem demand and generation.")
parser.add_argument("--demand-path", default="data/demand_data/demand_projection_clean.csv")
parser.add_argument("--generation-path", default="data/merged_generation_weather.csv")
parser.add_argument("--year", type=int, default=2020)
parser.add_argument("--month", type=int, default=12)
parser.add_argument("--demand-scaling", type=float, default=1.3)
parser.add_argument("--gen-scale-mean", type=float, default=1.0)
parser.add_argument("--gen-scale-std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--results-path", default="results/curtailment_results.csv")
args = parser.parse_args()

DEMAND_PATH = args.demand_path
GENERATION_DATA = args.generation_path

NETWORK_PATH = "models/pandapower_snapshots/brazil_network_with_subsys_generation.json"

# Demand parameters
DEMAND_SCALING = args.demand_scaling   # peak-hour multiplier
SUBSYS_COL = "subsystem"

# Generation column (actual or ML)
GEN_COL = "gen_total_mw"     # kept for compatibility if scenario files are used

PROXY_COST = 50.0  # Eur/MW cost so renewables dispatch first


def monthly_generation_avg(path, year, month):
    df = pd.read_csv(path, parse_dates=["date"])
    mask = (df["date"].dt.year == year) & (df["date"].dt.month == month)
    df = df.loc[mask]
    if df.empty:
        raise ValueError(f"No generation data for {year}-{month:02d}.")
    df["subsystem"] = (
        df["subsys_name"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    )
    agg = df.groupby("subsystem")["gen_val(MW)"].sum()
    hours = pd.Period(year=year, month=month, freq="M").days_in_month * 24
    return (agg / hours).to_dict()

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
# ============================================================
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
YEAR = args.year
MONTH = args.month   # numeric: 12 = December

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
# LOAD GENERATION SCENARIO (monthly average from merged data)
# ============================================================

print("⚡ Aggregating generation for selected month...")
subsys_gen = monthly_generation_avg(GENERATION_DATA, year=YEAR, month=MONTH)

print("  Avg generation (MW) by subsystem:")
print(subsys_gen)
print()

# Apply generation uncertainty if requested
rng = np.random.default_rng(args.seed)
if args.gen_scale_std > 0:
    scale_info = {}
    for subsys, val in subsys_gen.items():
        factor = max(rng.normal(args.gen_scale_mean, args.gen_scale_std), 0)
        subsys_gen[subsys] = val * factor
        scale_info[subsys] = factor
    print("  Applied stochastic scaling factors:")
    print(scale_info)
    print()

# Compute proxy requirements per subsystem (domestic only)
proxy_generation_mw = {}
for subsys, demand_mw in subsys_demand_mw.items():
    if subsys == "PARAGUAI":
        continue
    gen_mw = subsys_gen.get(subsys, 0.0)
    proxy_generation_mw[subsys] = max(demand_mw - gen_mw, 0.0)

print("  Proxy generation targets (MW):")
print(proxy_generation_mw)
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

# Add proxy conventional generators so total supply can meet demand
proxy_names = []
for subsys, pmax in proxy_generation_mw.items():
    if pmax <= 0:
        continue
    if subsys not in hub_lookup:
        print(f"  ⚠ Cannot place proxy generator for {subsys}; missing hub bus.")
        continue
    name = f"GEN_CONV_{subsys}"
    pp.create_sgen(
        net,
        bus=hub_lookup[subsys],
        p_mw=0.0,
        max_p_mw=pmax,
        min_p_mw=0.0,
        name=name,
        type="conv",
        controllable=True,
    )
    proxy_names.append(name)

if proxy_names:
    print(f"  Added proxy generators: {proxy_names}\n")

# Add loads at hub buses
for subsys, p_mw in subsys_demand_mw.items():
    if subsys == "PARAGUAI":
        continue
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
    name = str(row["name"])
    if name.startswith("GEN_CONV_"):
        continue  # proxy generators already configured
    subsys = name.replace("GEN_", "")
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
    name = str(net.sgen.at[idx, "name"])
    cost = PROXY_COST if name.startswith("GEN_CONV_") else GEN_COST
    pp.create_poly_cost(
        net,
        element=idx,
        et="sgen",
        cp1_eur_per_mw=cost,
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

if "min_q_mvar" not in net.sgen.columns:
    net.sgen["min_q_mvar"] = 0.0
if "max_q_mvar" not in net.sgen.columns:
    net.sgen["max_q_mvar"] = 0.0
net.sgen["controllable"] = True

if "controllable" not in net.load.columns:
    net.load["controllable"] = False


print("⚙️ Running DC–OPF...")

try:
    pp.rundcopp(net, verbose=False)
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
df_curta.to_csv(args.results_path, index=False)
print(f"💾 Saved: {args.results_path}")
def monthly_generation_avg(path, year, month):
    df = pd.read_csv(path, parse_dates=["date"])
    mask = (df["date"].dt.year == year) & (df["date"].dt.month == month)
    df = df.loc[mask]
    if df.empty:
        raise ValueError(f"No generation data for {year}-{month:02d}.")
    df["subsystem"] = (
        df["subsys_name"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    )
    agg = df.groupby("subsystem")["gen_val(MW)"].sum()
    hours = pd.Period(year=year, month=month, freq="M").days_in_month * 24
    avg_mw = (agg / hours).to_dict()
    return avg_mw
