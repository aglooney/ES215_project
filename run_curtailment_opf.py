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
import calendar

# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="DC OPF with subsystem demand and generation.")
parser.add_argument("--demand-path", default="data/demand_data/demand_projection_clean.csv")
parser.add_argument("--generation-path", default="data/merged_generation_weather_v2.csv")
parser.add_argument("--year", type=int, default=2020)
parser.add_argument("--month", type=int, default=12)
parser.add_argument("--demand-scaling", type=float, default=1.0)
parser.add_argument(
    "--allow-slack-imports",
    action="store_true",
    help="Allow the slack bus to inject/absorb power. "
    "If omitted, total subsystem generation must equal subsystem demand exactly.",
)
parser.add_argument("--gen-scale-mean", type=float, default=1.0)
parser.add_argument("--gen-scale-std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--results-path", default="results/curtailment_results.csv")
parser.add_argument(
    "--min-gen-frac",
    type=float,
    default=0.1,
    help="Minimum dispatch fraction (portion of pmax) to represent must-run generation.",
)
parser.add_argument(
    "--line-loading-percent",
    type=float,
    default=100.0,
    help="Global maximum loading percentage for all lines.",
)
parser.add_argument(
    "--storage-source",
    default=None,
    help="Subsystem where simplified storage absorbs energy (generation reduced).",
)
parser.add_argument(
    "--storage-target",
    default=None,
    help="Subsystem where simplified storage releases energy (generation increased).",
)
parser.add_argument(
    "--storage-transfer-mw",
    type=float,
    default=0.0,
    help="MW shifted each month from source to target to emulate basic storage.",
)
args = parser.parse_args()

DEMAND_PATH = args.demand_path
GENERATION_DATA = args.generation_path

NETWORK_PATH = "models/brazil_5bus_network.json"

# Demand parameters
DEMAND_SCALING = args.demand_scaling   # peak-hour multiplier
SUBSYS_COL = "subsystem"

# Generation column (actual or ML)
GEN_COL = "gen_total_mw"     # kept for compatibility if scenario files are used

def normalize_subsystem(name):
    if not name:
        return None
    name = str(name).upper().strip()
    return name.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")


PROXY_COST = 50.0  # Eur/MW cost so renewables dispatch first
STORAGE_SOURCE = normalize_subsystem(args.storage_source)
STORAGE_TARGET = normalize_subsystem(args.storage_target)




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
    df["plant_type"] = df["plant_type"].astype(str).str.upper().str.strip()
    hours = pd.Period(year=year, month=month, freq="M").days_in_month * 24
    agg_total = df.groupby("subsystem")["gen_val(MW)"].sum() / hours
    agg_types = df.groupby(["subsystem", "plant_type"])["gen_val(MW)"].sum() / hours
    return agg_total.to_dict(), agg_types.to_dict()


def apply_virtual_storage(gen_vals, source, target, amount):
    if not source or not target or amount <= 0:
        return gen_vals, 0.0
    if source == target:
        return gen_vals, 0.0
    if source not in gen_vals:
        print(f"⚠️ Storage source {source} missing generation; skipping transfer.")
        return gen_vals, 0.0
    available = gen_vals.get(source, 0.0)
    if available <= 0:
        return gen_vals, 0.0
    transfer = min(amount, available)
    gen_vals[source] = available - transfer
    gen_vals[target] = gen_vals.get(target, 0.0) + transfer
    return gen_vals, transfer

# ============================================================
# LOAD NETWORK
# ============================================================

print("🔌 Loading network...")
net = pp.from_json(NETWORK_PATH)
net.load.drop(net.load.index, inplace=True)
net.gen.drop(net.gen.index, inplace=True)
net.sgen.drop(net.sgen.index, inplace=True)
if not net.poly_cost.empty:
    net.poly_cost.drop(net.poly_cost.index, inplace=True)

print("  Loaded network with:")
print(f"  - {len(net.bus)} buses")
print(f"  - {len(net.line)} lines")
print(f"  - {len(net.gen)} generators\n")
if "max_loading_percent" in net.line.columns:
    net.line["max_loading_percent"] = args.line_loading_percent


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
df_subsys["P_peak"] = df_subsys["P_avg"] * args.demand_scaling

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
subsys_gen, plant_type_gen = monthly_generation_avg(GENERATION_DATA, year=YEAR, month=MONTH)

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

HYDRO_TYPE = "HIDROELÉTRICA"
NUCLEAR_TYPE = "NUCLEAR"


def subsystem_dispatch_bounds(subsys, total_cap):
    hydro = plant_type_gen.get((subsys, HYDRO_TYPE), 0.0)
    nuclear = plant_type_gen.get((subsys, NUCLEAR_TYPE), 0.0)
    other = max(total_cap - hydro - nuclear, 0.0)
    hydro_min = 0.0
    hydro_max = hydro
    max_allowed = total_cap
    if subsys == "NORTE" and hydro > 0.0:
        hydro_min = hydro * 0.5
        hydro_max = hydro * 0.7
        max_allowed = min(max_allowed, other + nuclear + hydro_max)
    nuclear_min = nuclear * 0.95
    min_required = max(total_cap * args.min_gen_frac, hydro_min + nuclear_min)
    max_allowed = max(min_required, max_allowed)
    return min_required, max_allowed

dispatch_bounds = {subsys: subsystem_dispatch_bounds(subsys, cap) for subsys, cap in subsys_gen.items()}

transferred = 0.0
if args.storage_transfer_mw > 0 and STORAGE_SOURCE and STORAGE_TARGET:
    subsys_gen, transferred = apply_virtual_storage(
        subsys_gen, STORAGE_SOURCE, STORAGE_TARGET, args.storage_transfer_mw
    )
    if transferred > 0:
        print(
            f"🪫 Applied basic storage: shifted {transferred:.2f} MW "
            f"from {STORAGE_SOURCE} to {STORAGE_TARGET}."
        )
        print()
    else:
        print("⚠️ Storage transfer requested but no energy shifted (check inputs).")
        print()





# ============================================================
# ATTACH DEMAND TO HUB BUSES
# ============================================================

print("🔗 Attaching subsystem demand to network...")

bus_lookup = {}
for name in ["NORDESTE", "NORTE", "SUDESTE", "SUL", "PARAGUAI"]:
    match = net.bus[net.bus["name"] == f"BUS_{name}"]
    if match.empty:
        raise ValueError(f"Bus BUS_{name} missing in network.")
    bus_lookup[name] = match.index[0]

net.ext_grid.drop(net.ext_grid.index, inplace=True)
slack_idx = pp.create_ext_grid(
    net,
    bus=bus_lookup["SUDESTE"],
    vm_pu=1.0,
    name="SLACK_SUDESTE"
)

print("  Hub buses:", bus_lookup, "\n")

gen_indices = {}
for subsys, pmax in subsys_gen.items():
    if subsys not in bus_lookup:
        continue
    min_output, max_output = dispatch_bounds.get(subsys, (max(pmax * args.min_gen_frac, 0.0), pmax))
    idx = pp.create_gen(
        net,
        bus=bus_lookup[subsys],
        p_mw=pmax,
        vm_pu=1.0,
        min_p_mw=min_output,
        max_p_mw=max_output,
        name=f"GEN_{subsys}",
        controllable=True,
    )
    gen_indices[subsys] = idx

proxy_indices = {}

# Add loads at hub buses
for subsys, p_mw in subsys_demand_mw.items():
    if subsys == "PARAGUAI":
        continue
    if subsys not in bus_lookup:
        print(f"  ⚠ Subsystem {subsys} missing bus, skipping load.")
        continue
    bus = bus_lookup[subsys]
    pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=0.0, name=f"LOAD_{subsys}")

print(f"  Added {len(net.load)} loads.\n")



# ============================================================
# SET GENERATOR MAX OUTPUTS (CURTAILMENT POSSIBILITY)
# ============================================================

print("🎯 Setting generator limits...")

for subsys, idx in gen_indices.items():
    pmax = subsys_gen.get(subsys, 0.0)
    min_p, max_p = dispatch_bounds.get(subsys, (max(pmax * args.min_gen_frac, 0.0), pmax))
    net.gen.at[idx, "max_p_mw"] = max_p
    net.gen.at[idx, "min_p_mw"] = min_p
    net.gen.at[idx, "p_mw"] = max_p
    net.gen.at[idx, "in_service"] = True
    net.gen.at[idx, "controllable"] = True


print("  Generator limits applied.\n")

total_demand = sum(subsys_demand_mw.values())
total_generation = sum(net.gen["max_p_mw"])

if not args.allow_slack_imports and total_generation + 1e-6 < total_demand:
    deficit = total_demand - total_generation
    raise ValueError(
        f"Perfect-demand mode active but available generation ({total_generation:.2f} MW) "
        f"is lower than demand ({total_demand:.2f} MW). Deficit {deficit:.2f} MW. "
        "Either lower demand scaling or allow slack imports."
    )

if args.allow_slack_imports:
    net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 1e6]
    print("  Slack imports allowed; deficit handled by ext_grid.")
else:
    net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 0.0]
    print("  Perfect-balance mode enabled; slack injection fixed to 0 MW.")
print(f"  Total demand: {total_demand:.2f} MW | Total generation capacity: {total_generation:.2f} MW\n")

print("HUB LOOKUP KEYS:", sorted(bus_lookup.keys()))
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

for idx in net.gen.index:
    pp.create_poly_cost(
        net,
        element=idx,
        et="gen",
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

if "min_q_mvar" not in net.gen.columns:
    net.gen["min_q_mvar"] = 0.0
if "max_q_mvar" not in net.gen.columns:
    net.gen["max_q_mvar"] = 0.0
net.gen["controllable"] = True

if "controllable" not in net.load.columns:
    net.load["controllable"] = False


print("⚙️ Running DC–OPF...")

try:
    pp.rundcopp(net, verbose=False)
    print("  DC–OPF solved successfully.\n")
    if not net.res_line.empty and "max_loading_percent" in net.line.columns:
        binding = net.res_line["loading_percent"] >= net.line["max_loading_percent"] - 1e-3
        if binding.any():
            congested = net.line.loc[binding, ["name", "max_loading_percent"]]
            flow_vals = net.res_line.loc[binding, "loading_percent"]
            print("  ⚠️ Binding lines:")
            for (_, line_row), load in zip(congested.iterrows(), flow_vals):
                print(f"    {line_row['name']}: loading {load:.1f}% of limit {line_row['max_loading_percent']}%")
        else:
            print("  No line hit its loading limit.")
    else:
        print("  Line loading data unavailable.")
except Exception as e:
    print("❌ OPF failed.")
    raise e


# ============================================================
# CURTAILMENT CALCULATION
# ============================================================

print("📉 Curtailment results:")

curtailment_results = []

for idx, row in net.gen.iterrows():
    name = str(row["name"])
    subsys = name.replace("GEN_", "")
    pmax = row["max_p_mw"]
    pdispatch = net.res_gen.at[idx, "p_mw"]
    curtailed = pmax - pdispatch
    curtailment_results.append({
        "subsystem": subsys,
        "name": name,
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
