"""
run_dcopp_curtailment.py

Curtailment-focused DC-OPF using:
- Monthly subsystem demand (from demand_projection_clean.csv)
- Monthly-average generation by subsystem and (optionally) plant type (from merged_generation_weather_v2.csv)
- Zonal 5-bus network JSON

Modeling choice (per user):
- PARAGUAI is a SUPPLY node (imports / contracted generation), not a demand node.
  Therefore we DO NOT attach a load at PARAGUAI and we also EXCLUDE it from demand totals.

Key properties:
- Curtailment = available (max_p_mw) - dispatched (res_gen.p_mw)
- Renewables should generally have min_p_mw = 0 for curtailment studies (default here)
- Nuclear can have must-run (default 0.95 of its available)
- Slack/ext_grid can be fixed to 0 (perfect-balance mode) or allowed to import with high cost

Usage examples:
  python run_dcopp_curtailment.py --by-plant-type
  python run_dcopp_curtailment.py --by-plant-type --allow-slack-imports
  python run_dcopp_curtailment.py --year 2020 --month 12 --demand-scaling 1.2 --by-plant-type
"""

import argparse
import calendar
import os

import numpy as np
import pandas as pd
import pandapower as pp


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(description="DC-OPF curtailment on zonal network.")

parser.add_argument("--network-path", default="models/pandapower_snapshots/brazil_network_zonal_5bus_from_tx.json")
parser.add_argument("--demand-path", default="data/demand_data/demand_projection_clean.csv")
parser.add_argument("--generation-path", default="data/merged_generation_weather_v2.csv")

parser.add_argument("--year", type=int, default=2020)
parser.add_argument("--month", type=int, default=12)
parser.add_argument("--demand-scaling", type=float, default=1.0)

parser.add_argument(
    "--allow-slack-imports",
    action="store_true",
    help="If set, ext_grid can supply deficit (p>=0). If not set, ext_grid fixed to 0 MW.",
)

parser.add_argument("--gen-scale-mean", type=float, default=1.0)
parser.add_argument("--gen-scale-std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=None)

parser.add_argument("--results-path", default="results/curtailment_results.csv")

parser.add_argument("--line-loading-percent", type=float, default=100.0)

parser.add_argument(
    "--by-plant-type",
    action="store_true",
    help="If set, create one gen per (subsystem, plant_type). Otherwise one gen per subsystem total.",
)

parser.add_argument(
    "--nuclear-mustrun-frac",
    type=float,
    default=0.95,
    help="Minimum dispatch fraction for nuclear only (default 0.95).",
)

parser.add_argument(
    "--other-mustrun-frac",
    type=float,
    default=0.0,
    help="Minimum dispatch fraction for non-nuclear (default 0.0 for curtailment studies).",
)

args = parser.parse_args()


# ============================================================
# CONSTANTS / NORMALIZATION
# ============================================================

ZONES = ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]

# Demand modeling choice:
# PARAGUAI is supply only (imports), so no load and excluded from demand totals
DEMAND_EXCLUDE_SUBSYSTEMS = {"PARAGUAI"}

DATE_COL = "date"
GEN_SUBSYS_COL = "subsys_name"
PLANT_TYPE_COL = "plant_type"
GEN_VALUE_COL = "gen_val(MW)"

NUCLEAR = "NUCLEAR"

SLACK_COST = 1000.0  # expensive imports
GEN_COST = 1.0       # cheap generation; costs only break ties


def normalize_subsystem(name):
    if name is None:
        return None
    s = str(name).upper().strip()
    s = s.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    s = s.replace("SUDESTE/CENTRO OESTE", "SUDESTE")
    s = s.replace("SOUTHEAST", "SUDESTE")
    return s


def normalize_plant_type(pt):
    if pt is None:
        return None
    return str(pt).upper().strip()


# ============================================================
# DATA HELPERS
# ============================================================

def monthly_demand_peak_mw(path, year, month, scaling):
    df = pd.read_csv(path)

    required = {"subsystem", "year", "month", "MWh"}
    if not required.issubset(df.columns):
        raise ValueError(f"Demand file must contain columns: {required}")

    df["subsystem"] = df["subsystem"].apply(normalize_subsystem)

    # kWh → MWh heuristic
    if df["MWh"].max() > 1e9:
        print("⚠️ Detected kWh-like scale in demand; converting to MWh (divide by 1000).")
        df["MWh"] = df["MWh"] / 1000.0

    df_m = df[(df["year"] == year) & (df["month"] == month)].copy()
    if df_m.empty:
        raise ValueError(f"No demand found for year={year} month={month}.")

    days = calendar.monthrange(year, month)[1]
    hours = days * 24

    subsys_mwh = df_m.groupby("subsystem")["MWh"].sum()
    p_avg = subsys_mwh / hours
    p_peak = p_avg * scaling
    return p_peak.to_dict()


def monthly_generation_avg(path, year, month):
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    mask = (df[DATE_COL].dt.year == year) & (df[DATE_COL].dt.month == month)
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No generation data for {year}-{month:02d}.")

    df["subsystem"] = df[GEN_SUBSYS_COL].apply(normalize_subsystem)
    df["plant_type"] = df[PLANT_TYPE_COL].apply(normalize_plant_type)

    days = pd.Period(year=year, month=month, freq="M").days_in_month
    hours = days * 24

    agg_total = df.groupby("subsystem")[GEN_VALUE_COL].sum() / hours
    agg_types = df.groupby(["subsystem", "plant_type"])[GEN_VALUE_COL].sum() / hours
    return agg_total.to_dict(), agg_types.to_dict()


# ============================================================
# BUILD OPF CASE
# ============================================================

print("🔌 Loading network...")
net = pp.from_json(args.network_path)

# wipe injections/costs for a clean run
for tbl in ["load", "gen", "sgen", "poly_cost", "ext_grid"]:
    df = getattr(net, tbl)
    if not df.empty:
        df.drop(df.index, inplace=True)

# ensure line constraint column exists (important!)
net.line["max_loading_percent"] = float(args.line_loading_percent)

print(f"  Network: {len(net.bus)} buses, {len(net.line)} lines")

# map buses
bus_lookup = {}
for z in ZONES:
    match = net.bus[net.bus["name"] == f"BUS_{z}"]
    if match.empty:
        raise ValueError(f"Bus BUS_{z} missing in network JSON.")
    bus_lookup[z] = int(match.index[0])

# add slack at SUDESTE bus
slack_idx = pp.create_ext_grid(net, bus=bus_lookup["SUDESTE"], vm_pu=1.0, name="SLACK_SUDESTE")


print("📘 Loading demand...")
subsys_demand_mw = monthly_demand_peak_mw(args.demand_path, args.year, args.month, args.demand_scaling)
subsys_demand_mw = {normalize_subsystem(k): float(v) for k, v in subsys_demand_mw.items()}

# keep only zones present in network, and EXCLUDE PARAGUAI by design
subsys_demand_mw = {
    k: v for k, v in subsys_demand_mw.items()
    if (k in bus_lookup) and (k not in DEMAND_EXCLUDE_SUBSYSTEMS)
}

print("  Demand MW (peak proxy) [PARAGUAI excluded]:")
print(subsys_demand_mw)
print()


print("⚡ Loading generation...")
subsys_gen, plant_type_gen = monthly_generation_avg(args.generation_path, args.year, args.month)

# stochastic scaling (applied per subsystem)
rng = np.random.default_rng(args.seed)
if args.gen_scale_std > 0:
    scale_info = {}
    for subsys in list(subsys_gen.keys()):
        factor = max(rng.normal(args.gen_scale_mean, args.gen_scale_std), 0.0)
        subsys_gen[subsys] *= factor
        scale_info[subsys] = factor

        # scale plant types too
        for (s, pt) in list(plant_type_gen.keys()):
            if s == subsys:
                plant_type_gen[(s, pt)] *= factor

    print("  Applied stochastic scaling factors:")
    print(scale_info)
    print()

# restrict to zones
subsys_gen = {k: float(v) for k, v in subsys_gen.items() if k in bus_lookup and np.isfinite(v)}
plant_type_gen = {(s, pt): float(v) for (s, pt), v in plant_type_gen.items()
                  if s in bus_lookup and np.isfinite(v)}

print("  Avg generation (MW) by subsystem:")
print(subsys_gen)
print()


print("🔗 Attaching loads...")
for subsys, p_mw in subsys_demand_mw.items():
    pp.create_load(net, bus=bus_lookup[subsys], p_mw=p_mw, q_mvar=0.0, name=f"LOAD_{subsys}")

print(f"  Added {len(net.load)} loads.\n")


print("🔗 Attaching generators (dispatchable)...")
gen_meta = []  # for result labeling

def add_gen(bus, name, pmax, pmin, cost):
    g = pp.create_gen(
        net,
        bus=bus,
        p_mw=pmax,           # initial
        vm_pu=1.0,
        min_p_mw=pmin,
        max_p_mw=pmax,
        name=name,
        controllable=True,
    )
    pp.create_poly_cost(net, element=g, et="gen", cp1_eur_per_mw=cost, cp0_eur=0.0)
    return g

if args.by_plant_type:
    for (subsys, pt), pmax in plant_type_gen.items():
        if pmax <= 0:
            continue
        bus = bus_lookup[subsys]
        if pt == NUCLEAR:
            pmin = args.nuclear_mustrun_frac * pmax
        else:
            pmin = args.other_mustrun_frac * pmax  # default 0.0 (good for curtailment)
        gidx = add_gen(bus, f"GEN_{subsys}__{pt}", pmax, pmin, GEN_COST)
        gen_meta.append({"gen_idx": gidx, "subsystem": subsys, "plant_type": pt, "pmax": pmax, "pmin": pmin})
else:
    for subsys, pmax in subsys_gen.items():
        if pmax <= 0:
            continue
        bus = bus_lookup[subsys]
        pmin = args.other_mustrun_frac * pmax  # default 0.0
        gidx = add_gen(bus, f"GEN_{subsys}", pmax, pmin, GEN_COST)
        gen_meta.append({"gen_idx": gidx, "subsystem": subsys, "plant_type": "ALL", "pmax": pmax, "pmin": pmin})

print(f"  Added {len(net.gen)} generators.\n")


print("🎯 Configuring slack behavior...")
# poly cost for slack imports.
pp.create_poly_cost(net, element=slack_idx, et="ext_grid", cp1_eur_per_mw=SLACK_COST, cp0_eur=0.0)

if args.allow_slack_imports:
    # allow only imports (supply deficits); no exports
    net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 1e6]
    print("  Slack imports allowed (p in [0, 1e6]).")
else:
    # force slack to 0 so dispatch must match demand using available gen
    net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 0.0]
    print("  Perfect-balance mode (slack fixed to 0 MW).")

total_demand = float(sum(subsys_demand_mw.values()))
total_cap = float(net.gen["max_p_mw"].sum())
print(f"  Total demand (PARAGUAI excluded): {total_demand:.2f} MW | Total available gen: {total_cap:.2f} MW\n")

if (not args.allow_slack_imports) and (total_cap + 1e-6 < total_demand):
    deficit = total_demand - total_cap
    raise ValueError(
        f"Not allowing slack imports, but total available gen < demand. Deficit {deficit:.2f} MW. "
        "Either reduce demand scaling or use --allow-slack-imports."
    )


# Ensure scaling exists (pandapower OPF expects it)
def ensure_scaling(df):
    if df is not None and "scaling" not in df.columns:
        df["scaling"] = 1.0

ensure_scaling(net.load)
ensure_scaling(net.gen)

print("⚙️ Running DC–OPF (rundcopp)...")
pp.rundcopp(net, verbose=False)
print("  DC–OPF solved successfully.\n")


# ============================================================
# RESULTS: CURTAILMENT
# ============================================================

print("📉 Curtailment results:")

rows = []
for m in gen_meta:
    idx = m["gen_idx"]
    pmax = float(net.gen.at[idx, "max_p_mw"])
    pdispatch = float(net.res_gen.at[idx, "p_mw"])
    curtailed = max(pmax - pdispatch, 0.0)
    rows.append({
        "subsystem": m["subsystem"],
        "plant_type": m["plant_type"],
        "name": net.gen.at[idx, "name"],
        "pmax_mw": pmax,
        "dispatch_mw": pdispatch,
        "curtailed_mw": curtailed,
        "curtailment_pct": (curtailed / pmax) if pmax > 0 else 0.0,
    })

df_curta = pd.DataFrame(rows)

print(df_curta.sort_values(["subsystem", "plant_type"]).to_string(index=False))
print("\nCurtailment by subsystem:")
print(df_curta.groupby("subsystem")[["pmax_mw", "dispatch_mw", "curtailed_mw"]].sum().round(3))
print()

# save
os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
df_curta.to_csv(args.results_path, index=False)
print(f"💾 Saved: {args.results_path}")
