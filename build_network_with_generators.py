"""
Build a useful Brazil power system object from:
- Transmission dataset (ONS-style, cleaned to English column names)
- Generation + weather dataset (your ML input)

What this script does:
1. Load transmission_lines_clean.csv and build a pandapower network (buses + lines).
2. Load merged_generation_weather.csv and aggregate generation by subsystem for a chosen date.
3. For each subsystem, choose a "hub" bus (most connected bus in that subsystem).
4. Attach one static generator (sgen) per subsystem at the hub bus with P = total generation in that subsystem.
5. Print a summary of the resulting network and injections.

You can later swap TARGET_COL to an ML-predicted column instead of actual gen.
"""

import pandas as pd
import numpy as np
import pandapower as pp
import pandapower.topology as top
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

GEN_PATH = "data/merged_generation_weather.csv"
TX_PATH  = "data/transmission_data.csv"

# Generation dataset columns
DATE_COL       = "date"
GEN_ID_COL     = "ceg"
GEN_SUBSYS_COL = "subsys_name"       # must exist in your generation file
TARGET_COL     = "gen_val(MW)"       # use actual generation; swap to ML prediction if desired

# Transmission dataset columns (English names you gave)
TX_REQUIRED_COLS = [
    "sending_bus_num",
    "receiving_bus_num",
    "line_length_km",
    "pos_seq_resistance",
    "pos_seq_reactance",
    "trans_voltage_kv",
    "subsys_name_from",
    "subsys_name_to",
    "opn_deactivation_date",
    "trans_line_name",
    "long_dur_capacity_lim",
    "long_dur_capacity_no_lim",
    "short_dur_opn_cap_lim",
    "short_dur_opn_cap_no_lim",
]

# Choose scenario date:
# - If SCENARIO_DATE is None: script uses the LAST date available in the generation data.
# - Else: use that date string (e.g. "2019-01-15").
SCENARIO_DATE = None  # or "2019-01-15"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_max_i_ka(row) -> float:
    """Pick a reasonable current limit in kA from available capacity columns."""
    for col in [
        "long_dur_capacity_lim",
        "long_dur_capacity_no_lim",
        "short_dur_opn_cap_lim",
        "short_dur_opn_cap_no_lim",
    ]:
        if col in row and not pd.isna(row[col]):
            return float(row[col]) / 1000.0  # A -> kA
    return 1.0


# ============================================================
# 1. LOAD TRANSMISSION DATA & BUILD NETWORK
# ============================================================

print("🔌 Loading transmission network data...")
df_tx = pd.read_csv(TX_PATH)

missing_cols = [c for c in TX_REQUIRED_COLS if c not in df_tx.columns]
if missing_cols:
    raise ValueError(f"Transmission dataset missing required columns: {missing_cols}")

# Ensure deactivation date is parsed
df_tx["opn_deactivation_date"] = pd.to_datetime(
    df_tx["opn_deactivation_date"], errors="coerce"
)

# Keep only active lines
df_tx_active = df_tx[df_tx["opn_deactivation_date"].isna()].copy()

# Drop lines with critical missing values
df_tx_active = df_tx_active.dropna(
    subset=["sending_bus_num", "receiving_bus_num", "line_length_km",
            "pos_seq_resistance", "pos_seq_reactance", "trans_voltage_kv"]
)

print(f"  Active lines: {len(df_tx_active)}")

# ------------------------------------------------------------
# Build bus -> subsystem mapping from transmission data
# ------------------------------------------------------------
bus_subsys = {}
for _, row in df_tx_active.iterrows():
    bus_subsys[int(row["sending_bus_num"])]   = row["subsys_name_from"]
    bus_subsys[int(row["receiving_bus_num"])] = row["subsys_name_to"]

# ------------------------------------------------------------
# Create pandapower network
# ------------------------------------------------------------
net = pp.create_empty_network()
bus_map = {}  # bus_num -> pp_bus index

def get_bus_voltage(bus_num: int) -> float:
    """Pick a nominal voltage from any line that touches this bus."""
    mask = (df_tx_active["sending_bus_num"] == bus_num) | (df_tx_active["receiving_bus_num"] == bus_num)
    sub  = df_tx_active[mask]
    if sub.empty:
        return 230.0
    return float(sub["trans_voltage_kv"].iloc[0])

def ensure_bus(bus_num: int) -> int:
    """Create bus in pandapower if not already present, and return its index."""
    if bus_num in bus_map:
        return bus_map[bus_num]
    vn_kv = get_bus_voltage(bus_num)
    pp_idx = pp.create_bus(net, vn_kv=vn_kv, name=f"bus_{bus_num}")
    bus_map[bus_num] = pp_idx
    return pp_idx

print("🏗️  Building buses and lines in pandapower...")
for _, row in df_tx_active.iterrows():
    bus_from_num = int(row["sending_bus_num"])
    bus_to_num   = int(row["receiving_bus_num"])

    b_from = ensure_bus(bus_from_num)
    b_to   = ensure_bus(bus_to_num)

    length = float(row["line_length_km"])
    if length <= 0:
        continue

    r_total = float(row["pos_seq_resistance"])
    x_total = float(row["pos_seq_reactance"])

    # Parameters per km
    r_per_km = r_total / length
    x_per_km = x_total / length

    max_i_ka = get_max_i_ka(row)
    line_name = row.get("trans_line_name", f"line_{bus_from_num}_{bus_to_num}")

    pp.create_line_from_parameters(
        net,
        from_bus=b_from,
        to_bus=b_to,
        length_km=length,
        r_ohm_per_km=r_per_km,
        x_ohm_per_km=x_per_km,
        c_nf_per_km=0.0,
        max_i_ka=max_i_ka,
        type="ol",
        name=line_name,
    )

print(f"✅ Network built: {len(net.bus)} buses, {len(net.line)} lines.\n")


# ============================================================
# 2. CHOOSE REPRESENTATIVE HUB BUS PER SUBSYSTEM
# ============================================================

print("📍 Choosing representative hub bus per subsystem...")

# Group buses by subsystem
buses_by_subsys = {}
for bus_num, subsys in bus_subsys.items():
    buses_by_subsys.setdefault(subsys, []).append(bus_num)

# Compute degree (number of incident lines) for each bus
line_incidence = pd.concat(
    [
        df_tx_active[["sending_bus_num"]].rename(columns={"sending_bus_num": "bus_num"}),
        df_tx_active[["receiving_bus_num"]].rename(columns={"receiving_bus_num": "bus_num"}),
    ],
    axis=0,
    ignore_index=True,
)
bus_degree = line_incidence["bus_num"].value_counts().to_dict()

def pick_hub(bus_list):
    """Pick the bus with largest degree from a list."""
    return max(bus_list, key=lambda b: bus_degree.get(b, 0))

representative_bus = {
    subsys: pick_hub(bus_list) for subsys, bus_list in buses_by_subsys.items()
}

print("  Representative buses by subsystem:")
for subsys, bus_num in representative_bus.items():
    print(f"    {subsys}: bus {bus_num} (degree {bus_degree.get(bus_num, 0)})")
print()


# ============================================================
# 3. LOAD GENERATION DATA & AGGREGATE BY SUBSYSTEM
# ============================================================

print("⚡ Loading generation data and aggregating by subsystem...")

df_gen = pd.read_csv(GEN_PATH)
df_gen[DATE_COL] = pd.to_datetime(df_gen[DATE_COL], errors="coerce")

needed_gen_cols = [DATE_COL, GEN_ID_COL, GEN_SUBSYS_COL, TARGET_COL]
missing_gen_cols = [c for c in needed_gen_cols if c not in df_gen.columns]
if missing_gen_cols:
    raise ValueError(f"Generation dataset missing required columns: {missing_gen_cols}")

# Choose scenario date
if SCENARIO_DATE is None:
    scenario_date = df_gen[DATE_COL].max()
else:
    scenario_date = pd.to_datetime(SCENARIO_DATE)

print(f"  Using scenario date: {scenario_date.date()}")

df_gen_scen = df_gen[df_gen[DATE_COL] == scenario_date].copy()
df_gen_scen = df_gen_scen.dropna(subset=[GEN_SUBSYS_COL, TARGET_COL])

if df_gen_scen.empty:
    raise ValueError("No generation data available for the chosen scenario date.")

# Aggregate total generation per subsystem
subsys_gen = (
    df_gen_scen.groupby(GEN_SUBSYS_COL)[TARGET_COL]
    .sum()
    .rename("gen_total_mw")
)

print("  Total generation by subsystem (MW):")
print(subsys_gen)
print()

# ============================================================
# 4. ATTACH SUBSYSTEM GENERATORS TO HUB BUSES
# ============================================================

print("🔗 Attaching one generator per subsystem at its hub bus...")

sgen_records = []
for subsys, total_p_mw in subsys_gen.items():
    if subsys not in representative_bus:
        print(f"  ⚠️ Subsystem {subsys} has generation but no mapped buses; skipping.")
        continue

    hub_bus_num = representative_bus[subsys]
    if hub_bus_num not in bus_map:
        print(f"  ⚠️ Hub bus {hub_bus_num} for subsystem {subsys} not in pandapower net; skipping.")
        continue

    hub_bus_idx = bus_map[hub_bus_num]

    sgen_records.append({
        "bus": hub_bus_idx,
        "p_mw": total_p_mw,
        "q_mvar": 0.0,
        "name": f"GEN_{subsys}",
        "type": "sgen",
        "in_service": True,
    })

df_sgen = pd.DataFrame(sgen_records)

if df_sgen.empty:
    print("  ⚠️ No generators could be mapped to the network.")
else:
    if net.sgen.empty:
        net.sgen = df_sgen
    else:
        net.sgen = pd.concat([net.sgen, df_sgen], ignore_index=True)
    print(f"  ✅ Added {len(df_sgen)} aggregated generators (one per subsystem) to the network.\n")


# ============================================================
# 5. SUMMARY & OPTIONAL CHECKS
# ============================================================

print("📊 FINAL NETWORK SUMMARY")
print("------------------------")
print(net)

print("\nGenerators in net.sgen:")
if net.sgen.empty:
    print("  (none)")
else:
    print(net.sgen[["name", "bus", "p_mw"]])

# Optional: save a snapshot of the pandapower net
OUT_DIR = Path("models/pandapower_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "brazil_network_with_subsys_generation.json"
pp.to_json(net, out_path)
print(f"\n💾 Saved pandapower network with generators to: {out_path}")
