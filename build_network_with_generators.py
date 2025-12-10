"""
build_brazil_network_combined.py (UPDATED: amps + optional multiline zonal + gen unit scaling)

Builds TWO pandapower networks:
1) Detailed network from transmission_data.csv (buses + lines) + optional subsystem-aggregated generators.
   - Optional voltage filter for the detailed network (recommended if you do NOT model transformers).
2) 5-bus zonal network (NORTE/NORDESTE/SUDESTE/SUL/PARAGUAI) from ACTIVE inter-zone lines:
   - ZONAL_MODE="multiline": keep one line per physical inter-zone line (parallel lines preserved)
   - ZONAL_MODE="collapse" : collapse each zone-pair into a single equivalent corridor

CRITICAL FIX:
- Capacity columns are in AMPERE per the data dictionary. We parse them as amps (A),
  convert to kA, then to MW using MW ≈ sqrt(3) * V_kV * I_kA (pf ~ 1).

GEN FIX (your recent unit issue):
- If your generator file's gen_val column is actually MWh/day (very likely given your totals),
  set GEN_POWER_SCALE = 1/24 so the network sees MW.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pandapower as pp

print("Imports satisfied")

# ============================================================
# CONFIG
# ============================================================

TX_PATH = "data/transmission_data.csv"
GEN_PATH = "data/merged_generation_weather_v2.csv"  # optional; can be None

DATE_COL       = "date"
GEN_SUBSYS_COL = "subsys_name"
TARGET_COL     = "gen_val(MW)"     # in your file name, but may be MWh/day in reality
SCENARIO_DATE  = None

# If TARGET_COL is actually MWh/day, set this to 1/24. If it's already MW, set to 1.0.
GEN_POWER_SCALE = 1.0 / 24.0 #converts to MW from MWh

# Detailed net voltage filter (only affects detailed build)
DETAILED_VOLTAGE_FILTER_KV = 230  # e.g. 230, 500, or None

# Zonal settings
ZONES = ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]
ZONAL_VN_KV = 500.0            # common zonal voltage base
ZONAL_MODE = "multiline"       # "multiline" or "collapse"
ZONAL_USE_ALL_VOLTAGES = True  # for zonal build, typically True

# Slack/ext_grid behavior (for OPF)
SLACK_SUBSYS_NAME = "SUDESTE"
SLACK_MAX_P_MW = 5000
SLACK_MIN_P_MW = -5000
SLACK_COST = 1000.0

# Subsystem gen cost
SUBSYS_GEN_COST = 1.0
MUST_RUN_FRAC = 0.0

# Output
OUT_DIR = Path("models/pandapower_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DETAILED = OUT_DIR / "brazil_network_detailed_combined.json"
OUT_ZONAL_5BUS = OUT_DIR / "brazil_network_zonal_5bus_from_tx.json"

# ============================================================
# REQUIRED TX COLS (your English-mapped names)
# ============================================================

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

# ============================================================
# HELPERS
# ============================================================

SQRT3 = 1.7320508075688772

def ensure_datetime(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_datetime(df[col], errors="coerce")

def a_to_ka(i_a: float) -> float:
    return float(i_a) / 1000.0

def mw_from_i_ka(i_ka: float, v_kv: float) -> float:
    # MW (≈ MVA) for pf~1
    return SQRT3 * float(v_kv) * float(i_ka)

def i_ka_from_mw(p_mw: float, v_kv: float) -> float:
    v = float(v_kv)
    if v <= 0:
        return 1.0
    return float(p_mw) / (SQRT3 * v)

def pick_capacity_a(row: pd.Series) -> float | None:
    """
    Pick first non-null capacity and interpret as AMPERE (A).
    """
    for col in [
        "long_dur_capacity_lim",
        "long_dur_capacity_no_lim",
        "short_dur_opn_cap_lim",
        "short_dur_opn_cap_no_lim",
    ]:
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return None

def infer_limit_mw_from_capacity(row: pd.Series) -> float:
    """
    Convert AMPERE capacity to MW using the line's nominal voltage.
    fallback: 1 kA if missing (very conservative)
    """
    v_kv = float(row["trans_voltage_kv"])
    cap_a = pick_capacity_a(row)
    if cap_a is None:
        return mw_from_i_ka(1.0, v_kv)
    return mw_from_i_ka(a_to_ka(cap_a), v_kv)

def x_pu_from_ohm(x_ohm: float, v_kv: float, s_base_mva: float) -> float:
    # x_pu = x_ohm * S_base / V^2  (V in kV, S in MVA)
    v = float(v_kv)
    if v <= 0:
        return np.nan
    return float(x_ohm) * float(s_base_mva) / (v * v)

def x_ohm_from_pu(x_pu: float, v_kv: float, s_base_mva: float) -> float:
    v = float(v_kv)
    if v <= 0:
        return np.nan
    return float(x_pu) * (v * v) / float(s_base_mva)

def norm_zone(x: str) -> str:
    return str(x).strip().upper()

def build_bus_subsys_mapping(df_tx_active: pd.DataFrame) -> dict[int, str]:
    bus_subsys = {}
    for _, r in df_tx_active.iterrows():
        bus_subsys[int(r["sending_bus_num"])] = norm_zone(r["subsys_name_from"])
        bus_subsys[int(r["receiving_bus_num"])] = norm_zone(r["subsys_name_to"])
    return bus_subsys

def compute_bus_degree(df_tx_active: pd.DataFrame) -> dict[int, int]:
    incidence = pd.concat(
        [
            df_tx_active[["sending_bus_num"]].rename(columns={"sending_bus_num": "bus_num"}),
            df_tx_active[["receiving_bus_num"]].rename(columns={"receiving_bus_num": "bus_num"}),
        ],
        axis=0,
        ignore_index=True,
    )
    return incidence["bus_num"].value_counts().astype(int).to_dict()

def pick_hub_bus(buses: list[int], degree: dict[int, int]) -> int:
    return max(buses, key=lambda b: degree.get(b, 0))

# ============================================================
# 1) LOAD TX + ACTIVE FILTER
# ============================================================

print("Loading transmission network data...")
df_tx = pd.read_csv(TX_PATH)

missing_cols = [c for c in TX_REQUIRED_COLS if c not in df_tx.columns]
if missing_cols:
    raise ValueError(f"Transmission dataset missing required columns: {missing_cols}")

# Parse dates + numerics defensively
ensure_datetime(df_tx, "opn_deactivation_date")
for c in ["sending_bus_num", "receiving_bus_num"]:
    df_tx[c] = pd.to_numeric(df_tx[c], errors="coerce")
for c in ["line_length_km", "pos_seq_resistance", "pos_seq_reactance", "trans_voltage_kv"]:
    df_tx[c] = pd.to_numeric(df_tx[c], errors="coerce")

df_tx["subsys_name_from"] = df_tx["subsys_name_from"].astype(str).map(norm_zone)
df_tx["subsys_name_to"]   = df_tx["subsys_name_to"].astype(str).map(norm_zone)

df_tx_active_all = df_tx[df_tx["opn_deactivation_date"].isna()].copy()

df_tx_active_all = df_tx_active_all.dropna(
    subset=["sending_bus_num", "receiving_bus_num", "line_length_km",
            "pos_seq_resistance", "pos_seq_reactance", "trans_voltage_kv"]
).copy()

df_tx_active_all = df_tx_active_all[df_tx_active_all["line_length_km"] > 0].copy()
df_tx_active_all["sending_bus_num"] = df_tx_active_all["sending_bus_num"].astype(int)
df_tx_active_all["receiving_bus_num"] = df_tx_active_all["receiving_bus_num"].astype(int)

print(f"Active lines (all voltages): {len(df_tx_active_all)}")

# ============================================================
# 2) BUILD DETAILED NETWORK (OPTIONAL VOLTAGE FILTER)
# ============================================================

df_tx_active_detailed = df_tx_active_all.copy()
if DETAILED_VOLTAGE_FILTER_KV is not None:
    df_tx_active_detailed = df_tx_active_detailed[
        df_tx_active_detailed["trans_voltage_kv"].astype(float) == float(DETAILED_VOLTAGE_FILTER_KV)
    ].copy()
    print(f"Detailed voltage filter: {DETAILED_VOLTAGE_FILTER_KV} kV -> kept {len(df_tx_active_detailed)} lines")

bus_subsys = build_bus_subsys_mapping(df_tx_active_detailed)
bus_degree = compute_bus_degree(df_tx_active_detailed)

buses_by_subsys: dict[str, list[int]] = {}
for bus_num, subsys in bus_subsys.items():
    buses_by_subsys.setdefault(str(subsys), []).append(int(bus_num))

representative_bus = {s: pick_hub_bus(blist, bus_degree) for s, blist in buses_by_subsys.items()}

print("Representative hub buses by subsystem (degree-based, detailed build):")
for s, b in sorted(representative_bus.items()):
    print(f"  {s}: bus {b} (degree {bus_degree.get(b, 0)})")
print()

net = pp.create_empty_network(sn_mva=100)
S_BASE = float(net.sn_mva)

bus_map: dict[int, int] = {}

def ensure_bus(bus_num: int, vn_kv: float) -> int:
    if bus_num in bus_map:
        return bus_map[bus_num]
    idx = pp.create_bus(net, vn_kv=float(vn_kv), name=f"bus_{bus_num}")
    bus_map[bus_num] = idx
    return idx

print("Building buses and lines in pandapower (detailed)...")
for _, row in df_tx_active_detailed.iterrows():
    b_from_num = int(row["sending_bus_num"])
    b_to_num   = int(row["receiving_bus_num"])
    v_kv       = float(row["trans_voltage_kv"])

    b_from = ensure_bus(b_from_num, v_kv)
    b_to   = ensure_bus(b_to_num, v_kv)

    length = float(row["line_length_km"])
    r_total = float(row["pos_seq_resistance"])
    x_total = float(row["pos_seq_reactance"])

    # Skip pathological impedances
    if not np.isfinite(x_total) or x_total <= 0 or not np.isfinite(length) or length <= 0:
        continue

    r_per_km = r_total / length
    x_per_km = x_total / length

    # capacity in AMPERE -> MW limit -> i_ka for pandapower
    limit_mw = infer_limit_mw_from_capacity(row)
    max_i_ka = i_ka_from_mw(limit_mw, v_kv)

    line_name = row.get("trans_line_name", f"line_{b_from_num}_{b_to_num}")
    pp.create_line_from_parameters(
        net,
        from_bus=b_from,
        to_bus=b_to,
        length_km=length,
        r_ohm_per_km=r_per_km,
        x_ohm_per_km=x_per_km,
        c_nf_per_km=0.0,
        max_i_ka=max_i_ka,
        name=str(line_name),
        type="ol",
    )

print(f"Detailed network built: {len(net.bus)} buses, {len(net.line)} lines.\n")

def choose_slack_bus() -> int:
    if SLACK_SUBSYS_NAME in representative_bus:
        hub_bus_num = representative_bus[SLACK_SUBSYS_NAME]
        if hub_bus_num in bus_map:
            return bus_map[hub_bus_num]
    if len(bus_degree) > 0:
        best_bus_num = max(bus_degree.keys(), key=lambda b: bus_degree.get(b, 0))
        if best_bus_num in bus_map:
            return bus_map[best_bus_num]
    return int(net.bus.index[0])

slack_bus_idx = choose_slack_bus()
eg = pp.create_ext_grid(net, bus=slack_bus_idx, vm_pu=1.0, name="GRID_SLACK")
net.ext_grid.loc[eg, "min_p_mw"] = SLACK_MIN_P_MW
net.ext_grid.loc[eg, "max_p_mw"] = SLACK_MAX_P_MW
pp.create_poly_cost(net, eg, "ext_grid", cp1_eur_per_mw=SLACK_COST)

print(f"Added ext_grid slack at bus index {slack_bus_idx} with bounds "
      f"[{SLACK_MIN_P_MW}, {SLACK_MAX_P_MW}] MW and cost {SLACK_COST}.\n")

# OPTIONAL: attach subsystem gens (still aggregated by subsystem)
if GEN_PATH is not None:
    print("Loading generation data and attaching subsystem generators (detailed)...")
    df_gen = pd.read_csv(GEN_PATH)
    if DATE_COL not in df_gen.columns:
        raise ValueError(f"Generation dataset missing required column: {DATE_COL}")
    df_gen[DATE_COL] = pd.to_datetime(df_gen[DATE_COL], errors="coerce")

    for c in [GEN_SUBSYS_COL, TARGET_COL]:
        if c not in df_gen.columns:
            raise ValueError(f"Generation dataset missing required column: {c}")

    scenario_date = df_gen[DATE_COL].max() if SCENARIO_DATE is None else pd.to_datetime(SCENARIO_DATE)
    df_gen_scen = df_gen[df_gen[DATE_COL] == scenario_date].dropna(subset=[GEN_SUBSYS_COL, TARGET_COL]).copy()
    if df_gen_scen.empty:
        raise ValueError(f"No generation rows found for scenario date {scenario_date.date()}")

    # IMPORTANT: apply GEN_POWER_SCALE here
    df_gen_scen[TARGET_COL] = pd.to_numeric(df_gen_scen[TARGET_COL], errors="coerce")
    df_gen_scen = df_gen_scen.dropna(subset=[TARGET_COL]).copy()

    df_gen_scen[GEN_SUBSYS_COL] = df_gen_scen[GEN_SUBSYS_COL].astype(str).map(norm_zone)
    subsys_gen = (df_gen_scen.groupby(GEN_SUBSYS_COL)[TARGET_COL].sum() * float(GEN_POWER_SCALE)).rename("avail_mw")

    print(f"Using scenario date: {scenario_date.date()}")
    print(f"Applied GEN_POWER_SCALE={GEN_POWER_SCALE} to {TARGET_COL} -> MW")
    print(subsys_gen)

    n_added = 0
    for subsys, avail_mw in subsys_gen.items():
        subsys = str(subsys)
        if subsys not in representative_bus:
            continue
        hub_bus_num = representative_bus[subsys]
        if hub_bus_num not in bus_map:
            continue
        hub_bus_idx = bus_map[hub_bus_num]
        avail_mw = float(avail_mw)

        if not np.isfinite(avail_mw) or avail_mw <= 0:
            continue

        g = pp.create_gen(
            net,
            bus=hub_bus_idx,
            p_mw=avail_mw,
            vm_pu=1.0,
            min_p_mw=MUST_RUN_FRAC * avail_mw,
            max_p_mw=avail_mw,
            name=f"GEN_{subsys}",
            controllable=True,
        )
        pp.create_poly_cost(net, g, "gen", cp1_eur_per_mw=SUBSYS_GEN_COST)
        n_added += 1
    print(f"Added {n_added} dispatchable subsystem generators.\n")

pp.to_json(net, OUT_DETAILED)
print(f"Saved detailed network to: {OUT_DETAILED}\n")

# ============================================================
# 3) BUILD ZONAL 5-BUS NETWORK (multiline OR collapse)
# ============================================================

print("Building zonal 5-bus network from ACTIVE inter-zone lines...")

df_z = df_tx_active_all.copy() if ZONAL_USE_ALL_VOLTAGES else df_tx_active_detailed.copy()

# Keep only inter-zone lines where both ends are one of the 5 zones
df_z = df_z[df_z["subsys_name_from"].isin(ZONES) & df_z["subsys_name_to"].isin(ZONES)].copy()
df_z = df_z[df_z["subsys_name_from"] != df_z["subsys_name_to"]].copy()

# per-line quantities
df_z["v_kv_line"] = df_z["trans_voltage_kv"].astype(float)
df_z["x_ohm_total"] = df_z["pos_seq_reactance"].astype(float)

# capacity amps -> MW using the line's own voltage
df_z["limit_mw_line"] = df_z.apply(infer_limit_mw_from_capacity, axis=1)

# convert each line impedance to per-unit on its own voltage, then to ohms on zonal voltage base
df_z["x_pu_line"] = df_z.apply(lambda r: x_pu_from_ohm(r["x_ohm_total"], r["v_kv_line"], S_BASE), axis=1)
df_z["x_ohm_on_zonal"] = df_z["x_pu_line"].apply(lambda xpu: x_ohm_from_pu(xpu, ZONAL_VN_KV, S_BASE))

# convert MW limit to current limit on zonal voltage base (so loading% works on zonal buses)
df_z["max_i_ka_on_zonal"] = df_z["limit_mw_line"].apply(lambda mw: i_ka_from_mw(mw, ZONAL_VN_KV))

# undirected corridor key
df_z["a"] = df_z[["subsys_name_from", "subsys_name_to"]].min(axis=1)
df_z["b"] = df_z[["subsys_name_from", "subsys_name_to"]].max(axis=1)

net5 = pp.create_empty_network(sn_mva=S_BASE)
b5 = {z: pp.create_bus(net5, name=f"BUS_{z}", vn_kv=float(ZONAL_VN_KV)) for z in ZONES}

eg5 = pp.create_ext_grid(net5, bus=b5.get("SUDESTE", list(b5.values())[0]), vm_pu=1.0, name="GRID_SUDESTE")
net5.ext_grid.loc[eg5, "min_p_mw"] = SLACK_MIN_P_MW
net5.ext_grid.loc[eg5, "max_p_mw"] = SLACK_MAX_P_MW
pp.create_poly_cost(net5, eg5, "ext_grid", cp1_eur_per_mw=SLACK_COST)

if ZONAL_MODE.lower() == "multiline":
    # One pandapower line per physical inter-zone line (parallel lines preserved)
    created = 0
    for i, r in df_z.iterrows():
        fr = str(r["subsys_name_from"])
        to = str(r["subsys_name_to"])

        x_total_ohm = float(r["x_ohm_on_zonal"])
        max_i_ka = float(r["max_i_ka_on_zonal"])

        if (not np.isfinite(x_total_ohm)) or x_total_ohm <= 0:
            continue
        if (not np.isfinite(max_i_ka)) or max_i_ka <= 0:
            continue

        name = str(r.get("trans_line_name", f"TX_{i}"))
        pp.create_line_from_parameters(
            net5,
            from_bus=b5[fr],
            to_bus=b5[to],
            length_km=1.0,            # pack total impedance into 1 km
            r_ohm_per_km=0.0,
            x_ohm_per_km=x_total_ohm, # total x in ohms on zonal base
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=name,
            type="ol",
        )
        created += 1
    print(f"Zonal multiline mode: created {created} inter-zone lines (parallel preserved).")

elif ZONAL_MODE.lower() == "collapse":
    # collapse to one equivalent corridor per zone-pair
    def agg_parallel(group: pd.DataFrame) -> pd.Series:
        limit_mw = float(group["limit_mw_line"].sum())

        xs = group["x_pu_line"].values
        xs = xs[np.isfinite(xs) & (xs > 0)]
        if len(xs) == 0:
            x_eq_pu = 10.0
        else:
            b_eq = np.sum(1.0 / xs)
            x_eq_pu = float(1.0 / b_eq) if b_eq > 0 else 10.0

        volts = sorted(group["v_kv_line"].dropna().astype(float).unique().tolist())
        return pd.Series({
            "limit_mw": limit_mw,
            "x_eq_pu": x_eq_pu,
            "n_lines": int(len(group)),
            "voltages_kv": ",".join(str(int(v)) for v in volts) if volts else "",
        })

    corr = df_z.groupby(["a", "b"]).apply(agg_parallel).reset_index()
    print("\n=== ZONAL CORRIDORS (collapsed) ===")
    print(corr.sort_values("limit_mw", ascending=False).to_string(index=False))
    print("==================================\n")

    for _, r in corr.iterrows():
        fr = str(r["a"])
        to = str(r["b"])
        limit_mw = float(r["limit_mw"])
        x_eq_pu = float(r["x_eq_pu"])

        x_eq_ohm = x_ohm_from_pu(x_eq_pu, ZONAL_VN_KV, S_BASE)
        max_i_ka = i_ka_from_mw(limit_mw, ZONAL_VN_KV)

        pp.create_line_from_parameters(
            net5,
            from_bus=b5[fr],
            to_bus=b5[to],
            length_km=1.0,
            r_ohm_per_km=0.0,
            x_ohm_per_km=x_eq_ohm,
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=f"CORRIDOR_{fr}_{to}",
            type="ol",
        )
else:
    raise ValueError(f"Unknown ZONAL_MODE={ZONAL_MODE}. Use 'multiline' or 'collapse'.")

pp.to_json(net5, OUT_ZONAL_5BUS)
print(f"Saved zonal 5-bus network to: {OUT_ZONAL_5BUS}\n")

print("Done.")
print(f"  Detailed: {OUT_DETAILED}")
print(f"  Zonal 5-bus: {OUT_ZONAL_5BUS}")
print(f"  GEN_POWER_SCALE used for detailed gen attachment: {GEN_POWER_SCALE} (set to 1.0 if already MW)")
