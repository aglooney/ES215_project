#!/usr/bin/env python3
"""
run_forecast_curtailment_sim.py

Curtailment + ENS + DUMP simulation on a 5-bus zonal Brazil network.

GEN INPUT (historical merged file columns supported):
ceg,gen_latitude,gen_longitude,date,gen_val(MW),subsys_name,plant_type,fuel_type,time,...

Forecast gen file supported:
year,month,subsystem,predicted_avg_mw   (optional fuel_bucket)

Demand input supported:
- Must have: subsystem, year, month, and either demand_mw OR MWh (monthly energy)
- If it has 'scenario', you MUST select one scenario (we default to 'referencia').

Key idea for units:
- Historical gen_val(MW) is a MW time series (hourly/sub-hourly). To avoid the 744x blow-up,
  we collapse to subsystem totals per timestamp, then take MEAN MW over sampled day/month.
- Forecast predicted_avg_mw is already monthly average MW.

Stochasticity:
- Historical mode: per trial+month, sample a random DAY and use that day's mean MW (per subsystem/bucket).
- Forecast mode: no day structure, so we apply per-trial multiplicative noise to generation.
"""

import argparse
import copy
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.auxiliary import OPFNotConverged
from tqdm.auto import tqdm


# -----------------------------
# Zones / conventions
# -----------------------------
ZONES = ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]
DEMAND_EXCLUDE_SUBSYSTEMS = {"PARAGUAI"}  # your convention: no load there

# Must-run constraints requested
DEFAULT_NUCLEAR_MIN_FRAC = 0.90
DEFAULT_NORTE_HYDRO_MIN_FRAC = 0.50

# Shares (fallback split if no type info)
DEFAULT_SUDESTE_NUCLEAR_SHARE = 0.10
DEFAULT_NORTE_HYDRO_SHARE = 0.60

# Costs
DEFAULT_COSTS = {
    "solar": 1.0,
    "wind": 2.0,
    "hydro": 5.0,
    "nuclear": 8.0,
    "thermal": 80.0,
}
DEFAULT_SLACK_COST = 1000.0
DEFAULT_DUMP_COST = 300.0
DEFAULT_ENS_COST = 5000.0

# Numeric tolerances
EPS = 1e-6
BINDING_TOL_PCT = 1e-3


# -----------------------------
# Helpers: normalization + typing
# -----------------------------
def normalize_subsystem(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    mapping = {
        "NORTHE": "NORTE",
        "NORT": "NORTE",
        "NORD": "NORDESTE",
        "NORTHEAST": "NORDESTE",
        "SOUTHEAST": "SUDESTE",
        "SOUTH": "SUL",
        "PARAGUAY": "PARAGUAI",
        "SUDESTE/CENTRO-OESTE": "SUDESTE",
        "SE/CO": "SUDESTE",
    }
    s = mapping.get(s, s)
    return s if s in ZONES else None


def strip_accents(s: str) -> str:
    s = str(s or "")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def classify_fuel_bucket(plant_type: str, fuel_type: str) -> str:
    pt = strip_accents(plant_type).upper()
    ft = strip_accents(fuel_type).upper()
    s = (pt + " " + ft).strip()

    if "NUCLEAR" in s:
        return "nuclear"
    if "SOLAR" in s or "FOTOV" in s or "FOTOVOL" in s:
        return "solar"
    if "EOLIC" in s or "EOL" in s or "WIND" in s:
        return "wind"
    if "HIDRO" in s or "HYDR" in s:
        return "hydro"
    return "thermal"


def fuel_cost(bucket: str, cost_map: Dict[str, float]) -> float:
    return float(cost_map.get(bucket, cost_map.get("thermal", 80.0)))


def ensure_scaling(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    if "scaling" not in df.columns:
        df["scaling"] = 1.0
    df["scaling"] = df["scaling"].fillna(1.0).astype(float)


def ensure_controllable_col(df: pd.DataFrame, default: bool = False) -> None:
    if df is None or df.empty:
        return
    if "controllable" not in df.columns:
        df["controllable"] = bool(default)
    df["controllable"] = df["controllable"].fillna(bool(default)).astype(bool)


def compute_binding_lines(net: pp.pandapowerNet) -> Tuple[int, str, float]:
    if net.line.empty or not hasattr(net, "res_line") or net.res_line.empty:
        return 0, "", 0.0

    max_allowed = net.line.get(
        "max_loading_percent",
        pd.Series([100.0] * len(net.line), index=net.line.index),
    )
    loading = net.res_line.get(
        "loading_percent",
        pd.Series([0.0] * len(net.line), index=net.line.index),
    )

    binding_mask = loading >= (max_allowed - BINDING_TOL_PCT)
    if int(binding_mask.sum()) == 0:
        return 0, "", float(loading.max()) if len(loading) else 0.0

    names = net.line.loc[binding_mask, "name"].astype(str).tolist()
    names = [n.strip() for n in names if n and str(n).strip()]
    return int(binding_mask.sum()), ";".join(names), float(loading.max())


def build_bus_lookup_from_net(net: pp.pandapowerNet) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx, row in net.bus.iterrows():
        name = str(row.get("name", "")).strip().upper()
        for z in ZONES:
            if name == f"BUS_{z}" or name.endswith(z):
                lookup[z] = int(idx)
    if len(lookup) < len(ZONES):
        for idx, row in net.bus.iterrows():
            name = str(row.get("name", "")).strip().upper()
            for z in ZONES:
                if z in name and z not in lookup:
                    lookup[z] = int(idx)
    return lookup


# -----------------------------
# Input loaders
# -----------------------------
def load_demand(demand_path: Path) -> pd.DataFrame:
    """
    Returns columns (at least): subsystem, year, month, demand_mw (+ scenario if present)

    IMPORTANT: This loader AGGREGATES:
      - state-level rows -> subsystem totals
      - duplicates -> summed
    """
    import calendar

    df = pd.read_csv(demand_path)

    if "subsystem" not in df.columns:
        raise ValueError(f"Demand file must have 'subsystem'. Columns: {list(df.columns)}")

    df["subsystem"] = (
        df["subsystem"].astype(str).str.upper()
        .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE", regex=False)
    )

    for c in ["year", "month"]:
        if c not in df.columns:
            raise ValueError(f"Demand file must have '{c}'. Columns: {list(df.columns)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["year", "month"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df = df[df["month"].between(1, 12)].copy()

    if "demand_mw" in df.columns:
        df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
        df = df.dropna(subset=["demand_mw"]).copy()
    elif "MWh" in df.columns:
        df["MWh"] = pd.to_numeric(df["MWh"], errors="coerce")
        df = df.dropna(subset=["MWh"]).copy()

        def hours_in_month(y, m):
            return calendar.monthrange(int(y), int(m))[1] * 24

        df["hours"] = [hours_in_month(y, m) for y, m in zip(df["year"], df["month"])]
        df["demand_mw"] = df["MWh"] / df["hours"]
    else:
        raise ValueError(
            f"Demand file needs demand_mw (MW) or MWh (monthly energy). Columns: {list(df.columns)}"
        )

    # Aggregate to subsystem totals (and scenario totals if scenario exists)
    group_cols = ["subsystem", "year", "month"]
    if "scenario" in df.columns:
        group_cols = ["scenario"] + group_cols

    df["subsystem"] = df["subsystem"].map(normalize_subsystem)
    df = df.dropna(subset=["subsystem"]).copy()
    df = df[df["subsystem"].isin(ZONES)].copy()

    df = df.groupby(group_cols, as_index=False)["demand_mw"].sum()
    return df


def load_generation(path: Path) -> pd.DataFrame:
    """
    Accepts either:
      - historical merged file with (date or time) + gen_val(MW) + subsys_name (+ plant_type/fuel_type)
      - forecast file with: year, month, subsystem, predicted_avg_mw (+ optional fuel_bucket)

    Returns:
      - forecast: year, month, subsystem, avail_mw, fuel_bucket
      - historical: timestamp, year, month, subsystem, avail_mw, fuel_bucket
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # --- Case 1: forecast file ---
    if "predicted_avg_mw" in cols:
        y = cols.get("year")
        m = cols.get("month")
        s = cols.get("subsystem")
        if not (y and m and s):
            raise ValueError(
                f"Forecast file must have year, month, subsystem, predicted_avg_mw. Columns: {list(df.columns)}"
            )

        out = df[[y, m, s, cols["predicted_avg_mw"]]].copy()
        out = out.rename(columns={y: "year", m: "month", s: "subsystem", cols["predicted_avg_mw"]: "avail_mw"})
        out["subsystem"] = out["subsystem"].map(normalize_subsystem)
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out["month"] = pd.to_numeric(out["month"], errors="coerce")
        out["avail_mw"] = pd.to_numeric(out["avail_mw"], errors="coerce")
        out = out.dropna(subset=["year", "month", "subsystem", "avail_mw"]).copy()
        out["year"] = out["year"].astype(int)
        out["month"] = out["month"].astype(int)
        out = out[out["month"].between(1, 12)]
        out = out[out["subsystem"].isin(ZONES)]
        # forecast usually has no type info
        out["fuel_bucket"] = "thermal"

        # Guardrail: if values look like monthly MW-day totals (very large), normalize by days.
        median_val = float(out["avail_mw"].median()) if not out.empty else 0.0
        if median_val > 2e5:  # >200 GW as avg MW is implausible
            import calendar

            def divide_by_days(row):
                days = calendar.monthrange(int(row["year"]), int(row["month"]))[1]
                return float(row["avail_mw"]) / float(days if days > 0 else 1)

            out["avail_mw"] = out.apply(divide_by_days, axis=1)
            print("Detected large forecast generation values; dividing by days in month (assumed MW-day totals).")
        return out

    # --- Case 2: historical merged file ---
    # Prefer 'time' if present (often includes hour); fall back to 'date'
    time_col = cols.get("time")
    date_col = cols.get("date") or cols.get("data") or cols.get("dt") or cols.get("timestamp")

    ts_col = None
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if df[time_col].notna().any():
            ts_col = time_col
    if ts_col is None:
        if date_col is None:
            raise ValueError(
                f"Generation file must have predicted_avg_mw OR a date/time column. Columns: {list(df.columns)}"
            )
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        ts_col = date_col

    df = df.dropna(subset=[ts_col]).copy()
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    subsys_col = cols.get("subsys_name") or cols.get("subsystem") or cols.get("subsys") or cols.get("zone")
    mw_col = cols.get("gen_val(mw)") or cols.get("avail_mw") or cols.get("mw")

    if subsys_col is None or mw_col is None:
        raise ValueError(
            f"Historical gen file missing subsys_name/subsystem and gen_val(MW). Columns: {list(df.columns)}"
        )

    df["subsystem"] = df[subsys_col].map(normalize_subsystem)
    df["avail_mw"] = pd.to_numeric(df[mw_col], errors="coerce")

    # If timestamps are daily (no hour info), treat gen_val as daily energy (MWh) and convert to avg MW.
    ts_norm = df["timestamp"].dt.normalize()
    if (ts_norm == df["timestamp"]).all():
        df["avail_mw"] = df["avail_mw"] / 24.0
        print("Detected daily timestamps in historical gen; treating values as MWh/day and dividing by 24 to get MW.")

    # types -> bucket
    pt = df[cols["plant_type"]] if "plant_type" in cols else ""
    ft = df[cols["fuel_type"]] if "fuel_type" in cols else ""
    df["fuel_bucket"] = [classify_fuel_bucket(p, f) for p, f in zip(pt, ft)]

    df = df.dropna(subset=["year", "month", "subsystem", "avail_mw", "timestamp"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df = df[df["month"].between(1, 12)]
    df = df[df["subsystem"].isin(ZONES)]
    df["fuel_bucket"] = df["fuel_bucket"].fillna("thermal").astype(str).str.lower()
    return df


def aggregate_generation_by_bucket(
    gsub_df: pd.DataFrame,
    subsys_gen: Dict[str, float],
    sudeste_nuclear_share: float,
    norte_hydro_share: float,
) -> Dict[Tuple[str, str], float]:
    """
    Return {(subsystem, fuel_bucket): avail_mw}.
    If gsub_df has fuel_bucket, we use it. Otherwise split subsystem totals by shares.
    """
    gen_map: Dict[Tuple[str, str], float] = {}

    if gsub_df is not None and not gsub_df.empty and "fuel_bucket" in gsub_df.columns:
        tmp = gsub_df.copy()
        tmp["fuel_bucket"] = tmp["fuel_bucket"].fillna("thermal").astype(str).str.lower()
        grouped = tmp.groupby(["subsystem", "fuel_bucket"], as_index=False)["avail_mw"].sum()
        for r in grouped.itertuples(index=False):
            subsys = normalize_subsystem(r.subsystem)
            bucket = str(r.fuel_bucket).lower()
            if subsys is None:
                continue
            gen_map[(subsys, bucket)] = gen_map.get((subsys, bucket), 0.0) + float(r.avail_mw)
        if gen_map:
            return gen_map

    # fallback split
    def clamp_share(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    sud_share = clamp_share(sudeste_nuclear_share)
    nor_share = clamp_share(norte_hydro_share)

    for subsys, total in subsys_gen.items():
        subsys_n = normalize_subsystem(subsys)
        if subsys_n is None:
            continue
        total = float(total)
        if total <= 0:
            continue

        if subsys_n == "SUDESTE":
            nuc = sud_share * total
            rem = total - nuc
            if nuc > 0:
                gen_map[(subsys_n, "nuclear")] = gen_map.get((subsys_n, "nuclear"), 0.0) + nuc
            if rem > 0:
                gen_map[(subsys_n, "thermal")] = gen_map.get((subsys_n, "thermal"), 0.0) + rem
        elif subsys_n == "NORTE":
            hydro = nor_share * total
            rem = total - hydro
            if hydro > 0:
                gen_map[(subsys_n, "hydro")] = gen_map.get((subsys_n, "hydro"), 0.0) + hydro
            if rem > 0:
                gen_map[(subsys_n, "thermal")] = gen_map.get((subsys_n, "thermal"), 0.0) + rem
        else:
            gen_map[(subsys_n, "thermal")] = gen_map.get((subsys_n, "thermal"), 0.0) + total

    return gen_map


def pick_trial_day(ts_df: pd.DataFrame, year: int, month: int, rng: np.random.Generator):
    """
    ts_df must have: timestamp, year, month.
    Returns a numpy datetime64 day (normalized).
    """
    days = ts_df.loc[
        (ts_df["year"] == year) & (ts_df["month"] == month),
        "timestamp"
    ].dropna().dt.normalize().unique()

    if len(days) == 0:
        return None
    return rng.choice(days)


# -----------------------------
# Core OPF for one month/trial draw
# -----------------------------
def run_single_month(
    base_net: pp.pandapowerNet,
    bus_lookup: Dict[str, int],
    subsys_demand: Dict[str, float],
    gen_by_subsys_bucket: Dict[Tuple[str, str], float],
    cost_map: Dict[str, float],
    slack_cost: float,
    ens_cost: float,
    dump_cost: float,
    nuclear_min_frac: float,
    norte_hydro_min_frac: float,
    allow_slack: bool,
    line_loading_percent: float,
) -> pd.DataFrame:
    net = copy.deepcopy(base_net)

    for table_name in ("load", "gen", "sgen", "poly_cost"):
        table = getattr(net, table_name)
        if not table.empty:
            table.drop(table.index, inplace=True)

    if not net.line.empty:
        net.line["max_loading_percent"] = float(line_loading_percent)

    # ext_grid / slack
    if net.ext_grid.empty:
        slack_bus = bus_lookup.get("SUDESTE", list(bus_lookup.values())[0])
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, name="GRID_SLACK")
    slack_idx = int(net.ext_grid.index[0])

    if allow_slack:
        net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, -1e6, 1e6]
    else:
        net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 0.0]

    pp.create_poly_cost(net, element=slack_idx, et="ext_grid", cp1_eur_per_mw=float(slack_cost), cp0_eur=0.0)

    # fixed demand loads
    for subsys, value in subsys_demand.items():
        subsys = normalize_subsystem(subsys)
        if subsys is None or subsys in DEMAND_EXCLUDE_SUBSYSTEMS:
            continue
        if subsys not in bus_lookup:
            continue
        value = float(value)
        if value <= 0:
            continue
        pp.create_load(net, bus=bus_lookup[subsys], p_mw=value, q_mvar=0.0, name=f"LOAD_{subsys}")

    ensure_scaling(net.load)
    ensure_controllable_col(net.load, default=False)
    net.load["controllable"] = net.load["controllable"].fillna(False).astype(bool)

    # real generators by (subsystem, bucket)
    real_gen_indices: List[int] = []

    def add_real_gen(bus: int, name: str, pmax: float, cp1: float, pmin: float = 0.0) -> Optional[int]:
        pmax = float(max(pmax, 0.0))
        pmin = float(max(min(pmin, pmax), 0.0))
        if pmax <= 0:
            return None
        g = pp.create_gen(
            net,
            bus=int(bus),
            p_mw=pmax,
            vm_pu=1.0,
            min_p_mw=pmin,
            max_p_mw=pmax,
            name=name,
            controllable=True,
        )
        pp.create_poly_cost(net, element=g, et="gen", cp1_eur_per_mw=float(cp1), cp0_eur=0.0)
        real_gen_indices.append(int(g))
        return int(g)

    for (subsys, bucket), pmax in gen_by_subsys_bucket.items():
        subsys_n = normalize_subsystem(subsys)
        bucket = str(bucket).lower()
        if subsys_n is None or subsys_n not in bus_lookup:
            continue
        pmax = float(pmax)
        if pmax <= 0:
            continue

        cp1 = fuel_cost(bucket, cost_map)

        pmin = 0.0
        if subsys_n == "SUDESTE" and bucket == "nuclear":
            pmin = float(nuclear_min_frac) * pmax
        if subsys_n == "NORTE" and bucket == "hydro":
            pmin = float(norte_hydro_min_frac) * pmax

        add_real_gen(
            bus=bus_lookup[subsys_n],
            name=f"GEN_{subsys_n}_{bucket.upper()}",
            pmax=pmax,
            cp1=cp1,
            pmin=pmin,
        )

    ensure_scaling(net.gen)
    net.gen["controllable"] = True

    # ENS at load buses
    ens_gen_indices: List[int] = []
    for _, row in net.load.iterrows():
        nm = str(row.get("name", ""))
        if not nm.startswith("LOAD_"):
            continue
        bus = int(row["bus"])
        load_mw = float(row["p_mw"])
        if load_mw <= 0:
            continue
        g = pp.create_gen(
            net,
            bus=bus,
            p_mw=0.0,
            vm_pu=1.0,
            min_p_mw=0.0,
            max_p_mw=load_mw,
            name=f"ENS_AT_BUS{bus}",
            controllable=True,
        )
        pp.create_poly_cost(net, element=g, et="gen", cp1_eur_per_mw=float(ens_cost), cp0_eur=0.0)
        ens_gen_indices.append(int(g))

    # DUMP controllable loads (sink)
    dump_load_indices: List[int] = []
    for subsys, bus in bus_lookup.items():
        dl = pp.create_load(
            net,
            bus=int(bus),
            p_mw=0.0,
            q_mvar=0.0,
            name=f"DUMP_{subsys}",
            controllable=True,
            min_p_mw=0.0,
            max_p_mw=1e6,
        )
        pp.create_poly_cost(net, element=dl, et="load", cp1_eur_per_mw=float(dump_cost), cp0_eur=0.0)
        dump_load_indices.append(int(dl))

    ensure_controllable_col(net.load, default=False)
    net.load.loc[dump_load_indices, "controllable"] = True

    # solve
    opf_converged = True
    try:
        pp.rundcopp(net, verbose=False)
    except OPFNotConverged:
        opf_converged = False

    total_demand = float(
        net.load.loc[net.load["name"].astype(str).str.startswith("LOAD_"), "p_mw"].sum()
    ) if not net.load.empty else 0.0

    total_available_gen = 0.0
    if not net.gen.empty:
        for _, row in net.gen.iterrows():
            nm = str(row.get("name", ""))
            if nm.startswith("ENS_") or nm.startswith("ENS_AT_"):
                continue
            total_available_gen += float(row.get("max_p_mw", 0.0))

    slack_import_mw = 0.0
    if opf_converged and hasattr(net, "res_ext_grid") and not net.res_ext_grid.empty:
        slack_import_mw = float(net.res_ext_grid.at[slack_idx, "p_mw"])

    ens_mw_total = 0.0
    if opf_converged and ens_gen_indices and hasattr(net, "res_gen") and not net.res_gen.empty:
        ens_mw_total = float(net.res_gen.loc[ens_gen_indices, "p_mw"].sum())

    dump_mw_total = 0.0
    dump_by_subsys = {z: 0.0 for z in ZONES}
    if opf_converged and dump_load_indices and hasattr(net, "res_load") and not net.res_load.empty:
        bus_to_subsys = {v: k for k, v in bus_lookup.items()}
        for li in dump_load_indices:
            p = float(net.res_load.at[li, "p_mw"])
            dump_mw_total += p
            subsys = bus_to_subsys.get(int(net.load.at[li, "bus"]), None)
            if subsys is not None:
                dump_by_subsys[subsys] += p

    binding_count, binding_names, binding_loading_max = (0, "", 0.0)
    if opf_converged:
        binding_count, binding_names, binding_loading_max = compute_binding_lines(net)

    ens_by_subsys = {z: 0.0 for z in ZONES}
    if opf_converged and ens_gen_indices and hasattr(net, "res_gen") and not net.res_gen.empty:
        bus_to_subsys = {v: k for k, v in bus_lookup.items()}
        for g_idx in ens_gen_indices:
            bus = int(net.gen.at[g_idx, "bus"])
            subsys = bus_to_subsys.get(bus, None)
            if subsys is not None:
                ens_by_subsys[subsys] += float(net.res_gen.at[g_idx, "p_mw"])

    rows = []
    if opf_converged and hasattr(net, "res_gen") and not net.res_gen.empty:
        for gi, row in net.gen.iterrows():
            nm = str(row.get("name", ""))
            if nm.startswith("ENS_") or nm.startswith("ENS_AT_"):
                continue
            if not nm.startswith("GEN_"):
                continue

            pmax = float(row["max_p_mw"])
            dispatch = float(net.res_gen.at[gi, "p_mw"])
            curtailed = max(pmax - dispatch, 0.0)

            parts = nm.split("_")
            subsys = normalize_subsystem(parts[1]) if len(parts) >= 2 else None
            if subsys is None:
                continue

            rows.append({
                "subsystem": subsys,
                "pmax_mw": pmax,
                "dispatch_mw": dispatch,
                "curtailed_mw": curtailed,
                "total_demand_mw": total_demand,
                "total_available_gen_mw": total_available_gen,
                "slack_import_mw": slack_import_mw,
                "ens_mw": ens_mw_total,
                "ens_mw_subsystem": float(ens_by_subsys.get(subsys, 0.0)),
                "dump_mw": float(dump_mw_total),
                "dump_mw_subsystem": float(dump_by_subsys.get(subsys, 0.0)),
                "slack_used": bool(allow_slack and slack_import_mw > EPS),
                "binding_lines_count": float(binding_count),
                "binding_line_names": binding_names,
                "binding_line_loading_pct_max": float(binding_loading_max),
                "opf_converged": True,
            })

        out = pd.DataFrame(rows)
        if not out.empty:
            sys_cols = [c for c in out.columns if c not in ["subsystem", "pmax_mw", "dispatch_mw", "curtailed_mw"]]
            agg = out.groupby("subsystem", as_index=False).agg(
                pmax_mw=("pmax_mw", "sum"),
                dispatch_mw=("dispatch_mw", "sum"),
                curtailed_mw=("curtailed_mw", "sum"),
                **{c: (c, "max") for c in sys_cols},
            )
            return agg

    # failure output
    subsystems_present = sorted({normalize_subsystem(k) for (k, _) in gen_by_subsys_bucket.keys()} | {normalize_subsystem(k) for k in subsys_demand.keys()})
    subsystems_present = [s for s in subsystems_present if s is not None]

    fail_rows = []
    for s in subsystems_present:
        fail_rows.append({
            "subsystem": s,
            "pmax_mw": np.nan,
            "dispatch_mw": np.nan,
            "curtailed_mw": np.nan,
            "total_demand_mw": total_demand,
            "total_available_gen_mw": np.nan,
            "slack_import_mw": np.nan,
            "ens_mw": np.nan,
            "ens_mw_subsystem": np.nan,
            "dump_mw": np.nan,
            "dump_mw_subsystem": np.nan,
            "slack_used": False,
            "binding_lines_count": np.nan,
            "binding_line_names": "",
            "binding_line_loading_pct_max": np.nan,
            "opf_converged": False,
        })
    return pd.DataFrame(fail_rows)


# -----------------------------
# Main simulation
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net-json", default="models/pandapower_snapshots/brazil_network_zonal_5bus_from_tx.json")
    parser.add_argument("--demand", default="data/demand_data/demand_projection_2025_2028.csv")
    parser.add_argument("--gen", default="results/generation_forecast_2025_2028.csv")

    parser.add_argument("--out", default="results/curtailment_simulations.csv")
    parser.add_argument("--opf-failures-out", default="results/curtailment_simulations_opf_failures.csv")

    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--allow-slack-imports", action="store_true")
    parser.add_argument("--line-loading-percent", type=float, default=100.0)

    # stochasticity controls
    parser.add_argument("--day-sampling", type=str, default="random", choices=["random", "monthly_mean"])
    parser.add_argument("--forecast-gen-noise-std", type=float, default=0.10,
                        help="Only used for forecast gen files: per-trial multiplicative noise on avail_mw.")

    # shares (fallback split)
    parser.add_argument("--sudeste-nuclear-share", type=float, default=DEFAULT_SUDESTE_NUCLEAR_SHARE)
    parser.add_argument("--norte-hydro-share", type=float, default=DEFAULT_NORTE_HYDRO_SHARE)

    # must-run fractions
    parser.add_argument("--nuclear-min-frac", type=float, default=DEFAULT_NUCLEAR_MIN_FRAC)
    parser.add_argument("--norte-hydro-min-frac", type=float, default=DEFAULT_NORTE_HYDRO_MIN_FRAC)

    # demand scenario selection (IMPORTANT if demand file includes 3 scenarios)
    parser.add_argument("--demand-scenario", type=str, default="referencia",
                        help="If demand file has 'scenario', filter to this scenario.")

    # costs
    parser.add_argument("--cost-solar", type=float, default=DEFAULT_COSTS["solar"])
    parser.add_argument("--cost-wind", type=float, default=DEFAULT_COSTS["wind"])
    parser.add_argument("--cost-hydro", type=float, default=DEFAULT_COSTS["hydro"])
    parser.add_argument("--cost-nuclear", type=float, default=DEFAULT_COSTS["nuclear"])
    parser.add_argument("--cost-thermal", type=float, default=DEFAULT_COSTS["thermal"])
    parser.add_argument("--slack-cost", type=float, default=DEFAULT_SLACK_COST)
    parser.add_argument("--dump-cost", type=float, default=DEFAULT_DUMP_COST)
    parser.add_argument("--ens-cost", type=float, default=DEFAULT_ENS_COST)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    net_path = Path(args.net_json)
    demand_path = Path(args.demand)
    gen_path = Path(args.gen)
    for p in [net_path, demand_path, gen_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = Path(args.opf_failures_out)
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    cost_map = {
        "solar": float(args.cost_solar),
        "wind": float(args.cost_wind),
        "hydro": float(args.cost_hydro),
        "nuclear": float(args.cost_nuclear),
        "thermal": float(args.cost_thermal),
    }

    print("Loading network...")
    base_net = pp.from_json(str(net_path))
    bus_lookup = build_bus_lookup_from_net(base_net)
    if not bus_lookup or len(bus_lookup) < 4:
        raise RuntimeError(f"Could not build bus lookup from net.bus. Got: {bus_lookup}")
    print("Bus lookup:", bus_lookup)

    print("Loading demand...")
    df_demand = load_demand(demand_path)

    # IMPORTANT: scenario filtering if present
    if "scenario" in df_demand.columns:
        df_demand["scenario"] = df_demand["scenario"].astype(str).str.lower()
        want = str(args.demand_scenario).lower()
        df_demand = df_demand[df_demand["scenario"] == want].copy()
        if df_demand.empty:
            raise RuntimeError(f"Demand scenario '{args.demand_scenario}' produced empty demand table.")

    print("Loading generation...")
    df_gen = load_generation(gen_path)

    # Determine mode
    is_forecast = "timestamp" not in df_gen.columns

    # If historical: collapse to subsystem totals per timestamp FIRST (this is the units fix)
    # This prevents the 744x blow-up from summing MW time series.
    ts_gen = None
    if not is_forecast:
        ts_gen = (
            df_gen.groupby(["year", "month", "timestamp", "subsystem", "fuel_bucket"], as_index=False)["avail_mw"]
            .sum()
        )

    # Shared months
    months_d = set(df_demand[["year", "month"]].itertuples(index=False, name=None))
    months_g = set(df_gen[["year", "month"]].itertuples(index=False, name=None))
    months = sorted(months_d & months_g)
    if not months:
        raise RuntimeError("No overlapping (year, month) between demand and generation.")

    print(f"Months: {len(months)} | Trials: {args.n_trials} | OPFs: {len(months) * args.n_trials}")
    results, failures = [], []

    for trial in tqdm(range(1, args.n_trials + 1), desc="Trials", unit="trial"):
        for (year, month) in months:
            # demand per subsystem
            dsub = df_demand[(df_demand["year"] == year) & (df_demand["month"] == month)]
            subsys_demand = {r.subsystem: float(r.demand_mw) for r in dsub.itertuples(index=False)}

            # generation for this trial/month
            if is_forecast:
                gsub_df = df_gen[(df_gen["year"] == year) & (df_gen["month"] == month)].copy()
                if gsub_df.empty:
                    continue

                # per-trial noise (this is your forecast stochasticity in the SIM, not in the forecaster)
                if args.forecast_gen_noise_std and args.forecast_gen_noise_std > 0:
                    noise = rng.normal(0.0, float(args.forecast_gen_noise_std), size=len(gsub_df))
                    gsub_df["avail_mw"] = np.maximum(gsub_df["avail_mw"].astype(float).values * (1.0 + noise), 0.0)

                subsys_gen = {r.subsystem: float(r.avail_mw) for r in gsub_df.itertuples(index=False)}

            else:
                # historical: sample random DAY then take MEAN MW over timestamps in that day
                assert ts_gen is not None
                if args.day_sampling == "random":
                    day = pick_trial_day(ts_gen, year, month, rng)
                    if day is None:
                        # fallback to month mean
                        day_mask = None
                    else:
                        day_norm = pd.to_datetime(day).normalize()
                        day_mask = ts_gen["timestamp"].dt.normalize() == day_norm

                    if day_mask is None:
                        sub_ts = ts_gen[(ts_gen["year"] == year) & (ts_gen["month"] == month)]
                    else:
                        sub_ts = ts_gen[(ts_gen["year"] == year) & (ts_gen["month"] == month) & day_mask]
                else:
                    sub_ts = ts_gen[(ts_gen["year"] == year) & (ts_gen["month"] == month)]

                # Mean over timestamps -> representative MW
                gsub_df = (
                    sub_ts.groupby(["subsystem", "fuel_bucket"], as_index=False)["avail_mw"]
                    .mean()
                )
                subsys_gen = (
                    gsub_df.groupby("subsystem", as_index=False)["avail_mw"].sum()
                )
                subsys_gen = {r.subsystem: float(r.avail_mw) for r in subsys_gen.itertuples(index=False)}

            # build bucket map (uses real bucket if present; else share fallback)
            gen_by_bucket = aggregate_generation_by_bucket(
                gsub_df=gsub_df,
                subsys_gen=subsys_gen,
                sudeste_nuclear_share=float(args.sudeste_nuclear_share),
                norte_hydro_share=float(args.norte_hydro_share),
            )

            out = run_single_month(
                base_net=base_net,
                bus_lookup=bus_lookup,
                subsys_demand=subsys_demand,
                gen_by_subsys_bucket=gen_by_bucket,
                cost_map=cost_map,
                slack_cost=float(args.slack_cost),
                ens_cost=float(args.ens_cost),
                dump_cost=float(args.dump_cost),
                nuclear_min_frac=float(args.nuclear_min_frac),
                norte_hydro_min_frac=float(args.norte_hydro_min_frac),
                allow_slack=bool(args.allow_slack_imports),
                line_loading_percent=float(args.line_loading_percent),
            )

            out["trial"] = trial
            out["year"] = year
            out["month"] = month

            if ("opf_converged" in out.columns) and (not bool(out["opf_converged"].iloc[0])):
                failures.append(out)
            else:
                results.append(out)

    if results:
        df_out = pd.concat(results, ignore_index=True)
        df_out.to_csv(out_path, index=False)
        print(f"Saved converged results to: {out_path}")
    else:
        print("No converged results to save.")

    if failures:
        df_fail = pd.concat(failures, ignore_index=True)
        df_fail.to_csv(failures_path, index=False)
        print(f"Saved OPF failures ONLY to: {failures_path}")

    print("Done.")


if __name__ == "__main__":
    main()
