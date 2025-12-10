#!/usr/bin/env python3
"""
diagnose_forecast_opf.py

Diagnose forecast-driven OPF feasibility + ENS root cause.

Inputs:
  --net-json   : zonal 5-bus pandapower JSON
  --demand     : demand CSV with columns [subsystem, year, month, demand_mw] OR [MWh] (converted)
  --gen-forecast : forecast CSV with columns [year, month, subsystem, predicted_avg_mw]

For each month, runs:
  Case1: slack allowed + NO line limits (should always be feasible if units OK)
  Case2: slack allowed + with line limits (congestion test)
  Case3: slack DISallowed + NO line limits (pure supply shortfall test)

Prints:
  - total forecast gen vs total demand margin
  - convergence, ENS, slack import, binding line count (case2)
"""

import argparse
import calendar
import copy
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.auxiliary import OPFNotConverged


ZONES = ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]
DEMAND_EXCLUDE_SUBSYSTEMS = {"PARAGUAI"}

GEN_COST = 1.0
SLACK_COST = 1000.0
ENS_COST = 1e6

EPS = 1e-6
BINDING_TOL_PCT = 1e-3


def normalize_subsystem(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    mapping = {
        "SUDESTE/CENTRO-OESTE": "SUDESTE",
        "SE/CO": "SUDESTE",
        "PARAGUAY": "PARAGUAI",
        "SOUTHEAST": "SUDESTE",
        "SOUTH": "SUL",
        "NORTHEAST": "NORDESTE",
    }
    s = mapping.get(s, s)
    return s if s in ZONES else None


def strip_accents(s: str) -> str:
    s = str(s or "")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def ensure_scaling(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    if "scaling" not in df.columns:
        df["scaling"] = 1.0
    df["scaling"] = df["scaling"].fillna(1.0).astype(float)


def build_bus_lookup_from_net(net: pp.pandapowerNet) -> Dict[str, int]:
    lookup = {}
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


def compute_binding_lines(net: pp.pandapowerNet) -> int:
    if net.line.empty or not hasattr(net, "res_line") or net.res_line.empty:
        return 0
    max_allowed = net.line.get("max_loading_percent", pd.Series([100.0]*len(net.line), index=net.line.index))
    loading = net.res_line.get("loading_percent", pd.Series([0.0]*len(net.line), index=net.line.index))
    binding_mask = loading >= (max_allowed - BINDING_TOL_PCT)
    return int(binding_mask.sum())


def load_demand(demand_path: Path) -> pd.DataFrame:
    df = pd.read_csv(demand_path)
    if "subsystem" not in df.columns:
        raise ValueError(f"Demand file must have subsystem. Columns={list(df.columns)}")
    for c in ["year", "month"]:
        if c not in df.columns:
            raise ValueError(f"Demand file must have {c}. Columns={list(df.columns)}")

    df["subsystem"] = df["subsystem"].astype(str).str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE", regex=False)
    df["subsystem"] = df["subsystem"].map(normalize_subsystem)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["subsystem", "year", "month"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    if "demand_mw" in df.columns:
        df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
        df = df.dropna(subset=["demand_mw"]).copy()
        return df

    if "MWh" in df.columns:
        df["MWh"] = pd.to_numeric(df["MWh"], errors="coerce")
        df = df.dropna(subset=["MWh"]).copy()
        hours = [calendar.monthrange(int(y), int(m))[1] * 24 for y, m in zip(df["year"], df["month"])]
        df["demand_mw"] = df["MWh"].values / np.array(hours, dtype=float)
        return df

    raise ValueError(f"Demand file must have demand_mw or MWh. Columns={list(df.columns)}")


def load_forecast_gen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"year", "month", "subsystem", "predicted_avg_mw"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Forecast must have {sorted(need)}. Columns={list(df.columns)}")
    df = df.copy()
    df["subsystem"] = df["subsystem"].map(normalize_subsystem)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["predicted_avg_mw"] = pd.to_numeric(df["predicted_avg_mw"], errors="coerce")
    df = df.dropna(subset=["subsystem", "year", "month", "predicted_avg_mw"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df = df[df["month"].between(1, 12)].copy()
    df = df[df["subsystem"].isin(ZONES)].copy()
    df = df.rename(columns={"predicted_avg_mw": "avail_mw"})
    return df


def run_opf_case(
    base_net: pp.pandapowerNet,
    bus_lookup: Dict[str, int],
    subsys_demand: Dict[str, float],
    subsys_gen: Dict[str, float],
    slack_allowed: bool,
    enforce_line_limits: bool,
) -> Tuple[bool, float, float, int]:
    """
    returns: (converged, ens_mw, slack_p_mw, binding_count)
    """
    net = copy.deepcopy(base_net)

    # wipe injections/costs
    for t in ("load", "gen", "sgen", "poly_cost"):
        tab = getattr(net, t)
        if not tab.empty:
            tab.drop(tab.index, inplace=True)

    # lines: either keep as-is (limits) or effectively remove by raising max_loading
    if not net.line.empty:
        if enforce_line_limits:
            # keep whatever the JSON has; just ensure column exists
            if "max_loading_percent" not in net.line.columns:
                net.line["max_loading_percent"] = 100.0
        else:
            net.line["max_loading_percent"] = 1e6  # effectively no binding

    # slack
    if net.ext_grid.empty:
        sb = bus_lookup.get("SUDESTE", list(bus_lookup.values())[0])
        pp.create_ext_grid(net, bus=sb, vm_pu=1.0, name="GRID_SLACK")
    slack_idx = int(net.ext_grid.index[0])
    if slack_allowed:
        net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 1e9]
    else:
        net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 0.0]
    pp.create_poly_cost(net, element=slack_idx, et="ext_grid", cp1_eur_per_mw=SLACK_COST, cp0_eur=0.0)

    # loads
    for z, mw in subsys_demand.items():
        if z in DEMAND_EXCLUDE_SUBSYSTEMS:
            continue
        if z not in bus_lookup:
            continue
        mw = float(mw)
        if mw <= 0:
            continue
        pp.create_load(net, bus=bus_lookup[z], p_mw=mw, q_mvar=0.0, name=f"LOAD_{z}")
    ensure_scaling(net.load)
    if not net.load.empty:
        net.load["controllable"] = False  # avoids your warning

    # gens (one per zone, just availability)
    for z, pmax in subsys_gen.items():
        if z not in bus_lookup:
            continue
        pmax = float(pmax)
        if pmax <= 0:
            continue
        g = pp.create_gen(
            net,
            bus=bus_lookup[z],
            p_mw=pmax,
            vm_pu=1.0,
            min_p_mw=0.0,
            max_p_mw=pmax,
            name=f"GEN_{z}",
            controllable=True,
        )
        pp.create_poly_cost(net, element=g, et="gen", cp1_eur_per_mw=GEN_COST, cp0_eur=0.0)
    ensure_scaling(net.gen)
    if not net.gen.empty:
        net.gen["controllable"] = True

    # ENS at each load bus
    ens_idx = []
    if not net.load.empty:
        for _, r in net.load.iterrows():
            bus = int(r["bus"])
            load_mw = float(r["p_mw"])
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
            pp.create_poly_cost(net, element=g, et="gen", cp1_eur_per_mw=ENS_COST, cp0_eur=0.0)
            ens_idx.append(int(g))

    # solve
    try:
        pp.rundcopp(net, verbose=False)
        conv = True
    except OPFNotConverged:
        return False, np.nan, np.nan, 0

    # metrics
    slack_p = float(net.res_ext_grid.at[slack_idx, "p_mw"]) if hasattr(net, "res_ext_grid") and not net.res_ext_grid.empty else 0.0
    ens = float(net.res_gen.loc[ens_idx, "p_mw"].sum()) if (ens_idx and hasattr(net, "res_gen") and not net.res_gen.empty) else 0.0
    bind = compute_binding_lines(net) if enforce_line_limits else 0

    return conv, ens, slack_p, bind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-json", default="models/pandapower_snapshots/brazil_network_zonal_5bus_from_tx.json")
    ap.add_argument("--demand", default="data/demand_data/demand_projection_2025_2028.csv")
    ap.add_argument("--gen-forecast", default="results/generation_forecast_2025_2028.csv")
    ap.add_argument("--max-months", type=int, default=24, help="limit printed months (for readability)")
    args = ap.parse_args()

    net = pp.from_json(args.net_json)
    bus_lookup = build_bus_lookup_from_net(net)
    if not bus_lookup:
        raise RuntimeError("Could not infer BUS_* mapping from net.bus names.")

    df_d = load_demand(Path(args.demand))
    df_g = load_forecast_gen(Path(args.gen_forecast))

    # totals
    demand_tot = df_d.groupby(["year", "month"], as_index=False)["demand_mw"].sum().rename(columns={"demand_mw": "total_dem_mw"})
    gen_tot = df_g.groupby(["year", "month"], as_index=False)["avail_mw"].sum().rename(columns={"avail_mw": "total_gen_mw"})
    m = demand_tot.merge(gen_tot, on=["year", "month"], how="inner")
    if m.empty:
        raise RuntimeError("No overlapping months between demand and forecast generation.")
    m["margin_mw"] = m["total_gen_mw"] - m["total_dem_mw"]
    m = m.sort_values(["year", "month"]).reset_index(drop=True)

    print("\n=== FORECAST SUPPLY CHECK (total avg MW) ===")
    print(m.head(args.max_months).to_string(index=False))
    frac_def = float((m["margin_mw"] < 0).mean())
    print(f"\nFraction of months with total_gen < total_demand: {frac_def:.3f}")

    print("\n=== OPF DIAGNOSTIC (forecast gen) ===")
    print("Case1: slack allowed + NO line limits")
    print("Case2: slack allowed + WITH line limits")
    print("Case3: slack DISALLOWED + NO line limits\n")

    rows = []
    for _, r in m.head(args.max_months).iterrows():
        y, mo = int(r["year"]), int(r["month"])

        dsub = df_d[(df_d.year == y) & (df_d.month == mo)]
        gsub = df_g[(df_g.year == y) & (df_g.month == mo)]

        subsys_demand = {z: float(v) for z, v in dsub.groupby("subsystem")["demand_mw"].sum().items() if normalize_subsystem(z)}
        subsys_gen = {z: float(v) for z, v in gsub.groupby("subsystem")["avail_mw"].sum().items() if normalize_subsystem(z)}

        c1 = run_opf_case(net, bus_lookup, subsys_demand, subsys_gen, slack_allowed=True,  enforce_line_limits=False)
        c2 = run_opf_case(net, bus_lookup, subsys_demand, subsys_gen, slack_allowed=True,  enforce_line_limits=True)
        c3 = run_opf_case(net, bus_lookup, subsys_demand, subsys_gen, slack_allowed=False, enforce_line_limits=False)

        print(f"{y:04d}-{mo:02d} margin={r['margin_mw']:.2f} MW")
        print(f"  Case1 slack+NOlimits : converged={c1[0]} ENS={c1[1]:.6f} slack_p={c1[2]:.2f}")
        print(f"  Case2 slack+limits   : converged={c2[0]} ENS={c2[1]:.6f} slack_p={c2[2]:.2f} binding_lines={c2[3]}")
        print(f"  Case3 NOslack+NOlims : converged={c3[0]} ENS={c3[1]:.6f} slack_p={c3[2]:.2f}\n")

        rows.append((y, mo, r["margin_mw"], *c1, *c2, *c3))

    out = pd.DataFrame(rows, columns=[
        "year","month","margin_mw",
        "c1_converged","c1_ens","c1_slack_p","c1_bind",
        "c2_converged","c2_ens","c2_slack_p","c2_bind",
        "c3_converged","c3_ens","c3_slack_p","c3_bind",
    ])
    print("=== SUMMARY FLAGS ===")
    # Simple interpretation per month:
    # - If Case1 has ENS > small -> units/setup wrong (should always be feasible)
    # - If Case1 ok but Case2 ENS -> congestion
    # - If Case3 ENS but Case1 ok -> true supply deficit (slack is what saved you)
    bad_units = (out["c1_converged"] & (out["c1_ens"] > 1e-3)).sum()
    congested = (out["c1_converged"] & (out["c1_ens"] <= 1e-3) & out["c2_converged"] & (out["c2_ens"] > 1e-3)).sum()
    supply_def = (out["c1_converged"] & (out["c1_ens"] <= 1e-3) & out["c3_converged"] & (out["c3_ens"] > 1e-3)).sum()
    print(f"Months flagged units/setup issue (Case1 ENS>1e-3): {int(bad_units)}")
    print(f"Months flagged congestion (only Case2 ENS):         {int(congested)}")
    print(f"Months flagged supply deficit (Case3 ENS):          {int(supply_def)}")
    print("\nDone.")

if __name__ == "__main__":
    main()
