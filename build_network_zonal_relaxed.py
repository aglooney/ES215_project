"""
build_network_zonal_relaxed.py

Create a 5-bus Brazil zonal network with looser transfer capability and sensible unit handling.

Changes vs the older builder:
- Treat generation values as MW (no 1/24 scaling).
- Unlimited-ish slack bounds (±1e6 MW) with low cost.
- Collapse inter-zone corridors; sum MW limits; combine reactances in parallel (1/x).
- Use a high max_loading_percent default (can override when running OPF).

Outputs:
  models/pandapower_snapshots/brazil_network_zonal_5bus_relaxed.json
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pandapower as pp

TX_PATH = "data/transmission_data.csv"
ZONES = ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]
ZONAL_VN_KV = 500.0

SLACK_ZONE = "SUDESTE"
SLACK_MIN_P_MW = -1e6
SLACK_MAX_P_MW = 1e6
SLACK_COST = 1.0

OUT_DIR = Path("models/pandapower_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "brazil_network_zonal_5bus_relaxed.json"

SQRT3 = 1.7320508075688772

REQUIRED_COLS = [
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


def norm_zone(x: str) -> str:
    return str(x).strip().upper()


def ensure_datetime(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_datetime(df[col], errors="coerce")


def a_to_ka(i_a: float) -> float:
    return float(i_a) / 1000.0


def mw_from_i_ka(i_ka: float, v_kv: float) -> float:
    return SQRT3 * float(v_kv) * float(i_ka)


def i_ka_from_mw(p_mw: float, v_kv: float) -> float:
    v = float(v_kv)
    return float(p_mw) / (SQRT3 * v) if v > 0 else 1.0


def pick_capacity_a(row: pd.Series) -> float | None:
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


def infer_limit_mw(row: pd.Series) -> float:
    v_kv = float(row["trans_voltage_kv"])
    cap_a = pick_capacity_a(row)
    if cap_a is None:
        return mw_from_i_ka(1.0, v_kv)
    return mw_from_i_ka(a_to_ka(cap_a), v_kv)


def x_pu_from_ohm(x_ohm: float, v_kv: float, s_base_mva: float) -> float:
    v = float(v_kv)
    if v <= 0:
        return np.nan
    return float(x_ohm) * float(s_base_mva) / (v * v)


def x_ohm_from_pu(x_pu: float, v_kv: float, s_base_mva: float) -> float:
    v = float(v_kv)
    if v <= 0:
        return np.nan
    return float(x_pu) * (v * v) / float(s_base_mva)


def main():
    df = pd.read_csv(TX_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in transmission data: {missing}")

    ensure_datetime(df, "opn_deactivation_date")
    for c in ["sending_bus_num", "receiving_bus_num", "line_length_km", "pos_seq_resistance", "pos_seq_reactance", "trans_voltage_kv"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["subsys_name_from"] = df["subsys_name_from"].astype(str).map(norm_zone)
    df["subsys_name_to"] = df["subsys_name_to"].astype(str).map(norm_zone)

    # active lines, clean numerics
    df = df[df["opn_deactivation_date"].isna()].copy()
    df = df.dropna(subset=["sending_bus_num", "receiving_bus_num", "line_length_km", "pos_seq_reactance", "trans_voltage_kv"]).copy()
    df = df[df["line_length_km"] > 0]

    # keep only inter-zone lines among the 5 zones
    df = df[df["subsys_name_from"].isin(ZONES) & df["subsys_name_to"].isin(ZONES)]
    df = df[df["subsys_name_from"] != df["subsys_name_to"]]

    # per-line computations
    df["limit_mw_line"] = df.apply(infer_limit_mw, axis=1)
    df["x_ohm_total"] = df["pos_seq_reactance"].astype(float)
    df["v_kv_line"] = df["trans_voltage_kv"].astype(float)

    net = pp.create_empty_network(sn_mva=100.0)
    b = {z: pp.create_bus(net, name=f"BUS_{z}", vn_kv=float(ZONAL_VN_KV)) for z in ZONES}

    eg = pp.create_ext_grid(net, bus=b.get(SLACK_ZONE, list(b.values())[0]), vm_pu=1.0, name="GRID_SLACK")
    net.ext_grid.loc[eg, ["min_p_mw", "max_p_mw"]] = [SLACK_MIN_P_MW, SLACK_MAX_P_MW]
    pp.create_poly_cost(net, eg, "ext_grid", cp1_eur_per_mw=SLACK_COST)

    # undirected key
    df["a"] = df[["subsys_name_from", "subsys_name_to"]].min(axis=1)
    df["b"] = df[["subsys_name_from", "subsys_name_to"]].max(axis=1)

    s_base = float(net.sn_mva)

    corridors = []
    for (a, bzone), group in df.groupby(["a", "b"]):
        limit_mw = float(group["limit_mw_line"].sum())

        xs = group["x_ohm_total"].astype(float).values
        xs = xs[np.isfinite(xs) & (xs > 0)]
        if len(xs):
            b_eq = np.sum(1.0 / xs)
            x_eq_ohm = float(1.0 / b_eq) if b_eq > 0 else 10.0
        else:
            x_eq_ohm = 10.0

        volts = sorted(group["v_kv_line"].dropna().astype(float).unique().tolist())
        x_pu = x_pu_from_ohm(x_eq_ohm, ZONAL_VN_KV, s_base)
        corridors.append(
            {
                "fr": str(a),
                "to": str(bzone),
                "limit_mw": limit_mw,
                "x_pu": x_pu,
                "voltages": ",".join(str(int(v)) for v in volts) if volts else "",
                "n_lines": len(group),
            }
        )

    print("\n=== Collapsed corridors (relaxed) ===")
    for c in sorted(corridors, key=lambda x: x["limit_mw"], reverse=True):
        print(c)
    print("====================================\n")

    for c in corridors:
        fr = c["fr"]
        to = c["to"]
        limit_mw = c["limit_mw"]
        x_pu = c["x_pu"]
        x_ohm = x_ohm_from_pu(x_pu, ZONAL_VN_KV, s_base)
        max_i_ka = i_ka_from_mw(limit_mw, ZONAL_VN_KV)

        if not (np.isfinite(x_ohm) and x_ohm > 0 and np.isfinite(max_i_ka) and max_i_ka > 0):
            continue

        pp.create_line_from_parameters(
            net,
            from_bus=b[fr],
            to_bus=b[to],
            length_km=1.0,
            r_ohm_per_km=0.0,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=f"CORRIDOR_{fr}_{to}",
            type="ol",
        )

    pp.to_json(net, OUT_JSON)
    print(f"Saved relaxed zonal 5-bus network to: {OUT_JSON}")
    print(f"Slack bounds: [{SLACK_MIN_P_MW}, {SLACK_MAX_P_MW}] MW @ cost {SLACK_COST}")


if __name__ == "__main__":
    main()
