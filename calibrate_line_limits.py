"""Calibrate simplified 5-bus network line limits using transmission dataset."""
import json
import math
from pathlib import Path

import pandas as pd
import pandapower as pp

NETWORK_PATH = Path("models/brazil_5bus_network.json")
TRANSMISSION_PATH = Path("data/transmission_data.csv")
OUTPUT_PATH = NETWORK_PATH

LINE_MAP = {
    ("NORTE", "NORDESTE"): "LINE_NORTE_NORDESTE",
    ("NORTE", "SUDESTE"): "LINE_NORTE_SUDESTE",
    ("NORDESTE", "SUDESTE"): "LINE_NORDESTE_SUDESTE",
    ("NORDESTE", "SUL"): "LINE_NORDESTE_SUL",
    ("SUDESTE", "SUL"): "LINE_SUDESTE_SUL",
    ("PARAGUAI", "SUDESTE"): "LINE_PARAGUAI_SUDESTE",
    ("PARAGUAI", "SUL"): "LINE_PARAGUAI_SUL",
}

BUS_VN_KV = 230.0

def compute_pair_capacities(csv_path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_csv(csv_path)
    for col in ["subsys_name_from", "subsys_name_to"]:
        df[col] = df[col].str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    df = df[df["subsys_name_from"] != df["subsys_name_to"]]
    grouped = df.groupby(["subsys_name_from", "subsys_name_to"])["long_dur_capacity_no_lim"].sum().reset_index()
    agg = {}
    for _, row in grouped.iterrows():
        pair = tuple(sorted([row["subsys_name_from"], row["subsys_name_to"]]))
        agg[pair] = agg.get(pair, 0.0) + float(row["long_dur_capacity_no_lim"])
    return agg

def mw_to_ika(mw: float, voltage_kv: float) -> float:
    if mw <= 0:
        return 0.0
    return mw / (math.sqrt(3) * voltage_kv)

pair_caps = compute_pair_capacities(TRANSMISSION_PATH)
net = pp.from_json(str(NETWORK_PATH))
line_df = net.line.copy()

updates = {}
for (sub1, sub2), line_name in LINE_MAP.items():
    pair = tuple(sorted([sub1, sub2]))
    cap_mw = pair_caps.get(pair)
    if cap_mw is None or cap_mw <= 0:
        continue
    new_ika = mw_to_ika(cap_mw, BUS_VN_KV)
    idx = line_df[line_df["name"] == line_name].index
    if len(idx) == 0:
        continue
    net.line.loc[idx, "max_i_ka"] = new_ika
    updates[line_name] = {"cap_mw": cap_mw, "max_i_ka": new_ika}

pp.to_json(net, str(OUTPUT_PATH))
print("Updated line limits:")
for name, info in updates.items():
    print(f"  {name}: {info['cap_mw']:.1f} MW -> max_i_ka={info['max_i_ka']:.3f}")
