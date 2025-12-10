"""Visualize the aggregated 5-bus Brazilian grid on a real basemap."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pyogrio

GEN_PATH = Path("data/merged_generation_weather_v2.csv")
TRANS_PATH = Path("data/transmission_data.csv")
OUTPUT = Path("results/brazil_5bus_map.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Default coordinates if a subsystem is missing from the dataset
DEFAULT_COORDS = {
    "NORTE": (-2.5, -60.0),        # Amazonas basin
    "NORDESTE": (-9.0, -40.0),     # interior Nordeste
    "SUDESTE": (-22.0, -47.0),     # São Paulo region
    "SUL": (-27.5, -51.0),         # southern Brazil
    "PARAGUAI": (-25.4, -54.6),    # Itaipu region
}

# Manual overrides (MW) for missing or zero-capacity interties
MANUAL_CAPS = {
    ("PARAGUAI", "SUDESTE"): 6500.0,
    ("PARAGUAI", "SUL"): 2000.0,
}

PAIR_ADJUSTMENTS = {
    frozenset({"PARAGUAI", "SUDESTE"}): (2.0, 1.6),
    frozenset({"SUDESTE", "SUL"}): (1.6, 1.2),
}


def compute_bus_coords() -> dict:
    coords = DEFAULT_COORDS.copy()
    if not GEN_PATH.exists():
        return coords
    usecols = ["gen_latitude", "gen_longitude", "subsys_name"]
    df = pd.read_csv(GEN_PATH, usecols=usecols)
    df["subsystem"] = (
        df["subsys_name"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE", regex=False)
    )
    df = df.dropna(subset=["gen_latitude", "gen_longitude"])
    df = df[(df["gen_latitude"].between(-30, 5)) & (df["gen_longitude"].between(-73, -35))]
    grouped = df.groupby("subsystem")[["gen_latitude", "gen_longitude"]].mean().dropna()
    for subsys, row in grouped.iterrows():
        lat, lon = row["gen_latitude"], row["gen_longitude"]
        default_lat, default_lon = coords.get(subsys, (lat, lon))
        if abs(lat - default_lat) > 5 or abs(lon - default_lon) > 8:
            lat, lon = default_lat, default_lon
        coords[subsys] = (lat, lon)
    return coords


def aggregate_line_caps() -> dict:
    if not TRANS_PATH.exists():
        return MANUAL_CAPS.copy()
    df = pd.read_csv(TRANS_PATH)
    for col in ["subsys_name_from", "subsys_name_to"]:
        df[col] = df[col].str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE", regex=False)
    df = df[df["subsys_name_from"] != df["subsys_name_to"]]
    grouped = df.groupby(["subsys_name_from", "subsys_name_to"])["long_dur_capacity_no_lim"].sum().reset_index()
    caps = {}
    for _, row in grouped.iterrows():
        key = (row["subsys_name_from"], row["subsys_name_to"])
        caps[key] = caps.get(key, 0.0) + float(row["long_dur_capacity_no_lim"])
    for key, value in MANUAL_CAPS.items():
        caps[key] = caps.get(key, 0.0) + value
    return caps


def load_basemap() -> gpd.GeoDataFrame:
    shp = Path(pyogrio.__file__).resolve().parent / "tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
    world = gpd.read_file(shp)
    return world[world["name"].isin(["Brazil", "Paraguay"])]


def main():
    coords = compute_bus_coords()
    caps = aggregate_line_caps()
    basemap = load_basemap()

    fig, ax = plt.subplots(figsize=(8, 9))
    basemap.plot(ax=ax, color="#f2f2f2", edgecolor="#666666")
    ax.set_xlim(-75, -32)
    ax.set_ylim(-35, 6)
    ax.set_axis_off()
    ax.set_title("Aggregated 5-Bus Brazilian Grid", fontsize=14)

    label_lines = []

    # Collect label text without drawing lines (per user request)
    for (a, b), cap in caps.items():
        if cap <= 0 or a not in coords or b not in coords:
            continue
        label_lines.append(f"{a} -> {b}: {cap/1000:.1f} GW")

    for subsys, (lat, lon) in coords.items():
        ax.scatter(lon, lat, s=160, color="#cc3300", edgecolor="white", linewidth=1.2, zorder=3)
        if subsys == "PARAGUAI":
            ax.text(lon - 5.0, lat, subsys, fontsize=10, weight="bold", color="#202020")
        elif subsys == "SUL":
            ax.text(lon + 0.6, lat - 1.5, subsys, fontsize=10, weight="bold", color="#202020")
        else:
            ax.text(lon + 0.6, lat + 0.6, subsys, fontsize=10, weight="bold", color="#202020")

    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    label_path = OUTPUT.with_name("brazil_5bus_line_caps.txt")
    with label_path.open("w") as f:
        for line in label_lines:
            f.write(line + "\n")
    print(f"Saved line capacity legend to {label_path}")
    print(f"Saved map to {OUTPUT}")


if __name__ == "__main__":
    main()
