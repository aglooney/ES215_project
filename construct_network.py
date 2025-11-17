import pandas as pd
import numpy as np
import time
import pandapower as pp
import pandapower.topology as top
import requests
from pathlib import Path

# ===========================================================
# CONFIG
# ===========================================================
TX_PATH = "data/transmission_data.csv"
CACHE_PATH = "data/substation_coords.csv"
USER_AGENT = "es215_project_openstreetmap"


# ===========================================================
# STEP 1 — LOAD TRANSMISSION DATA
# ===========================================================
df = pd.read_csv(TX_PATH)

# Only active lines
df["opn_deactivation_date"] = pd.to_datetime(df["opn_deactivation_date"], errors="coerce")
df = df[df["opn_deactivation_date"].isna()].copy()

# Drop empty bus info
df = df.dropna(subset=["sending_bus_num", "receiving_bus_num"])


# ===========================================================
# STEP 2 — EXTRACT UNIQUE SUBSTATION NAMES
# ===========================================================
subs_from = df["substation_name_from"].dropna().unique()
subs_to   = df["substation_name_to"].dropna().unique()

all_subs = np.unique(np.concatenate([subs_from, subs_to]))
print(f"Found {len(all_subs)} unique substation names.")


# ===========================================================
# STEP 3 — LOAD CACHE IF EXISTS (recommended)
# ===========================================================
if Path(CACHE_PATH).exists():
    df_cache = pd.read_csv(CACHE_PATH)
    cache = dict(zip(df_cache["substation_name"], 
                     zip(df_cache["lat"], df_cache["lon"])))
    print(f"Loaded {len(cache)} cached coordinates.")
else:
    cache = {}

def geocode_substation(name):
    """Query OSM Nominatim and return (lat, lon)."""
    # Check cache first
    if name in cache and not pd.isna(cache[name][0]):
        return cache[name]
    
    # Query OSM
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"Subestação {name}, Brazil",
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        if len(data) == 0:
            cache[name] = (np.nan, np.nan)
        else:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            cache[name] = (lat, lon)

        # Sleep to stay within Nominatim limits
        time.sleep(1.1)
        return cache[name]

    except Exception:
        cache[name] = (np.nan, np.nan)
        return cache[name]


# ===========================================================
# STEP 4 — GEOCODE ALL SUBSTATIONS
# ===========================================================
coords = []
for name in all_subs:
    lat, lon = geocode_substation(name)
    coords.append((name, lat, lon))
    print(f"Geocoded {name}: {lat}, {lon}")

df_coords = pd.DataFrame(coords, columns=["substation_name", "lat", "lon"])
df_coords.to_csv(CACHE_PATH, index=False)
print("Saved substation coordinates to cache.")


# ===========================================================
# STEP 5 — MERGE COORDINATES INTO TRANSMISSION DATA
# ===========================================================
df = df.merge(
    df_coords.add_prefix("from_"),
    how="left",
    left_on="substation_name_from",
    right_on="from_substation_name"
).merge(
    df_coords.add_prefix("to_"),
    how="left",
    left_on="substation_name_to",
    right_on="to_substation_name"
)

# Coordinates for buses (sending and receiving)
df["from_lat"] = df["from_lat"]
df["from_lon"] = df["from_lon"]
df["to_lat"] = df["to_lat"]
df["to_lon"] = df["to_lon"]


# ===========================================================
# STEP 6 — BUILD PANDAPOWER NETWORK WITH REAL LOCATIONS
# ===========================================================
net = pp.create_empty_network()

# Create bus map
bus_map = {}

for _, row in df.iterrows():
    for bus_num, lat, lon in [
        (row["sending_bus_num"], row["from_lat"], row["from_lon"]),
        (row["receiving_bus_num"], row["to_lat"], row["to_lon"]),
    ]:
        if bus_num not in bus_map:
            vn = row["trans_voltage_kv"]
            # Create bus
            b_idx = pp.create_bus(net, vn_kv=vn, name=f"bus_{bus_num}")
            bus_map[bus_num] = b_idx

            # Add real-world coordinates
            net.bus.loc[b_idx, "x"] = lon
            net.bus.loc[b_idx, "y"] = lat


# Add lines
def get_max_i_ka(row):
    candidates = [
        "long_dur_capacity_lim",
        "long_dur_capacity_no_lim",
        "short_dur_opn_cap_lim",
        "short_dur_opn_cap_no_lim",
        "summer_day_long_cap",
        "summer_night_long_cap",
        "winter_day_long_cap",
        "winter_night_long_cap",
    ]
    for c in candidates:
        if c in row and not pd.isna(row[c]):
            return row[c] / 1000.0
    return 1.0  # default


for _, row in df.iterrows():
    from_bus = bus_map[row["sending_bus_num"]]
    to_bus   = bus_map[row["receiving_bus_num"]]

    length = row["line_length_km"]
    r_tot  = row["pos_seq_resistance"]
    x_tot  = row["pos_seq_reactance"]

    if length <= 0:
        continue

    r_per_km = r_tot / length
    x_per_km = x_tot / length
    max_i_ka = get_max_i_ka(row)

    pp.create_line_from_parameters(
        net,
        from_bus,
        to_bus,
        length_km=length,
        r_ohm_per_km=r_per_km,
        x_ohm_per_km=x_per_km,
        c_nf_per_km=0.0,
        max_i_ka=max_i_ka,
        name=row["trans_line_name"],
        type="ol"
    )


# ===========================================================
# DONE! NETWORK WITH REAL GEO COORDS
# ===========================================================
print("Network built successfully with real substation coordinates.")
print(net)
