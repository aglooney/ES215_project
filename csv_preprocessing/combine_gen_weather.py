import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

gen_path = "data/generation_data/all_generation_data.csv"
weather_path = "data/weather_data/merged_weather.csv"
output_path = "data/merged_generation_weather.csv"

print("Reading CSVs")
gen_data = pd.read_csv(gen_path)
weather_data = pd.read_csv(weather_path)

def nearest_weather_grid(gen_df, weather_df, gen_lat='grid_lat', gen_lon='grid_lon',
                         wx_lat='latitude', wx_lon='longitude'):
    """
    Matches each generator location to the nearest weather grid point.
    Returns a dataframe with generator lat/lon and the matched weather lat/lon.
    """
    # Extract coordinate arrays
    gen_coords = np.array(list(zip(gen_df[gen_lat], gen_df[gen_lon])))
    wx_coords = np.array(list(zip(weather_df[wx_lat], weather_df[wx_lon])))

    # Build KDTree for fast nearest-neighbor search
    tree = cKDTree(wx_coords)

    # Query nearest neighbor for each generator coordinate
    dist, idx = tree.query(gen_coords, k=1)

    # Add matched weather grid coordinates to generators
    gen_df = gen_df.copy()
    gen_df['wx_lat'] = weather_df.iloc[idx][wx_lat].values
    gen_df['wx_lon'] = weather_df.iloc[idx][wx_lon].values
    gen_df['wx_distance_deg'] = dist  # optional: distance in degrees

    return gen_df


print("Applying date transformation")
gen_data["date"] = pd.to_datetime(gen_data["date"]).dt.date
weather_data["date"] = pd.to_datetime(weather_data["time"]).dt.date

print("Applying lat/long grid transformation")
assert weather_data.duplicated(subset=["latitude", "longitude", "date"]).sum() == 0




keep_static_cols = ["subsys_name", "state_name", "plant_opn_mode", "plant_type", "fuel_type", "ons_id", "ceg"]
print("Computing Daily Generation")
daily_gen = (gen_data.groupby(["ceg", "latitude", "longitude", "date"], as_index=False)["gen_val(MW)"].sum())
gen_static = gen_data.groupby("ceg").first().reset_index()[keep_static_cols]
daily_gen = daily_gen.merge(gen_static, on="ceg", how='left', validate='m:m')

daily_gen = nearest_weather_grid(daily_gen, weather_data,
                                 gen_lat = "latitude", gen_lon="longitude",
                                 wx_lat="latitude", wx_lon="longitude")
print(f"Average distance between generator and weather grid cell: {daily_gen['wx_distance_deg'].mean():.3f}°")

daily_gen["wx_lat"] = daily_gen["wx_lat"].round(3)
daily_gen["wx_lon"] = daily_gen["wx_lon"].round(3)
weather_data["latitude"] = weather_data["latitude"].round(3)
weather_data["longitude"] = weather_data["longitude"].round(3)



print("Merging Dataframes")
merged = pd.merge(
    daily_gen,
    weather_data,
    how='left',
    left_on=['wx_lat', 'wx_lon', 'date'],
    right_on=['latitude', 'longitude', 'date']
)


print("Merged dataset summary:")
print("Rows:", merged.shape[0])
print("Columns:", merged.shape[1])
print("Example columns:", merged.columns[:15].tolist())

# Optional check: verify some weather columns merged
weather_vars = [c for c in merged.columns if c not in keep_static_cols + ["ceg", "date", "gen_val(MW)", "latitude", "longitude", "wx_lat", "wx_lon", "wx_distance_deg"]]
print("\nSample of merged weather variables:", weather_vars[:6])


print("Saving Merged DF")
merged.to_csv(output_path, index=False)

print("Job Completed!")