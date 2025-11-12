import pandas as pd
import glob
import os

csv_folder = "data/weather_data"
csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))

# Start with the first file
first_file = csv_files[0]
var_name = os.path.basename(first_file).split("_")[0]
df_merged = pd.read_csv(first_file)
value_cols = [c for c in df_merged.columns if c not in ["time", "latitude", "longitude"]]
if len(value_cols) == 1:
    df_merged = df_merged.rename(columns={value_cols[0]: var_name})

# Merge one at a time
for f in csv_files[1:]:
    print(f"Merging: {f}")
    var_name = os.path.basename(f).split("_")[0]
    df = pd.read_csv(f)
    value_cols = [c for c in df.columns if c not in ["time", "latitude", "longitude"]]
    if len(value_cols) == 1:
        df = df.rename(columns={value_cols[0]: var_name})
    # merge incrementally to avoid many copies
    df_merged = pd.merge(df_merged, df, on=["time", "latitude", "longitude"], how="inner")

# Optional: downcast numerical columns to save RAM
for col in df_merged.select_dtypes(include="float"):
    df_merged[col] = pd.to_numeric(df_merged[col], downcast="float")

output_path = os.path.join(csv_folder, "merged_weather.csv")
print("Writing merged CSV...")
df_merged.to_csv(output_path, index=False)
print(f"Saved to {output_path}, shape: {df_merged.shape}")
