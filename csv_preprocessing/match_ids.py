import pandas as pd
import os

csv_files = [file for file in os.listdir("data/generation_data") if "generation_202" in file]


dfs = [pd.read_csv(f"data/generation_data/{f}") for f in csv_files]


df_gen = pd.concat(dfs, ignore_index=True)

df_coord = pd.read_csv("data/generation_coordinates_fixed.csv")

df_merged = df_gen.merge(df_coord[["ceg", "latitude", "longitude"]], on="ceg", how="left")

df_merged.to_csv("data/all_generation_data.csv", index=False)