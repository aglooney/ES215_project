import pandas as pd

df = pd.read_csv("data/merged_generation_weather_v2.csv")

# 1) What does "time" actually look like?
print(df["time"].head(20))
print(df["time"].dtype)
print("unique time sample:", df["time"].dropna().astype(str).unique()[:20])

# 2) How many unique hours per day do you really have?
dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
df["dt"] = dt
df = df.dropna(subset=["dt"]).copy()
df["day"] = df["dt"].dt.normalize()
df["hour"] = df["dt"].dt.hour

hours_per_day = df.groupby("day")["hour"].nunique()
print(hours_per_day.describe())
print("fraction with only 1 hour:", (hours_per_day == 1).mean())
