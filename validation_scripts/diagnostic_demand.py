import pandas as pd
import calendar
from pathlib import Path

DEMAND_PATH = Path("data/demand_data/demand_projection_2025_2028.csv")

def hours_in_month(y, m):
    return calendar.monthrange(int(y), int(m))[1] * 24

df = pd.read_csv(DEMAND_PATH)

# basic checks
print("Demand columns:", list(df.columns))
if "year" not in df.columns or "month" not in df.columns:
    raise SystemExit("Demand file missing year/month columns.")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["month"] = pd.to_numeric(df["month"], errors="coerce")
df = df.dropna(subset=["year","month"]).copy()
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

# pick demand column
if "demand_mw" in df.columns:
    dcol = "demand_mw"
elif "MWh" in df.columns:
    dcol = "MWh"
else:
    raise SystemExit("Need either demand_mw or MWh in demand file.")

df[dcol] = pd.to_numeric(df[dcol], errors="coerce")
df = df.dropna(subset=[dcol]).copy()

# system totals by month (raw)
sys = df.groupby(["year","month"], as_index=False)[dcol].sum()
sys["hours"] = [hours_in_month(y,m) for y,m in zip(sys["year"], sys["month"])]

# candidate interpretations -> convert to avg MW
sys["as_is"] = sys[dcol]                               # if already MW
sys["if_MWh_month"] = sys[dcol] / sys["hours"]         # if MWh per month
sys["if_MWh_day"] = sys[dcol] / 24.0                   # if MWh per day (already aggregated daily)
sys["if_MW_should_be_MWh_month"] = sys[dcol] / sys["hours"]  # same as above, for readability

print("\n=== SYSTEM TOTALS (first 12 months) ===")
print(sys.head(12)[["year","month",dcol,"as_is","if_MWh_month","if_MWh_day"]].to_string(index=False))

def describe(name, s):
    print(f"\n{name} (MW) describe:")
    print(s.describe(percentiles=[0.1,0.5,0.9]).to_string())

describe("as_is", sys["as_is"])
describe("if_MWh_month", sys["if_MWh_month"])
describe("if_MWh_day", sys["if_MWh_day"])

print("\nRule of thumb: system avg MW should be ~O(1e4–1e5). If you see ~1e6, you're in energy units.")
