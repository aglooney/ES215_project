import pandas as pd, calendar
raw = pd.read_csv("data/merged_generation_weather_v2.csv", parse_dates=["date"])
raw["subsystem"] = raw["subsys_name"].astype(str).str.upper().str.replace("SUDESTE/CENTRO-OESTE","SUDESTE")
raw["year"] = raw["date"].dt.year
raw["month"] = raw["date"].dt.month

subsys, year, month = "SUDESTE", 2020, 1
m = raw[(raw.subsystem==subsys) & (raw.year==year) & (raw.month==month)]
hours = calendar.monthrange(year, month)[1]*24

B = m["gen_val(MW)"].sum()
A = B / hours
print("sum(gen_val)=", B, "  avgMW=sum/hours=", A, "  ratio=", B/max(A,1e-9))
