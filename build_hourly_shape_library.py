#!/usr/bin/env python3
"""
build_hourly_shape_library.py

Build an hourly shape library s_h (mean=1) for each (subsystem, fuel_bucket, month).
Uses historical generation with date + time + gen_val(MW).

Output columns:
  subsystem, fuel_bucket, month, date, hour, shape

shape is dimensionless, mean(shape over 24h) == 1 for each (subsystem,bucket,date).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import unicodedata

ZONES = ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]

def normalize_subsystem(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    s = s.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    mapping = {"PARAGUAY": "PARAGUAI"}
    s = mapping.get(s, s)
    return s if s in ZONES else None

def strip_accents(s: str) -> str:
    s = str(s or "")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def classify_fuel_bucket(plant_type: str, fuel_type: str) -> str:
    pt = strip_accents(plant_type).upper()
    ft = strip_accents(fuel_type).upper()
    s = (pt + " " + ft).strip()
    if "NUCLEAR" in s:
        return "nuclear"
    if "SOLAR" in s or "FOTOV" in s:
        return "solar"
    if "EOL" in s or "WIND" in s:
        return "wind"
    if "HIDRO" in s or "HYDR" in s:
        return "hydro"
    return "thermal"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Historical merged gen file with date,time,gen_val(MW),subsys_name,plant_type,fuel_type")
    ap.add_argument("--output", default="results/hourly_shape_library.parquet")
    ap.add_argument("--min-days", type=int, default=10, help="Min distinct days required to keep a (subsys,bucket,month) group")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)

    # required cols
    needed = ["date", "time", "gen_val(MW)", "subsys_name"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Columns={list(df.columns)}")

    df["subsystem"] = df["subsys_name"].map(normalize_subsystem)
    df = df.dropna(subset=["subsystem"]).copy()

    # bucket
    if "fuel_bucket" in df.columns:
        df["fuel_bucket"] = df["fuel_bucket"].fillna("").astype(str).str.lower()
    else:
        pt = df["plant_type"] if "plant_type" in df.columns else ""
        ft = df["fuel_type"] if "fuel_type" in df.columns else ""
        df["fuel_bucket"] = [classify_fuel_bucket(a, b) for a, b in zip(pt, ft)]

    # datetime parsing: date + time
    # Handles time like "13:00:00" or "13:00"
    dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    if dt.isna().mean() > 0.2:
        # fallback: try just date (if you only have daily)
        dt = pd.to_datetime(df["date"], errors="coerce")
    df["dt"] = dt
    df = df.dropna(subset=["dt"]).copy()

    df["date_only"] = df["dt"].dt.normalize()
    df["month"] = df["dt"].dt.month
    df["hour"] = df["dt"].dt.hour

    df["mw"] = pd.to_numeric(df["gen_val(MW)"], errors="coerce")
    df = df.dropna(subset=["mw"]).copy()

    # hourly totals per day
    hourly = (
        df.groupby(["subsystem", "fuel_bucket", "month", "date_only", "hour"], as_index=False)["mw"].sum()
    )

    # ensure all 24 hours exist per day; fill missing with 0
    # compute daily mean MW (over 24h)
    # pivot each day to 24-vector, normalize by its mean -> shape (mean=1)
    out_rows = []
    for (subsys, bucket, month, day), g in hourly.groupby(["subsystem", "fuel_bucket", "month", "date_only"]):
        vec = np.zeros(24, dtype=float)
        for r in g.itertuples(index=False):
            if 0 <= int(r.hour) <= 23:
                vec[int(r.hour)] = float(r.mw)

        mean = vec.mean()
        if mean <= 0:
            continue
        shape = vec / mean  # mean(shape)=1

        for h in range(24):
            out_rows.append({
                "subsystem": subsys,
                "fuel_bucket": bucket,
                "month": int(month),
                "date": pd.Timestamp(day).date().isoformat(),
                "hour": int(h),
                "shape": float(shape[h]),
            })

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise RuntimeError("No shapes produced. Check your input columns/time resolution.")

    # prune groups with too few distinct days
    day_counts = out.groupby(["subsystem","fuel_bucket","month"])["date"].nunique().reset_index(name="n_days")
    keep = day_counts[day_counts["n_days"] >= args.min_days][["subsystem","fuel_bucket","month"]]
    out = out.merge(keep, on=["subsystem","fuel_bucket","month"], how="inner")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Saved shape library: {args.output}")
    print(out.head())

if __name__ == "__main__":
    main()
