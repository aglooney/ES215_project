"""Forecast subsystem monthly generation (avg MW) using autoregressive + seasonal features."""
import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

parser = argparse.ArgumentParser(description="Forecast subsystem generation using AR+seasonal regression.")
parser.add_argument("--input", default="data/merged_generation_weather_v2.csv")
parser.add_argument("--output", default="results/generation_forecast_2025_2028.csv")
parser.add_argument("--start-year", type=int, default=2024)
parser.add_argument("--end-year", type=int, default=2028)
parser.add_argument("--noise-std", type=float, default=0.05, help="Std dev for multiplicative Gaussian noise.")
parser.add_argument("--alpha", type=float, default=5.0, help="Ridge regularization strength.")
parser.add_argument(
    "--model",
    choices=["ridge", "hgb"],
    default="hgb",
    help="Base regressor: ridge (linear) or hgb (HistGradientBoosting).",
)
parser.add_argument("--hgb-learning-rate", type=float, default=0.1, help="HistGB learning rate.")
parser.add_argument("--hgb-max-depth", type=int, default=6, help="HistGB max depth.")
parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31, help="HistGB max leaves.")
parser.add_argument(
    "--val-start-year",
    type=int,
    default=None,
    help="First historical year reserved for validation metrics (defaults to last available year).",
)
parser.add_argument(
    "--val-end-year",
    type=int,
    default=None,
    help="Last historical year reserved for validation metrics (defaults to same as --val-start-year).",
)
parser.add_argument(
    "--val-leakage",
    action="store_true",
    help="When set, validation uses actual lag features (leaky, optimistic metrics).",
)
parser.add_argument(
    "--trend-weight",
    type=float,
    default=1.0,
    help="Weight applied to annual growth trend (1.0 applies full slope).",
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)

WEATHER_FEATURES = ["ETo", "RH", "Rs", "Tmin", "Tmax", "pr", "u2"]


def month_hours(year: int, month: int) -> int:
    return calendar.monthrange(int(year), int(month))[1] * 24


raw = pd.read_csv(args.input, parse_dates=["date"])
raw["subsystem"] = (
    raw["subsys_name"].astype(str).str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
)
raw["year"] = raw["date"].dt.year
raw["month"] = raw["date"].dt.month

# ---- UNIT FIX: input values look like daily energy (MWh) per plant ----
# Step 1: subsystem daily energy (MWh) = sum across plants per calendar day
subsys_daily = (
    raw.groupby(["subsystem", "date"])
    .agg({**{"gen_val(MW)": "sum"}, **{feat: "mean" for feat in WEATHER_FEATURES}})
    .reset_index()
    .rename(columns={"gen_val(MW)": "energy_mwh"})
)

subsys_daily["year"] = subsys_daily["date"].dt.year
subsys_daily["month"] = subsys_daily["date"].dt.month
subsys_daily["day"] = subsys_daily["date"].dt.day

# Step 2: monthly energy -> average MW
monthly = (
    subsys_daily.groupby(["subsystem", "year", "month"])
    .agg({**{"energy_mwh": "sum"}, **{feat: "mean" for feat in WEATHER_FEATURES}})
    .reset_index()
)

monthly["hours"] = monthly.apply(lambda r: month_hours(r["year"], r["month"]), axis=1)
monthly["avg_mw"] = monthly["energy_mwh"] / monthly["hours"]
monthly.sort_values(["subsystem", "year", "month"], inplace=True)


base_year = monthly["year"].min()

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = (df["year"] - base_year) * 12 + (df["month"] - 1)
    df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

monthly = add_time_features(monthly)

# annual trend per subsystem (linear fit of yearly averages)
annual_means = (
    monthly.groupby(["subsystem", "year"])["avg_mw"]
    .mean()
    .reset_index()
)
annual_trend = {}
for subsys, sub_df in annual_means.groupby("subsystem"):
    if len(sub_df) >= 2:
        slope = np.polyfit(sub_df["year"], sub_df["avg_mw"], 1)[0]
    else:
        slope = 0.0
    annual_trend[subsys] = slope * args.trend_weight
last_hist_year = annual_means["year"].max()
default_val_year = args.start_year - 1
if default_val_year < base_year:
    default_val_year = last_hist_year
val_start_year = (
    args.val_start_year if args.val_start_year is not None else min(last_hist_year, default_val_year)
)
val_end_year = args.val_end_year if args.val_end_year is not None else val_start_year

# fill missing weather values with subsystem-month climatology, then global mean
for feat in WEATHER_FEATURES:
    monthly[feat] = monthly.groupby(["subsystem", "month"])[feat].transform(
        lambda x: x.fillna(x.mean())
    )
    if monthly[feat].isna().any():
        monthly[feat] = monthly[feat].fillna(monthly[feat].mean())

# Monthly climatology for weather features
climatology = (
    monthly.groupby(["subsystem", "month"])[WEATHER_FEATURES]
    .mean()
    .reset_index()
    .set_index(["subsystem", "month"])
)

future_months = []
for year in range(args.start_year, args.end_year + 1):
    for month in range(1, 13):
        future_months.append({"year": year, "month": month})
future_df_template = pd.DataFrame(future_months)

feature_names = ["t", "sin_m", "cos_m", "lag1", "lag12"] + WEATHER_FEATURES

def build_model(model_type: str):
    if model_type == "ridge":
        return Ridge(alpha=args.alpha)
    return HistGradientBoostingRegressor(
        learning_rate=args.hgb_learning_rate,
        max_depth=args.hgb_max_depth,
        max_leaf_nodes=args.hgb_max_leaf_nodes,
        random_state=args.seed,
    )


def predict_single(model, values: list[float]) -> float:
    """Predict using a fitted model with feature names preserved."""
    feature_df = pd.DataFrame([values], columns=feature_names)
    return float(model.predict(feature_df)[0])

all_rows_main_model = []
val_metrics = []
MODEL_TYPES = ["ridge", "hgb"]

for subsys, sub_df in monthly.groupby("subsystem"):
    sub_df = sub_df.reset_index(drop=True)
    sub_df["lag1"] = sub_df["avg_mw"].shift(1)
    sub_df["lag12"] = sub_df["avg_mw"].shift(12)
    train_df = sub_df.dropna(subset=["lag1", "lag12"])

    if train_df.empty:
        continue

    X_all = train_df[feature_names]
    y_all = train_df["avg_mw"].values
    for model_type in MODEL_TYPES:
        model_full = build_model(model_type)
        model_full.fit(X_all, y_all)

        model_eval = None
        if args.val_leakage:
            model_eval = model_full
        elif val_start_year is not None:
            pre_val_df = train_df[train_df["year"] < val_start_year]
            if not pre_val_df.empty:
                model_eval = build_model(model_type)
                model_eval.fit(pre_val_df[feature_names], pre_val_df["avg_mw"].values)

        history = sub_df[["year", "month", "avg_mw"]].copy().values.tolist()
        hist_values = [row[2] for row in history]

        if model_eval is not None:
            val_mask = (sub_df["year"] >= val_start_year) & (sub_df["year"] <= val_end_year)
            val_rows = sub_df[val_mask].copy()
            if not val_rows.empty:
                eval_history = sub_df[sub_df["year"] < val_start_year]["avg_mw"].tolist()
                if eval_history or args.val_leakage:
                    preds_v = []
                    actuals = []
                    for _, row in val_rows.iterrows():
                        year = int(row["year"])
                        month = int(row["month"])
                        t = (year - base_year) * 12 + (month - 1)
                        sin_m = np.sin(2 * np.pi * month / 12)
                        cos_m = np.cos(2 * np.pi * month / 12)
                        if args.val_leakage:
                            lag1 = row["lag1"]
                            lag12 = row["lag12"]
                            if pd.isna(lag1) or pd.isna(lag12):
                                continue
                        else:
                            lag1 = eval_history[-1] if eval_history else row["avg_mw"]
                            lag12 = eval_history[-12] if len(eval_history) >= 12 else eval_history[0]
                        wx_vals = [row[feat] for feat in WEATHER_FEATURES]
                        features = [t, sin_m, cos_m, lag1, lag12] + wx_vals
                        pred = predict_single(model_eval, features)
                        pred = max(pred, 0.0)
                        preds_v.append(pred)
                        actuals.append(row["avg_mw"])
                        if not args.val_leakage:
                            eval_history.append(pred)
                    if preds_v:
                        mae = mean_absolute_error(actuals, preds_v)
                        r2 = r2_score(actuals, preds_v)
                        weight = float(np.sum(actuals))
                        val_metrics.append(
                            {
                                "model": model_type,
                                "subsystem": subsys,
                                "val_start_year": val_start_year,
                                "val_end_year": val_end_year,
                                "mae": mae,
                                "r2": r2,
                                "weight": weight,
                            }
                        )

        last_year = int(sub_df["year"].iloc[-1])
        last_month = int(sub_df["month"].iloc[-1])
        future_df = future_df_template[
            (future_df_template["year"] > last_year)
            | (
                (future_df_template["year"] == last_year)
                & (future_df_template["month"] > last_month)
            )
        ].copy()

        preds = []
        history_values = hist_values.copy()
        for _, row in future_df.iterrows():
            year = int(row["year"])
            month = int(row["month"])
            t = (year - base_year) * 12 + (month - 1)
            sin_m = np.sin(2 * np.pi * month / 12)
            cos_m = np.cos(2 * np.pi * month / 12)
            lag1 = history_values[-1] if history_values else sub_df["avg_mw"].iloc[-1]
            lag12 = history_values[-12] if len(history_values) >= 12 else history_values[0]
            if (subsys, month) in climatology.index:
                wx_vals = climatology.loc[(subsys, month)]
            else:
                wx_vals = {feat: sub_df[feat].mean() for feat in WEATHER_FEATURES}
            features = [t, sin_m, cos_m, lag1, lag12] + [wx_vals[feat] for feat in WEATHER_FEATURES]
            pred = predict_single(model_full, features)
            pred = max(pred, 0.0)
            if year > last_hist_year:
                trend_years = year - last_hist_year
                pred += annual_trend.get(subsys, 0.0) * trend_years
                pred = max(pred, 0.0)
            history_values.append(pred)
            preds.append(
                {
                    "year": year,
                    "month": month,
                    "subsystem": subsys,
                    "predicted_avg_mw": pred,
                }
            )

        if preds and model_type == args.model:
            all_rows_main_model.append(pd.DataFrame(preds))

if not all_rows_main_model:
    raise RuntimeError("No subsystems produced forecasts.")

output_df = pd.concat(all_rows_main_model, ignore_index=True)
noise = np.random.normal(0.0, args.noise_std, size=len(output_df))
output_df["predicted_avg_mw"] = np.maximum(
    output_df["predicted_avg_mw"] * (1 + noise), 0.0
)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"Saved generation forecast to {output_path} using model '{args.model}'")
if val_metrics:
    metrics_df = pd.DataFrame(val_metrics)
    print(f"Validation metrics for {val_start_year}-{val_end_year} (both models):")
    metrics_df = metrics_df.sort_values(["subsystem", "model"])
    print(metrics_df.drop(columns=["weight"]).to_string(index=False))
    for model_type in MODEL_TYPES:
        mdf = metrics_df[metrics_df["model"] == model_type]
        total_weight = mdf["weight"].sum()
        if total_weight > 0:
            weighted_mae = (mdf["mae"] * mdf["weight"]).sum() / total_weight
            weighted_r2 = (mdf["r2"] * mdf["weight"]).sum() / total_weight
            print(
                f"Weighted metrics for {model_type}: MAE={weighted_mae:.2f}, R2={weighted_r2:.3f}"
            )
else:
    print("Validation metrics unavailable (insufficient pre-validation data).")
