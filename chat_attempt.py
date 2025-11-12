#ETo: proxy for overall atmosphereic energy
#RH: relative humidity
#Rs: solar radiation
#Tmin: minimum temperature
#Tmax: maximum temperature
#pr: precipitation
#u2: wind speed
"""
Temporal Cross-Validation for Brazil Generation Prediction
----------------------------------------------------------
Evaluates weather-driven models (Linear Regression & Random Forest)
using time-ordered (rolling-origin) cross-validation per plant type.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# CONFIGURATION
# =====================================================
DATA_PATH = "data/merged_generation_weather.csv"
DATE_COL = "date"
PLANT_ID_COL = "ceg"
PLANT_TYPE_COL = "plant_type"
TARGET_COL = "gen_val(MW)"

# Weather features available in your dataset
weather_features = ["ETo", "RH", "Rs", "Tmin", "Tmax", "pr", "u2"]

# Random forest hyperparameters
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

N_SPLITS = 5  # number of temporal CV folds

# =====================================================
# LOAD AND PREPARE DATA
# =====================================================
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).dropna(subset=[TARGET_COL, PLANT_ID_COL, PLANT_TYPE_COL])

# Filter to columns that actually exist
weather_features = [f for f in weather_features if f in df.columns]
if not weather_features:
    raise ValueError("No valid weather features found in dataset.")

# Optional: target encoding of CEG mean (fixed effect)
ceg_mean_map = df.groupby(PLANT_ID_COL)[TARGET_COL].mean().to_dict()
df["ceg_encoded"] = df[PLANT_ID_COL].map(ceg_mean_map)

# Full feature set
X_cols = weather_features + ["ceg_encoded"]

# Fill NaNs using median imputation
df[X_cols] = df[X_cols].fillna(df[X_cols].median(numeric_only=True))

# =====================================================
# DEFINE TIME SERIES CV FUNCTION
# =====================================================
def temporal_cv(subset: pd.DataFrame, X_cols: list, y_col: str, n_splits: int = 5):
    """
    Perform time-series cross-validation for one generation type.
    Returns average MSE and R² across folds for Linear Regression and Random Forest.
    """
    subset = subset.sort_values(DATE_COL).copy()
    X = subset[X_cols].values
    y = subset[y_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Linear Regression (scaled)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lin = LinearRegression()
        lin.fit(X_train_s, y_train)
        y_pred_lin = lin.predict(X_test_s)
        mse_lin = mean_squared_error(y_test, y_pred_lin)
        r2_lin = r2_score(y_test, y_pred_lin)

        # Random Forest (unscaled)
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)

        metrics.append({
            "fold": fold,
            "mse_linear": mse_lin,
            "r2_linear": r2_lin,
            "mse_rf": mse_rf,
            "r2_rf": r2_rf
        })

        print(f"    Fold {fold}: R²_linear={r2_lin:.3f}, R²_rf={r2_rf:.3f}")

    # Average across folds
    avg = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()
    avg["r2_std_linear"] = np.std([m["r2_linear"] for m in metrics])
    avg["r2_std_rf"] = np.std([m["r2_rf"] for m in metrics])
    return avg


# =====================================================
# APPLY CV PER PLANT TYPE
# =====================================================
results = []

for ptype, subset in df.groupby(PLANT_TYPE_COL):
    if subset.shape[0] < 100:
        print(f"⚠️  Skipping {ptype}: not enough data ({len(subset)} rows).")
        continue

    print(f"\n🔹 Running temporal CV for {ptype} ({len(subset)} records)")
    avg_metrics = temporal_cv(subset, X_cols, TARGET_COL, N_SPLITS)
    avg_metrics["plant_type"] = ptype
    avg_metrics["n_obs"] = len(subset)
    results.append(avg_metrics)

# =====================================================
# RESULTS SUMMARY
# =====================================================
results_df = pd.DataFrame(results).sort_values("r2_rf", ascending=False)

cols = [
    "plant_type", "n_obs",
    "mse_linear", "r2_linear", "r2_std_linear",
    "mse_rf", "r2_rf", "r2_std_rf"
]
print("\n=== Temporal Cross-Validation Results ===")
print(results_df[cols].to_string(index=False, float_format="%.4f"))

# Optional: Save results
Path("models/cv_results").mkdir(parents=True, exist_ok=True)
results_df.to_csv("models/cv_results/time_series_cv_results.csv", index=False)
print("\n✅ Results saved to models/cv_results/time_series_cv_results.csv")
