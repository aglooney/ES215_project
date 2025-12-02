import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor



df = pd.read_csv("data/merged_generation_weather_v2.csv")

df = df[df['date']<"2024-03-01"]

df = df.dropna()

plant_types = df['plant_type'].unique()


weather_features = ["ETo", "RH", "Rs", "Tmin", "Tmax", "pr", "u2"]
target = "gen_val(MW)"



tscv = TimeSeriesSplit(n_splits=5)

df = df.sort_values("date")

split_date = "2023-06-01"

print(df.groupby('plant_type')['gen_val(MW)'].describe())



results = []

for ptype in plant_types:
    subset = df[df['plant_type']==ptype].sort_values("date")

    train = subset[subset['date'] < split_date]
    test = subset[subset['date'] >= split_date]

    if train.shape[0] == 0 or test.shape[0] == 0:
        print(f'Not sufficient data for type: {ptype}')
        continue

    X_train = train[weather_features]
    X_test = test[weather_features]
    y_train = train[target]
    y_test = test[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)


    y_preds = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)

    results.append({"type": ptype, "MSE": mse, "r-squared": r2})


print(results)

