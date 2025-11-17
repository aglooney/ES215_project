import pandas as pd
import re
from collections import Counter

# Helper -----------------------------------------------------
def make_unique_columns(columns):
    counts = Counter()
    unique_cols = []
    for col in columns:
        counts[col] += 1
        if counts[col] == 1:
            unique_cols.append(col)
        else:
            unique_cols.append(f"{col}_{counts[col]-1}")
    return unique_cols


# ============================================
# 1. Load Excel with multi-row header
# ============================================
df = pd.read_excel("data/demand_data/demand_better_formatted.xlsx", header=[0, 1])

# ============================================
# 2. Flatten MultiIndex columns
# ============================================
flat_cols = []
for top, bottom in df.columns:
    # For first two columns:
    # ('Unnamed: 0_level_0', 'State') → 'state'
    # ('Unnamed: 1_level_0', 'Subsystem') → 'subsystem'
    if bottom == "State":
        flat_cols.append("state")
    elif bottom == "Subsystem":
        flat_cols.append("subsystem")
    else:
        # e.g. ('2004', 'JAN') → "2004_JAN"
        # e.g. ('2012', 'AGO.147') → "2012_AGO.147"
        flat_cols.append(f"{top}_{bottom}")

# Assign flattened columns first and make them unique
df.columns = make_unique_columns(flat_cols)

# ============================================
# 3. Identify all year-month columns
# ============================================
yearmonth_cols = [c for c in df.columns if "_" in c]

# ============================================
# 4. Clean the month names
# Remove trailing digits after periods: "AGO.147" → "AGO"
# ============================================
clean_mapping = {}
for col in yearmonth_cols:
    year, raw_month = col.split("_")

    # Remove everything after the first non-letter
    cleaned_month = re.match(r"[A-Za-z]+", raw_month).group(0)

    clean_mapping[col] = f"{year}_{cleaned_month}"

df = df.rename(columns=clean_mapping)

# Cleaning removed the suffixes that made columns unique, so fix duplicates again
df.columns = make_unique_columns(df.columns)
yearmonth_cols = [c for c in df.columns if "_" in c]

# ============================================
# 5. Convert comma formatting to numbers
# ============================================
for col in yearmonth_cols:
    # Skip columns that are already numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        continue

    series = df[col].astype(str)

    # Clean thousand separators and decimal commas when data is still string-formatted
    series = series.str.replace(".", "", regex=False)
    series = series.str.replace(",", ".", regex=False)

    df[col] = pd.to_numeric(series, errors="coerce")



# ============================================
# 6. Melt into long format
# ============================================
df_long = df.melt(
    id_vars=["state", "subsystem"],
    value_vars=yearmonth_cols,
    var_name="year_month",
    value_name="MWh"
)

# ============================================
# 7. Split year and month
# ============================================
df_long["year"] = df_long["year_month"].str.split("_").str[0].astype(int)
df_long["month_abbr"] = df_long["year_month"].str.split("_").str[1]

# PT → numeric month map
month_map = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4,
    "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
    "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12
}

df_long["month"] = df_long["month_abbr"].map(month_map)

# ============================================
# 8. Final tidy table
# ============================================
df_final = df_long[["state", "subsystem", "year", "month", "MWh"]] \
    .sort_values(["year", "month", "subsystem", "state"])

# ============================================
# 9. Save output
# ============================================
csv_path = "data/demand_data/demand_projection_clean.csv"
parquet_path = "data/demand_data/demand_projection_clean.parquet"

df_final.to_csv(csv_path, index=False)

try:
    df_final.to_parquet(parquet_path, index=False)
    parquet_msg = f" {parquet_path}"
except ImportError as exc:
    parquet_msg = f" skipped parquet export ({exc})"

print("Saved:")
print(f" {csv_path}")
print(parquet_msg)
print(df_final.head(20))
