import pandas as pd

input_csv = "data/all_generation_data_v2.csv"

df = pd.read_csv(input_csv)

def convert_latlon(x):
    """
    Converts messy latitude/longitude values to decimal degrees.
    Handles:
      - Commas as decimal separators
      - Fixed-point integers (e.g., -22345678 -> -22.345678)
      - Invalid inputs (returns pd.NA)
    """
    import pandas as pd
    
    if pd.isna(x):
        return pd.NA

    # Replace comma with period and strip whitespace
    x_str = str(x).replace(",", ".").strip()

    try:
        x_float = float(x_str)
    except ValueError:
        return pd.NA

    # Handle fixed-point formats (e.g., -22345678)
    abs_val = abs(x_float)
    if abs_val > 1000:  # likely fixed-point
        digits = len(str(int(abs_val)))
        # assume 2 degree digits if more than 6 digits total
        scale = 10 ** (digits - 2)
        x_float = x_float / scale

    return x_float

# Convert latitude
df["latitude"] = df["latitude"].apply(convert_latlon)

# Convert longitude
df["longitude"] = df["longitude"].apply(convert_latlon)

df.to_csv("data/all_generation_data_v2.csv", index=False)