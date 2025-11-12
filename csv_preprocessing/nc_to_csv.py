import xarray as xr
import pandas as pd
import os

def nc_to_pandas(nc_path, save_csv=True):
    """
    Convert a NetCDF (.nc) file into a pandas DataFrame.
    
    Parameters
    ----------
    nc_path : str
        Path to the .nc file.
    save_csv : bool, optional
        If True, saves the DataFrame as a .csv file with the same name.
    
    Returns
    -------
    pd.DataFrame
        Flattened DataFrame with coordinates and variable values.
    """
    # Load dataset
    ds = xr.open_dataset(nc_path)

    # Filter time dimension (assuming your time coordinate is called 'time')
    ds_filtered = ds.sel(time=slice("2020-01-01", "2025-09-30"))

    # Convert to DataFrame
    df = ds_filtered.to_dataframe().reset_index()

    df = df.dropna()

    print(f"Loaded {nc_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Variables: {list(ds.data_vars)}")

    if save_csv:
        out_path = nc_path.replace(".nc", ".csv")
        df.to_csv(out_path, index=False)
        print(f"Saved CSV: {out_path}")

    return df

if __name__ == "__main__":
    for file in os.listdir("data/weather_data"):
        if file.endswith(".nc"):
            nc_file = f"data/weather_data/{file}"
            print(f"Now Processing: {nc_file}")
            nc_to_pandas(nc_file)
