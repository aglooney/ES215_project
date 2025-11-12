import pandas as pd
import glob
import os

# Folder containing your Excel files
input_folder = "data"

# Get all .xlsx files in the folder
excel_files = glob.glob(os.path.join(input_folder, "*.xlsx"))

for file in excel_files:
    # Load the Excel file
    print(f"Processing {file}")
    try:
        df = pd.read_excel(file)
    except ValueError:
        print(f"Could not process {file}")
        continue
    # Define the output CSV path
    csv_path = file.replace(".xlsx", ".csv")
    
    # Save as CSV
    df.to_csv(csv_path, index=False)
    
    print(f"Converted: {file} → {csv_path}")

print("All Excel files converted to CSV.")
