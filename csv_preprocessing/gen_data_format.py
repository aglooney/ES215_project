import pandas as pd
import os

input_csv = "data/generation_data/GERACAO_USINA-2_2024_10.csv"

df = pd.read_csv(input_csv, sep=";")

new_cols = ["date","subsys_id","subsys_name","state_id","state_name","plant_opn_mode","plant_type","fuel_type","plant_name","ons_id","ceg","gen_val(MW)"]

df.columns = new_cols

output_csv = "data/generation_data/generation_2024oct.csv"

df.to_csv(output_csv, index=False)


os.remove(input_csv)

print('Done Processing')