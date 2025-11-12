import pandas as pd

df = pd.read_csv("data/generation_coordinates.csv")
print("Dataframe loaded")

def apply_modification(entry):
    return entry[:-1] + "0" + entry[-1]

df['ceg'] = df['CEG'].apply(apply_modification)

df = df.drop(columns=["CEG"])

df.to_csv("data/generation_coordinates_fixed.csv", index=False)

print("Completed Adjustment")