# What datapoints should the Tover Pixie gather 
# to be able to correlate real user data with medical dementia prediction datasets?



import os
import kagglehub
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = kagglehub.dataset_download("fatemehmehrparvar/dementia")
csv_name = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_name))

print("Entries (rows, columns):", df.shape)
print("\nColumns and types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())

# Category counts for every text column
for col in df.select_dtypes(include="object").columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))