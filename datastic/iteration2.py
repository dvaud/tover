# What datapoints should the Tover Pixie gather 
# to be able to correlate real user data with medical dementia prediction datasets?

# %%
import os
import kagglehub
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = kagglehub.dataset_download("fatemehmehrparvar/dementia")
files = os.listdir(path)
print(files)

csv_name = [f for f in files if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_name))



