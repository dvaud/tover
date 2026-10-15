from pathlib import Path
import pandas as pd
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")

df = pd.read_csv(Path(path) / "dementia_dataset.csv")
print(df.shape)
df.info()
print(df.head())