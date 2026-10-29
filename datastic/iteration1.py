from pathlib import Path

import kagglehub
import pandas as pd

# Download latest versionx1
path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")

df = pd.read_csv(Path(path) / "dementia_dataset.csv")
print(df.shape)
df.info()
print(len(df))