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
    
print('-------------------------------------------------------')
print('-------------------------------------------------------')

# Testing data processing
# =====================================================================
# Testing data processing
# Paste below the exploration part of iteration2.py (needs `df`).
#
# Schema taken from the dataset's Kaggle listing: 22 columns, ~1,842 rows,
# pooled from three cerebral small vessel disease cohorts (SCANS, RUN DMC,
# ASPS). The value encodings were NOT checked against the real file, so
# Test 0 verifies them before anything else runs.
# =====================================================================
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

OUTCOME = "dementia_all"                  # switch to "dementia" if Test 1 says so
COGNITIVE = ["ef", "ps", "global"]        # executive function, processing speed, global cognition
CLINICAL = ["age", "educationyears", "diabetes", "hypertension",
            "hypercholesterolemia", "smoking"]
MRI = ["svd_simple_score", "fazekas", "lac_count", "cmb_count"]

expected = ["id", "study", "gender", "dementia", OUTCOME] + COGNITIVE + CLINICAL + MRI
missing_cols = [c for c in expected if c not in df.columns]
if missing_cols:
    raise KeyError(f"Columns not found: {missing_cols}\nActual columns: {list(df.columns)}")


def header(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


# ---------------------------------------------------------------------
# TEST 0: Can every column we need be turned into numbers without
# silently losing values? (Several are stored as text, e.g. hypertension.)
# Expectation: 0 values lost per column. If not, inspect the printed
# leftovers and extend YES_NO.
# ---------------------------------------------------------------------
header("TEST 0: converting text columns to numbers")
YES_NO = {"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0,
          "current": 1, "ever": 1, "never": 0}

for col in [OUTCOME, "dementia"] + COGNITIVE + CLINICAL + MRI:
    raw = df[col]
    converted = pd.to_numeric(raw, errors="coerce")
    text_mask = converted.isna() & raw.notna()
    converted[text_mask] = raw[text_mask].astype(str).str.strip().str.lower().map(YES_NO)
    lost = converted.isna() & raw.notna()
    if lost.any():
        print(f"{col}: {lost.sum()} values could not be converted -> {raw[lost].unique()[:10]}")
    df[col] = converted
print("Done. Anything printed above this line needs a look.")


# ---------------------------------------------------------------------
# TEST 1: Basic sanity
# Expectations: no duplicate (study, id) pairs; ages roughly 40-100;
# education 0-30 years; the two outcome columns agree wherever both exist.
# ---------------------------------------------------------------------
header("TEST 1: sanity checks")
dupes = df.duplicated(subset=["study", "id"]).sum()
print(f"Duplicate (study, id) rows: {dupes}")

for col, lo, hi in [("age", 40, 100), ("educationyears", 0, 30)]:
    out = df[(df[col] < lo) | (df[col] > hi)]
    print(f"{col}: {len(out)} rows outside {lo}-{hi}  (range {df[col].min()} to {df[col].max()})")

print("\nOutcome columns compared (rows: dementia, columns: dementia_all):")
print(pd.crosstab(df["dementia"], df["dementia_all"], dropna=False))

print(f"\nDementia cases per study ({OUTCOME}):")
print(df.groupby("study")[OUTCOME].agg(["count", "sum", "mean"]).round(3))

# Are cognitive scores z-scores (mean ~0, sd ~1)? Matters for reading Test 2.
print("\nCognitive scores per study (z-scored if mean ~0 and std ~1):")
print(df.groupby("study")[COGNITIVE].agg(["mean", "std"]).round(2))


# ---------------------------------------------------------------------
# TEST 2: Which cognitive domain separates future dementia cases most?
# Effect size d = gap between group means, measured in standard
# deviations (~0.2 small, ~0.5 medium, ~0.8 large).
# Expectation: cases score lower on all three. The domain with the
# biggest gap is the strongest candidate for a Pixie gameplay proxy
# (e.g. processing speed <-> reaction time to a game prompt).
# ---------------------------------------------------------------------
header("TEST 2: cognitive gap between dementia and no dementia")


def cohens_d(a, b):
    a, b = a.dropna(), b.dropna()
    pooled_sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_sd


cases, controls = df[df[OUTCOME] == 1], df[df[OUTCOME] == 0]
rows = []
for col in COGNITIVE:
    rows.append({"measure": col,
                 "mean_dementia": cases[col].mean(),
                 "mean_no_dementia": controls[col].mean(),
                 "cohens_d": cohens_d(cases[col], controls[col])})
print(pd.DataFrame(rows).round(2).to_string(index=False))


# ---------------------------------------------------------------------
# TEST 3: Does the Test 2 pattern hold in every study separately?
# Expectation: same direction in each cohort. If one study flips,
# the pooled result is driven by differences between studies, not
# by dementia itself.
# ---------------------------------------------------------------------
header("TEST 3: cognitive gap per study")
per_study = {}
for study, g in df.groupby("study"):
    per_study[study] = {col: cohens_d(g.loc[g[OUTCOME] == 1, col],
                                      g.loc[g[OUTCOME] == 0, col]) for col in COGNITIVE}
print(pd.DataFrame(per_study).T.round(2))


# ---------------------------------------------------------------------
# TEST 4: Is it just age? Older people are slower AND more likely to
# develop dementia. Compare within age bands.
# Expectation: the gap shrinks but stays inside each band. If it
# vanishes, a Pixie metric would mostly be measuring age.
# ---------------------------------------------------------------------
header("TEST 4: processing speed gap within age bands")
df["age_band"] = pd.cut(df["age"], bins=[0, 60, 70, 80, 120],
                        labels=["<60", "60-69", "70-79", "80+"])
band = df.groupby(["age_band", OUTCOME], observed=True)["ps"].agg(["count", "mean"]).round(2)
print(band.unstack(OUTCOME))


# ---------------------------------------------------------------------
# TEST 5: How far do cognitive scores alone get you?
# AUC = chance that a random future case gets a higher risk score
# than a random non-case (0.5 = coin flip, 1.0 = perfect).
# Expectation (from the original paper): clinical + MRI predicts best.
# The interesting number is "cognitive only": that is the ceiling of
# what a Pixie-style cognitive proxy could add.
# ---------------------------------------------------------------------
header("TEST 5: prediction with different feature sets (5-fold cross-validated AUC)")
feature_sets = {
    "cognitive only": COGNITIVE,
    "clinical only": CLINICAL,
    "cognitive + clinical": COGNITIVE + CLINICAL,
    "cognitive + clinical + MRI": COGNITIVE + CLINICAL + MRI,
}
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, cols in feature_sets.items():
    data = df[cols + [OUTCOME]].dropna()        # complete cases only; n is reported
    X, y = data[cols], data[OUTCOME].astype(int)
    if y.sum() < 10:
        print(f"{name:28s} skipped: only {y.sum()} cases left after dropping missing rows")
        continue
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"{name:28s} AUC {auc.mean():.2f} (+/- {auc.std():.2f})   n={len(data)}, cases={y.sum()}")