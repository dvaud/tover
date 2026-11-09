# What datapoints should the Tover Pixie gather
# to be able to correlate real user data with medical dementia prediction datasets?
#
# Iteration 3: easySHARE (https://share-eric.eu/data/data-set-details/easyshare)
# A simplified version of SHARE, the Survey of Health, Ageing and Retirement in
# Europe: people aged 50+ in ~28 countries, interviewed every ~2 years since 2004.
# One row per person per wave ("long format"), ~410,000 rows, ~110 columns.
#
# GETTING THE DATA
# easySHARE cannot be downloaded from code like the Kaggle sets: it needs a free
# SHARE user account (https://share-eric.eu/data/become-a-user). After approval,
# download the Stata version from the SHARE Research Data Center, unzip it and
# put the .dta file (e.g. easySHARE_rel9-0-0.dta) in datastic/data/, or set
#   EASYSHARE_PATH=/full/path/to/the/file.dta
# The SPSS .sav file also works but needs `pip install pyreadstat`.
# SHARE's conditions of use forbid passing the data on, so datastic/data/ is
# git-ignored. Do not commit the file.

import os
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 300)
pd.set_option("display.width", None)

DATA_DIR = Path(__file__).resolve().parent / "data"


def find_data_file():
    if os.environ.get("EASYSHARE_PATH"):
        return Path(os.environ["EASYSHARE_PATH"])
    hits = sorted(p for p in DATA_DIR.glob("*")
                  if p.name.lower().startswith("easyshare") and p.suffix.lower() in (".dta", ".sav"))
    if not hits:
        raise FileNotFoundError(
            f"No easySHARE file found in {DATA_DIR}.\n"
            "Download it from SHARE (account needed, see the top of this file), "
            "put the .dta file there or set EASYSHARE_PATH.")
    return hits[0]


path = find_data_file()
print("Loading", path)
if path.suffix.lower() == ".dta":
    # Keep the raw numbers: easySHARE stores the reason a value is missing as a
    # negative code (-13 = "not asked in this wave" etc.), and text labels would hide that.
    df = pd.read_stata(path, convert_categoricals=False)
    with pd.read_stata(path, iterator=True) as reader:
        var_labels = reader.variable_labels()
    country_names = pd.read_stata(path, columns=["country"])["country"].astype(str)
else:
    df = pd.read_spss(path, convert_categoricals=False)
    var_labels = {}
    country_names = pd.read_spss(path, usecols=["country"])["country"].astype(str)
df["country_name"] = country_names.values

print("Entries (rows, columns):", df.shape)
print("Different people (mergeid):", df["mergeid"].nunique())

print("\nColumns, types and labels:")
labels = pd.Series(var_labels, dtype=object).reindex(df.columns).fillna("")
print(pd.DataFrame({"type": df.dtypes.astype(str), "label": labels}))

# Missing values. Release guide, section "Missing codes".
MISSING_CODES = {
    -3: "implausible value", -4: "not codable", -7: "not yet coded",
    -9: "not applicable (filtered)", -10: "SHARELIFE interview (wave 7)",
    -11: "regular interview (wave 7)", -12: "don't know / refusal",
    -13: "not asked in this wave", -14: "not asked in this country",
    -15: "no information", -16: "no drop-off questionnaire",
}
num_cols = df.select_dtypes(include="number").columns
code_counts = pd.DataFrame({col: df[col].value_counts().reindex(list(MISSING_CODES), fill_value=0)
                            for col in num_cols}).T
code_counts["NaN"] = df[num_cols].isna().sum()
code_counts["% missing"] = (code_counts.sum(axis=1) / len(df) * 100).round(1)
code_counts = code_counts.loc[:, (code_counts != 0).any()]   # drop codes that never occur

print("\nMissing-value codes:")
for code, label in MISSING_CODES.items():
    print(f"  {code}: {label}")
print("\nMissing values per column, by reason (only columns with any):")
print(code_counts[code_counts["% missing"] > 0].sort_values("% missing", ascending=False))

# Text columns are identifiers, so print how many distinct values they have
for col in [c for c in df.columns if c not in num_cols and c != "country_name"]:
    print(f"\n--- {col} --- {df[col].nunique()} distinct values, e.g. {df[col].iloc[0]!r}")

for col in ["wave", "country_name", "female", "int_version"]:
    print(f"\n--- {col} ---")
    counts = df[col].value_counts(dropna=False)
    print(counts.sort_index() if col == "wave" else counts)

# From here on every missing code counts as plain missing
df[num_cols] = df[num_cols].mask(df[num_cols].isin(list(MISSING_CODES)))

print('-------------------------------------------------------')
print('-------------------------------------------------------')

# =====================================================================
# Testing data processing
#
# easySHARE has NO dementia diagnosis column. SHARE asks about Alzheimer's
# in waves 4-9, but easySHARE leaves it out of chronic_mod and has no column
# for it. So instead of "dementia yes/no" these tests look at:
#   - difficulty with the "cognitive" daily tasks (phone calls, taking
#     medication, managing money = iadla) as the closest case marker, and
#   - drop in word recall between two interviews of the same person.
# Ranges and codings come from the easySHARE 9.0.0 release guide; Test 0
# checks them against the real file before anything else runs.
# =====================================================================
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COGNITIVE = ["recall_1", "recall_2", "orienti", "numeracy_1", "numeracy_2"]
BACKGROUND = ["age", "female", "eduyears_mod", "chronic_mod"]
# Things a Pixie could plausibly notice in daily life
EVERYDAY = ["adla",          # basic self-care difficulties (dressing, eating, ...)
            "mobilityind",   # walking / stairs difficulties
            "finemotor",     # picking up a coin, cutting food, dressing
            "br015_",        # vigorous activity (1 = more than weekly ... 4 = hardly ever)
            "eurod",         # depression symptoms 0-12
            "euro5",         # sleep problems
            "sp008_"]        # gives help to others outside the household (social contact)
CASE_MARKER = "iadla"        # 0-3 difficulties with phone, medication, money

expected = ["mergeid", "wave", "int_year", CASE_MARKER] + COGNITIVE + BACKGROUND + EVERYDAY
missing_cols = [c for c in expected if c not in df.columns]
if missing_cols:
    raise KeyError(f"Columns not found: {missing_cols}\nActual columns: {list(df.columns)}")

df["sp008_"] = df["sp008_"].replace({5: 0})   # 1 yes / 5 no  ->  1 / 0


def header(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def cohens_d(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_sd


# ---------------------------------------------------------------------
# TEST 0: Do values sit inside the ranges the release guide promises?
# Expectation: 0 values outside for every column. Also checks which way
# orientation is scored: the guide says 0 = good, 4 = bad, which would
# make it the only cognitive score where high is bad. If it correlates
# negatively with word recall the guide is right and we flip it, so that
# from here on high = better for every cognitive measure.
# ---------------------------------------------------------------------
header("TEST 0: value ranges from the release guide")
RANGES = {"recall_1": (0, 10), "recall_2": (0, 10), "orienti": (0, 4),
          "numeracy_1": (1, 5), "numeracy_2": (0, 5),   # numeracy_2 range not in guide: assumed
          "iadla": (0, 3), "adla": (0, 5), "mobilityind": (0, 4), "finemotor": (0, 3),
          "br015_": (1, 4), "eurod": (0, 12), "euro5": (0, 1), "sp008_": (0, 1),
          "female": (0, 1), "chronic_mod": (0, 11)}
rows = []
for col, (lo, hi) in RANGES.items():
    s = df[col].dropna()
    rows.append({"column": col, "expected": f"{lo}-{hi}", "n": len(s),
                 "min": s.min(), "max": s.max(), "outside": int(((s < lo) | (s > hi)).sum())})
print(pd.DataFrame(rows).to_string(index=False))

r = df["orienti"].corr(df["recall_2"])
print(f"\nCorrelation orientation vs delayed recall: {r:.2f}")
if r < 0:
    df["orienti"] = 4 - df["orienti"]
    print("Negative: guide is right (0 = good). Flipped, so 4 = fully oriented from here on.")
else:
    print("Positive: high already means better orientation. Left as is (guide text is wrong).")


# ---------------------------------------------------------------------
# TEST 1: Basic sanity
# Expectations: no duplicate (mergeid, wave) rows; nearly everyone 50+
# (younger people are partners of respondents); education 0-25 years;
# no cognitive data in wave 3 (SHARELIFE life-history wave); numeracy_1
# mostly in early waves and numeracy_2 only from wave 4.
# ---------------------------------------------------------------------
header("TEST 1: sanity checks")
print(f"Duplicate (mergeid, wave) rows: {df.duplicated(subset=['mergeid', 'wave']).sum()}")
print(f"age: range {df['age'].min()} to {df['age'].max()}, "
      f"{(df['age'] < 50).mean() * 100:.1f}% under 50")
out = df[(df["eduyears_mod"] < 0) | (df["eduyears_mod"] > 25)]
print(f"eduyears_mod: {len(out)} rows outside 0-25 "
      f"(range {df['eduyears_mod'].min()} to {df['eduyears_mod'].max()})")

print("\nInterviews per person (how many people took part in 1, 2, ... waves):")
print(df.groupby("mergeid").size().value_counts().sort_index())

print("\nNon-missing cognitive values per wave:")
print(df.groupby("wave")[COGNITIVE + [CASE_MARKER]].count())

# One row per person from here to Test 4 (their latest interview with word
# recall), so people interviewed eight times do not count eight times.
latest = (df.dropna(subset=["recall_2", CASE_MARKER])
            .sort_values("wave").groupby("mergeid").tail(1).copy())
latest["case"] = (latest[CASE_MARKER] >= 1).astype(int)
print(f"\nPeople with recall and {CASE_MARKER} data: {len(latest)}, "
      f"of whom {latest['case'].sum()} ({latest['case'].mean() * 100:.1f}%) "
      f"report >= 1 difficulty with phone / medication / money")


# ---------------------------------------------------------------------
# TEST 2: Which cognitive score separates people who struggle with
# phone calls, medication or money from those who don't?
# Effect size d = gap between group means in standard deviations
# (~0.2 small, ~0.5 medium, ~0.8 large). Negative = cases score lower.
# Expectation: cases lower on all scores; delayed recall the biggest gap,
# as memory loss is usually the first sign of Alzheimer's.
# ---------------------------------------------------------------------
header(f"TEST 2: cognitive gap, {CASE_MARKER} >= 1 vs 0 (latest interview per person)")
cases, controls = latest[latest["case"] == 1], latest[latest["case"] == 0]
rows = []
for col in COGNITIVE:
    rows.append({"measure": col,
                 "mean_difficulty": cases[col].mean(),
                 "mean_no_difficulty": controls[col].mean(),
                 "cohens_d": cohens_d(cases[col], controls[col]),
                 "n": latest[col].notna().sum()})
print(pd.DataFrame(rows).round(2).to_string(index=False))


# ---------------------------------------------------------------------
# TEST 3: Does the Test 2 pattern hold in every country?
# Expectation: negative d everywhere. Big differences between countries
# would mean the Pixie needs country- or language-specific baselines
# (word lists differ by language).
# ---------------------------------------------------------------------
header("TEST 3: cognitive gap per country (cohens_d)")
per_country = {}
for country, g in latest.groupby("country_name"):
    per_country[country] = {col: cohens_d(g.loc[g["case"] == 1, col], g.loc[g["case"] == 0, col])
                            for col in ["recall_1", "recall_2", "orienti"]}
    per_country[country]["cases"] = int(g["case"].sum())
print(pd.DataFrame(per_country).T.sort_values("recall_2").round(2))


# ---------------------------------------------------------------------
# TEST 4: Is it just age? Older people recall fewer words AND struggle
# more with daily tasks. Compare within age bands.
# Expectation: the gap shrinks but stays inside each band. If it
# vanishes, a Pixie memory metric would mostly be measuring age.
# ---------------------------------------------------------------------
header("TEST 4: delayed recall gap within age bands")
latest["age_band"] = pd.cut(latest["age"], bins=[0, 60, 70, 80, 120],
                            labels=["<60", "60-69", "70-79", "80+"], right=False)
band = latest.groupby(["age_band", "case"], observed=True)["recall_2"].agg(["count", "mean"]).round(2)
band = band.unstack("case")
band["cohens_d"] = [cohens_d(g.loc[g["case"] == 1, "recall_2"], g.loc[g["case"] == 0, "recall_2"])
                    for _, g in latest.groupby("age_band", observed=True)]
print(band.round(2))


# ---------------------------------------------------------------------
# TEST 5: Can we predict who loses memory before the next interview?
# This is where easySHARE beats the Kaggle sets: the same people are
# followed over years. "Declined" = total word recall (immediate +
# delayed, 0-20) falls by DECLINE_WORDS or more at the next interview.
# AUC = chance that a random decliner gets a higher risk score than a
# random non-decliner (0.5 = coin flip, 1.0 = perfect). Folds are split
# by person, so nobody is in both training and test data.
# Note: people who score high now have more room to fall, so "cognition
# now" partly predicts decline for that reason alone. The interesting
# number is how much "everyday" adds on top of background + cognition.
# ---------------------------------------------------------------------
DECLINE_WORDS = 4
header(f"TEST 5: predicting a drop of >= {DECLINE_WORDS} recalled words by the next interview")
cog = df.dropna(subset=["recall_1", "recall_2"]).sort_values(["mergeid", "wave"]).copy()
cog["recall_total"] = cog["recall_1"] + cog["recall_2"]
nxt = cog.groupby("mergeid")
cog["next_total"] = nxt["recall_total"].shift(-1)
cog["years_to_next"] = nxt["int_year"].shift(-1) - cog["int_year"]
pairs = cog.dropna(subset=["next_total"]).copy()
pairs["declined"] = ((pairs["next_total"] - pairs["recall_total"]) <= -DECLINE_WORDS).astype(int)
print(f"Interview pairs: {len(pairs)} from {pairs['mergeid'].nunique()} people, "
      f"median {pairs['years_to_next'].median():.0f} years apart, "
      f"{pairs['declined'].mean() * 100:.1f}% declined")

NOW = ["recall_total", "orienti"]
feature_sets = {
    "background only": BACKGROUND,
    "cognition now only": NOW,
    "everyday only": EVERYDAY,
    "background + cognition now": BACKGROUND + NOW,
    "background + cognition + everyday": BACKGROUND + NOW + EVERYDAY,
}
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for name, cols in feature_sets.items():
    data = pairs[cols + ["declined", "mergeid"]].dropna()   # complete cases only; n is reported
    X, y = data[cols], data["declined"]
    if y.sum() < 10:
        print(f"{name:34s} skipped: only {y.sum()} decliners left after dropping missing rows")
        continue
    auc = cross_val_score(model, X, y, cv=cv, groups=data["mergeid"], scoring="roc_auc")
    print(f"{name:34s} AUC {auc.mean():.2f} (+/- {auc.std():.2f})   n={len(data)}, declined={y.sum()}")

# Which inputs carry the most weight in the full model? Inputs are
# standardised, so weights are comparable. Positive = raises decline risk.
cols = feature_sets["background + cognition + everyday"]
data = pairs[cols + ["declined"]].dropna()
model.fit(data[cols], data["declined"])
weights = pd.Series(model[-1].coef_[0], index=cols).sort_values(key=abs, ascending=False)
print("\nWeights in the full model (bigger size = more influence):")
print(weights.round(3).to_string())
