import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("datastic")
OUTPUT_FILE = Path("county_all_years.csv")

COLUMNS = [
    # identifiers
    "year", "state", "county",

    # dementia and cognitive severity
    "pctalz_dem_cty", "aggalz_dem_cty",
    "pctlowcfs_cty", "pctmidcfs_cty", "pcthighcfs_cty",     # current residents
    "agglowcfs_cty", "aggmidcfs_cty", "agghighcfs_cty",     # new admissions

    # dementia-specific care
    "alzunit_cty",

    # functioning
    "avgadl_mds3_cty", "nhcadl_mds3_cty", "pctwalking_cty",
    "pctbedft_mds3_cty", "pctfall30_mds3_cty",

    # wellbeing / care quality
    "restrain_cty",

    # staffing (hours per resident day)
    "cnahprd_pbj_cty", "dchprd_pbj_cty", "rnhprd_pbj_cty",

    # facility mix and residents
    "totbeds_cty", "nresid_cty", "occpct_cty", "profit_cty", "multifac_cty",
    "avgage_cty", "pctfem_cty",
]

# Optional: older (2000-2010) versions of some measures. They are not directly
# comparable with the newer ones above; uncomment to include them.
# COLUMNS += [
#     "pctlowcps_cty", "pctmidcps_cty", "pcthighcps_cty", "agg_cps_cty",
#     "avgadl_cty", "nhcadl_cty", "pctbedft_cty", "pctfall30_cty",
#     "cnahrppd_cty", "dchrppd_cty", "rnhrppd_cty",
# ]

ID_COLUMNS = ["year", "state", "county"]


def read_file(path):
    """Read one county file, whether it is .XLS or .CSV."""
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(path, low_memory=False, encoding="latin-1")
    try:
        return pd.read_excel(path)
    except Exception:
        # some ".XLS" downloads are really text files
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


frames = []
for path in sorted(DATA_DIR.iterdir()):
    match = re.match(r"county_?(\d{4})", path.name.lower())
    if not match:
        continue  # skips COUNTY_PCTPVTDAYS.XLS and any other non-yearly file

    df = read_file(path)
    df.columns = [str(col).strip().lower() for col in df.columns]  # names differ in case/spacing between years
    df["year"] = int(match.group(1))                               # take the year from the file name
    frames.append(df.reindex(columns=COLUMNS))                     # keep our columns; missing ones become NaN
    print(f"{path.name}: {len(df)} counties")

df_all = pd.concat(frames, ignore_index=True)

# Identifiers as text, all measures as numbers (non-numeric codes become NaN).
for col in ["state", "county"]:
    df_all[col] = df_all[col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
measures = [col for col in COLUMNS if col not in ID_COLUMNS]
df_all[measures] = df_all[measures].apply(pd.to_numeric, errors="coerce")

df_all = df_all.sort_values(["year", "state", "county"]).reset_index(drop=True)
df_all.to_csv(OUTPUT_FILE, index=False)

never_found = [col for col in measures if df_all[col].isna().all()]
if never_found:
    print("\nEmpty in every year (check the name in the dictionary): " + ", ".join(never_found))
print(f"\n{len(df_all)} rows, {df_all['year'].min()}-{df_all['year'].max()} -> {OUTPUT_FILE}")