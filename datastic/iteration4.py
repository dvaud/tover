"""
Iteration 4: select project-relevant variables from the LTCFocus facility files.

Reads every yearly LTCFocus facility file (FACIL2000NEWWEB.XLSX ... FACILITY_2023.XLS),
keeps only the variables relevant to our dementia-care question, stacks the years
into one facility-year panel and writes three CSVs to the output folder:

    ltcfocus_selected_panel.csv     one row per nursing home per year
    ltcfocus_selected_codebook.csv  each selected variable, its group, label and years present
    ltcfocus_coverage_by_year.csv   % of facilities with a value, per variable per year

- Each row is a nursing home's averages, not a person.
- The resident assessment (MDS) changed version in October 2010. Variables ending in
  _mds3 (2011 onwards) are not directly comparable with their older counterparts, so
  both are kept and an `mds_version` column marks which era a row belongs to.
  The cognition bands work the same way: CPS = older scale, CFS = newer scale.
- Only facility-level variables are used; the _cty/_sta versions live in the
  county and state files.

"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Default locations, overridable from the command line.
DATA_DIR = Path("data/ltcfocus/county")
OUT_DIR = Path("output")

# Variables to keep, grouped by why they matter for the project.
# Names are lower case; column names in the files are matched case-insensitively.
VARIABLE_GROUPS = {
    "identifiers": [
        "accpt_id",   # LTCFocus facility ID
        "prov1680",   # Medicare provider number
        "prov0475",   # facility name
        "state",
        "county",
    ],
    "dementia_cognition": [
        "pctalz_dem",   # % Alzheimer's/dementia (current residents)
        "aggalz_dem",   # % Alzheimer's/dementia (all admissions)
        "pctlowcps",    # % CPS 0-2 (intact to mild) - older scale
        "pctmidcps",    # % CPS 3-4 (moderate)
        "pcthighcps",   # % CPS 5-6 (severe)
        "agg_cps",      # % CPS 5-6 at admission
        "pctlowcfs",    # % CFS 1 (intact) - newer scale
        "pctmidcfs",    # % CFS 2-3 (mild to moderate)
        "pcthighcfs",   # % CFS 4 (severe)
        "agghighcfs",   # % CFS 4 at admission
    ],
    "dementia_care": [
        "alzunit",      # has an Alzheimer's special care unit
    ],
    "functioning": [
        "avgadl", "avgadl_mds3",            # average dependency in daily activities
        "nhcadl", "nhcadl_mds3",            # % long-stay residents with ADL decline
        "pctwalking",                       # % walk independently in corridor
        "pctbedft", "pctbedft_mds3",        # % bedfast
        "pctfall30", "pctfall30_mds3",      # % fallen recently
    ],
    "wellbeing_quality": [
        "restrain",                         # % physically restrained
        "nhcpain", "nhcpain_mds3",          # % long-stay residents in daily pain
        "avgrxnum",                         # average number of medications
    ],
    "staffing": [
        "cnahprd_pbj", "dchprd_pbj", "rnhprd_pbj",   # hours per resident day, payroll data (newer)
        "cnahrppd", "dchrppd", "rnhrppd",            # hours per resident day (older source)
    ],
    "facility": [
        "totbeds", "nresid", "occpct", "profit", "multifac",
    ],
    "residents": [
        "avgage", "pctfem", "obs_medianlos",
    ],
}

ID_COLUMNS = set(VARIABLE_GROUPS["identifiers"])
WANTED = {variable: group for group, variables in VARIABLE_GROUPS.items() for variable in variables}

# Matches FACIL2000NEWWEB.XLSX, FACIL2007NEW.XLS, FACILITY_2011.XLS, ... (lower-cased)
YEAR_PATTERN = re.compile(r"facil(?:ity)?_?(\d{4})")
YES_NO = {"yes": 1, "y": 1, "no": 0, "n": 0}


def clean_name(name):
    """Column names vary in case and sometimes carry trailing spaces."""
    return str(name).strip().lower()


def find_facility_files(data_dir):
    """Return {year: path} for every facility-level file in the folder."""
    files = {}
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in {".xls", ".xlsx", ".csv"}:
            continue
        match = YEAR_PATTERN.search(path.name.lower())
        if not match:
            continue
        year = int(match.group(1))
        if year in files:
            print(f"  warning: two files for {year}; using {files[year].name}, ignoring {path.name}")
            continue
        files[year] = path
    return dict(sorted(files.items()))


def read_table(path, usecols=None):
    """Read an LTCFocus download (.xls, .xlsx or .csv); some files aren't what their extension says."""
    csv_attempts = [
        lambda: pd.read_csv(path, usecols=usecols, low_memory=False),
        lambda: pd.read_csv(path, usecols=usecols, low_memory=False, encoding="latin-1"),
        lambda: pd.read_csv(path, usecols=usecols, sep=None, engine="python", encoding="latin-1"),
    ]
    excel_attempts = [
        lambda: pd.read_excel(path, usecols=usecols),
        lambda: pd.read_excel(path, usecols=usecols, engine="openpyxl"),
        lambda: pd.read_excel(path, usecols=usecols, engine="xlrd"),
    ]
    # Try the format the extension suggests first, then the other one.
    if path.suffix.lower() == ".csv":
        attempts = csv_attempts + excel_attempts
    else:
        attempts = excel_attempts + csv_attempts
    errors = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    raise RuntimeError(f"Could not read {path.name}:\n  " + "\n  ".join(errors))


def prepare_year(path, year):
    """Load one year's file, keeping only the wanted columns."""
    table = read_table(path, usecols=lambda column: clean_name(column) in WANTED)
    table.columns = [clean_name(column) for column in table.columns]
    table = table.loc[:, ~table.columns.duplicated()]
    if table.shape[1] == 0:
        print(f"  warning: no selected columns found in {path.name} (is the header in the first row?)")
    table.insert(0, "year", year)
    table.insert(1, "mds_version", 2 if year <= 2010 else 3)
    return table


def clean_ids(panel):
    """Keep identifiers as text so provider numbers keep their leading zeros."""
    for column in ID_COLUMNS & set(panel.columns):
        panel[column] = (
            panel[column].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        )
    if "prov1680" in panel.columns:
        digits_only = panel["prov1680"].str.fullmatch(r"\d+", na=False)
        panel.loc[digits_only, "prov1680"] = panel.loc[digits_only, "prov1680"].str.zfill(6)
    return panel


def clean_values(panel):
    """Turn yes/no columns into 1/0 and everything else into numbers; report what was lost."""
    for column in panel.columns:
        if column in ID_COLUMNS or column in ("year", "mds_version"):
            continue
        # Years can differ: one file may say "Yes"/"No" where another says 1/0.
        as_text = panel[column].astype("string").str.strip().str.lower()
        is_yes_no = as_text.isin(YES_NO.keys()).fillna(False).astype(bool)
        if is_yes_no.any():
            panel[column] = panel[column].mask(is_yes_no, as_text.map(YES_NO))
        before = panel[column].notna().sum()
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
        lost = before - panel[column].notna().sum()
        if lost:
            print(f"  {column}: {lost} non-numeric values set to missing")
    return panel


def load_dictionary(data_dir):
    """Read variable labels from DATA_DICTIONARY_*.XLSX (first column = name, second = label)."""
    matches = sorted(
        path for path in data_dir.iterdir()
        if path.name.lower().startswith("data_dictionary") and path.suffix.lower() in {".xls", ".xlsx"}
    )
    if not matches:
        print("  no DATA_DICTIONARY file found; codebook labels will be empty")
        return {}
    table = read_table(matches[-1])
    names, labels = table.iloc[:, 0], table.iloc[:, 1]
    return {clean_name(name): str(label).strip() for name, label in zip(names, labels) if pd.notna(name)}


def build_codebook(panel, labels):
    rows = []
    for variable, group in WANTED.items():
        years = panel.loc[panel[variable].notna(), "year"] if variable in panel.columns else pd.Series(dtype=int)
        rows.append({
            "variable": variable,
            "group": group,
            "label": labels.get(variable, ""),
            "first_year": int(years.min()) if len(years) else None,
            "last_year": int(years.max()) if len(years) else None,
        })
    return pd.DataFrame(rows).astype({"first_year": "Int64", "last_year": "Int64"})


def coverage_by_year(panel):
    """% of facilities with a value for each variable in each year (variables as rows)."""
    variables = [v for v in WANTED if v in panel.columns and v not in ID_COLUMNS]
    return panel[variables].notna().groupby(panel["year"]).mean().mul(100).round(1).T


def main():
    parser = argparse.ArgumentParser(description="Select project variables from LTCFocus facility files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="folder with the LTCFocus downloads")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="folder for the output CSVs")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"Data folder not found: {args.data_dir.resolve()}")

    files = find_facility_files(args.data_dir)
    if not files:
        sys.exit(f"No LTCFocus facility files found in {args.data_dir.resolve()}")
    print(f"Found facility files for {len(files)} years ({min(files)}-{max(files)})")

    yearly = []
    for year, path in files.items():
        table = prepare_year(path, year)
        print(f"  {year}: {len(table):>6} facilities, {table.shape[1] - 2:>2} selected columns  ({path.name})")
        yearly.append(table)

    panel = pd.concat(yearly, ignore_index=True, sort=False)
    panel = clean_ids(panel)
    panel = clean_values(panel)
    panel = panel[["year", "mds_version"] + [v for v in WANTED if v in panel.columns]]

    never_found = [v for v in WANTED if v not in panel.columns]
    if never_found:
        print("Not found in any year (check the dictionary for renamed columns): " + ", ".join(never_found))

    labels = load_dictionary(args.data_dir)
    codebook = build_codebook(panel, labels)
    coverage = coverage_by_year(panel)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_dir / "ltcfocus_selected_panel.csv", index=False)
    codebook.to_csv(args.out_dir / "ltcfocus_selected_codebook.csv", index=False)
    coverage.to_csv(args.out_dir / "ltcfocus_coverage_by_year.csv")

    id_column = "prov1680" if "prov1680" in panel.columns else "accpt_id"
    facilities = panel[id_column].nunique() if id_column in panel.columns else "unknown"
    print(f"\nDone: {len(panel)} facility-years, {facilities} distinct facilities, "
          f"{panel.shape[1] - 2} variables -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()