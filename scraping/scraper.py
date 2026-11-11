"""
Download all LTCFocus county files into one folder (Selenium version).

How it works
1. Selenium opens the LTCFocus download page in Chrome (the link from your email).
2. It finds every link (<a> tag) on the page and keeps the ones whose button text
   starts with "COUNTY", plus the data dictionary.
3. It closes the browser and downloads each file's address (href) with requests.
   Selenium could click the buttons instead, but browsers sometimes show CSV files
   on screen rather than saving them; downloading the address directly always saves
   the file.

Requires: Google Chrome, plus
    pip install selenium requests
(Selenium 4.6+ downloads the matching ChromeDriver by itself.)

Usage
    python download_ltcfocus_county_selenium.py "<link from the LTCFocus email>"
    python download_ltcfocus_county_selenium.py "<link>" --out-dir data/ltcfocus/county
    python download_ltcfocus_county_selenium.py "<link>" --headless    # no browser window
"""

import argparse
import sys
import time
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

FILE_EXTENSIONS = (".xls", ".xlsx", ".csv")
DELAY_SECONDS = 1   # pause between downloads, to be polite to the server


def start_browser(headless):
    """Start Chrome. With --headless it runs without opening a window."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def find_county_links(driver, page_url):
    """Open the download page and return a sorted list of (file name, file address)."""
    driver.get(page_url)

    # Wait (max 20 seconds) until the page has at least one link.
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "a")))

    files = {}
    for link in driver.find_elements(By.TAG_NAME, "a"):
        name = (link.text or link.get_attribute("textContent") or "").strip()
        href = link.get_attribute("href")   # Selenium returns the full address, even for relative links

        # Only keep buttons that are files, e.g. "COUNTY_2017.CSV".
        if not href or not name.lower().endswith(FILE_EXTENSIONS):
            continue

        is_county = name.upper().startswith("COUNTY")
        is_dictionary = name.upper().startswith("DATA_DICTIONARY")
        if (is_county or is_dictionary) and name not in files:
            files[name] = href

    return sorted(files.items())


def download(url, target):
    """Save the file at url to target and return its size in bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()   # stop if the server answered with an error (e.g. 404)
    if "text/html" in response.headers.get("Content-Type", ""):
        raise ValueError("got a web page instead of a file")
    target.write_bytes(response.content)
    return len(response.content)


def main():
    parser = argparse.ArgumentParser(description="Download LTCFocus county files into one folder.")
    parser.add_argument("page_url", help="the download-page link LTCFocus emailed you")
    parser.add_argument("--out-dir", type=Path, default=Path("data/ltcfocus/county"),
                        help="folder to save the files in (default: data/ltcfocus/county)")
    parser.add_argument("--headless", action="store_true", help="run Chrome without a window")
    args = parser.parse_args()

    # Step 1-2: use the browser only to read the page and collect the file links.
    driver = start_browser(args.headless)
    try:
        files = find_county_links(driver, args.page_url)
    finally:
        driver.quit()   # always close the browser, even if something went wrong

    if not files:
        sys.exit("No county files found on the page. Has the link expired? "
                 "Request a new one at https://ltcfocus.org/data")
    print(f"Found {len(files)} files")

    # Step 3: download every file into one folder, skipping ones we already have.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for number, (name, url) in enumerate(files, start=1):
        target = args.out_dir / name
        if target.exists():
            print(f"[{number}/{len(files)}] {name}: already there, skipped")
            continue
        try:
            size = download(url, target)
            print(f"[{number}/{len(files)}] {name}: {size / 1024:,.0f} KB")
        except Exception as error:
            print(f"[{number}/{len(files)}] {name}: FAILED - {error}")
            failed.append(name)
        time.sleep(DELAY_SECONDS)

    print(f"\nDone -> {args.out_dir.resolve()}")
    if failed:
        print("Failed: " + ", ".join(failed) + " (run the script again to retry them)")


if __name__ == "__main__":
    main()