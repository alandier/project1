"""
Download county-level data using alternative sources
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os

DATA_DIR = '/Users/landieraugustin/project1/data'

# =============================================================================
# 1. DOWNLOAD BEA COUNTY INCOME DATA
# =============================================================================
print("="*70)
print("1. BEA COUNTY PERSONAL INCOME")
print("="*70)

# BEA Regional Data - CAINC1 (Personal Income Summary by County)
bea_url = "https://apps.bea.gov/regional/zip/CAINC1.zip"

print(f"Downloading BEA county income from: {bea_url}")
try:
    response = requests.get(bea_url, timeout=60)
    if response.status_code == 200:
        # Extract ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print(f"ZIP contents: {z.namelist()}")
            # Find the CSV file
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if csv_files:
                csv_file = csv_files[0]
                print(f"Extracting: {csv_file}")
                with z.open(csv_file) as f:
                    bea_df = pd.read_csv(f, encoding='latin-1', low_memory=False)
                print(f"Loaded: {len(bea_df)} rows")
                print(f"Columns: {bea_df.columns.tolist()[:10]}")

                # Save raw
                bea_df.to_csv(f'{DATA_DIR}/bea_county_income_raw.csv', index=False)
                print(f"Saved: data/bea_county_income_raw.csv")
    else:
        print(f"HTTP {response.status_code}")
except Exception as e:
    print(f"BEA download failed: {e}")

# =============================================================================
# 2. DOWNLOAD BLS COUNTY UNEMPLOYMENT (alternative approach)
# =============================================================================
print("\n" + "="*70)
print("2. BLS COUNTY UNEMPLOYMENT")
print("="*70)

# Try the LAU database Excel files
bls_excel_url = "https://www.bls.gov/lau/laucnty.xlsx"
print(f"Trying BLS Excel: {bls_excel_url}")

try:
    response = requests.get(bls_excel_url, timeout=30,
                           headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        bls_df = pd.read_excel(io.BytesIO(response.content))
        print(f"Loaded: {len(bls_df)} rows")
        bls_df.to_csv(f'{DATA_DIR}/bls_county_unemployment_raw.csv', index=False)
        print("Saved: data/bls_county_unemployment_raw.csv")
    else:
        print(f"HTTP {response.status_code}")
except Exception as e:
    print(f"BLS Excel failed: {e}")
    print("Trying text file alternative...")

    # Try direct text file for a recent year
    try:
        bls_txt_url = "https://www.bls.gov/web/metro/laucntycur14.txt"
        response = requests.get(bls_txt_url, timeout=30,
                               headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            print(f"Got BLS text file: {len(response.text)} chars")
            with open(f'{DATA_DIR}/bls_county_unemployment.txt', 'w') as f:
                f.write(response.text)
            print("Saved: data/bls_county_unemployment.txt")
    except Exception as e2:
        print(f"BLS text also failed: {e2}")

# =============================================================================
# 3. DOWNLOAD USDA ATLAS DATA (alternative for county economics)
# =============================================================================
print("\n" + "="*70)
print("3. USDA COUNTY DATA (Food Environment Atlas)")
print("="*70)

# USDA Food Environment Atlas has county demographics
usda_url = "https://www.ers.usda.gov/webdocs/DataFiles/80591/DataDownload.xlsx"
print(f"Trying USDA Atlas: {usda_url}")

try:
    response = requests.get(usda_url, timeout=60,
                           headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        # This is an Excel file with multiple sheets
        xls = pd.ExcelFile(io.BytesIO(response.content))
        print(f"Sheets: {xls.sheet_names}")

        # Get relevant sheets
        for sheet in xls.sheet_names[:5]:
            df = pd.read_excel(xls, sheet_name=sheet)
            print(f"  {sheet}: {len(df)} rows, cols: {df.columns.tolist()[:5]}")

    else:
        print(f"HTTP {response.status_code}")
except Exception as e:
    print(f"USDA Atlas failed: {e}")

# =============================================================================
# 4. CENSUS COUNTY POPULATION (via API or direct download)
# =============================================================================
print("\n" + "="*70)
print("4. CENSUS COUNTY POPULATION")
print("="*70)

# Census county population estimates
# Direct download from Census FTP
census_url = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/asrh/cc-est2023-agesex-all.csv"
print(f"Trying Census population: {census_url}")

try:
    response = requests.get(census_url, timeout=60)
    if response.status_code == 200:
        census_df = pd.read_csv(io.StringIO(response.text), encoding='latin-1')
        print(f"Loaded: {len(census_df)} rows")
        print(f"Columns: {census_df.columns.tolist()[:10]}")
        census_df.to_csv(f'{DATA_DIR}/census_county_pop_raw.csv', index=False)
        print("Saved: data/census_county_pop_raw.csv")
    else:
        print(f"HTTP {response.status_code}")
except Exception as e:
    print(f"Census download failed: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("DOWNLOAD SUMMARY")
print("="*70)

files = os.listdir(DATA_DIR)
new_files = [f for f in files if 'county' in f.lower() or 'bea' in f.lower() or 'bls' in f.lower() or 'census' in f.lower()]
print("New data files:")
for f in new_files:
    size = os.path.getsize(f'{DATA_DIR}/{f}') / 1024
    print(f"  {f}: {size:.0f} KB")
