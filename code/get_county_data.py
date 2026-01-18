"""
Download county-level data from official sources:
- BEA: Personal income per capita
- BLS: Unemployment
- Census: Age demographics
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import time

# =============================================================================
# 1. COUNTY PERSONAL INCOME FROM BEA
# =============================================================================
print("="*70)
print("1. COUNTY PERSONAL INCOME (BEA)")
print("="*70)

# BEA provides county-level personal income via their Regional Data
# Download from: https://apps.bea.gov/regional/downloadzip.cfm
# Table CAINC1: Personal Income Summary

# Direct API approach
bea_url = "https://apps.bea.gov/regional/zip/CAINC1.zip"
print(f"BEA data available at: {bea_url}")
print("(Large file - using pre-processed approach)")

# Alternative: Use FRED for county per capita income
# FRED has series like "PCPI01001" for Autauga County, AL
# Pattern: PCPI + [5-digit FIPS]

# For efficiency, let's download a sample and work with what we can get
# BEA also provides data via API

# Try BEA API (requires key, but let's try public endpoint)
try:
    # BEA bulk download for county income
    print("\nDownloading county income data...")

    # This is a direct link to county income CSV
    # Using FRED's county data as alternative

    # Sample approach: get data for major counties
    print("Using state-level per capita income as proxy (will download full county data separately)")

except Exception as e:
    print(f"BEA download error: {e}")

# =============================================================================
# 2. COUNTY UNEMPLOYMENT FROM BLS LAUS
# =============================================================================
print("\n" + "="*70)
print("2. COUNTY UNEMPLOYMENT (BLS LAUS)")
print("="*70)

# BLS provides county unemployment at:
# https://www.bls.gov/lau/#tables

# Annual average data files
years_to_get = range(2000, 2024)
all_unemp = []

for year in years_to_get:
    # BLS file naming convention
    if year >= 2020:
        url = f"https://www.bls.gov/lau/laucnty{str(year)[-2:]}.txt"
    else:
        url = f"https://www.bls.gov/lau/laucnty{str(year)[-2:]}.txt"

    try:
        print(f"  Trying {year}...", end=" ")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Parse the fixed-width format
            lines = response.text.strip().split('\n')
            # Skip header lines
            data_lines = [l for l in lines if l.strip() and not l.startswith('LAUS') and
                         not l.startswith('---') and not l.startswith('State') and
                         len(l) > 50]

            if len(data_lines) > 100:
                print(f"OK ({len(data_lines)} records)")
                # Parse will be complex due to fixed-width format
                # For now, note that data is available
            else:
                print("format issue")
        else:
            print(f"HTTP {response.status_code}")
    except Exception as e:
        print(f"failed: {e}")

    time.sleep(0.2)

print("\nBLS county unemployment data is available but requires parsing fixed-width format")
print("Alternative: Use USDA ERS county unemployment data (cleaner CSV format)")

# =============================================================================
# 3. TRY USDA ERS DATA (cleaner source)
# =============================================================================
print("\n" + "="*70)
print("3. USDA ERS COUNTY DATA")
print("="*70)

# USDA ERS provides county-level unemployment and other data
# https://www.ers.usda.gov/data-products/county-level-data-sets/

ers_unemp_url = "https://www.ers.usda.gov/webdocs/DataFiles/48747/Unemployment.csv"
print(f"Trying USDA ERS: {ers_unemp_url}")

try:
    unemp_df = pd.read_csv(ers_unemp_url, encoding='latin-1')
    print(f"Downloaded: {len(unemp_df)} rows")
    print(f"Columns: {unemp_df.columns.tolist()[:10]}...")

    # Save raw
    unemp_df.to_csv('/Users/landieraugustin/project1/data/county_unemployment_raw.csv', index=False)
    print("Saved: data/county_unemployment_raw.csv")

except Exception as e:
    print(f"USDA ERS download failed: {e}")

# =============================================================================
# 4. CENSUS COUNTY POPULATION BY AGE
# =============================================================================
print("\n" + "="*70)
print("4. COUNTY POPULATION BY AGE (Census)")
print("="*70)

# Census county population estimates by age
# Available via Census API or data.census.gov

print("County age data available from Census Bureau Population Estimates")
print("URL: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html")

# =============================================================================
# 5. CDC WONDER BIRTH DATA
# =============================================================================
print("\n" + "="*70)
print("5. COUNTY BIRTH DATA (CDC WONDER)")
print("="*70)

print("""
CDC WONDER requires manual query:
1. Go to https://wonder.cdc.gov/natality-current.html
2. Group by: County, Year
3. Years: 2007-2022
4. Export as tab-delimited text

Note: Counties with <10 births are suppressed for privacy.
""")

print("\n" + "="*70)
print("SUMMARY: Data Sources")
print("="*70)
print("""
For county-level analysis, we need:

1. COUNTY UNEMPLOYMENT:
   - USDA ERS (cleanest): https://www.ers.usda.gov/data-products/county-level-data-sets/
   - BLS LAUS (official): https://www.bls.gov/lau/

2. COUNTY PERSONAL INCOME:
   - BEA CAINC1: https://apps.bea.gov/regional/downloadzip.cfm
   - Per capita personal income by county and year

3. COUNTY AGE DISTRIBUTION:
   - Census Population Estimates
   - Need % population aged 20-40

4. COUNTY BIRTHS:
   - CDC WONDER Natality (requires manual query)
   - Or use state births with county FE

5. COMPUSTAT FIRM LOCATIONS:
   - Already have via ZIP-to-county crosswalk
""")
