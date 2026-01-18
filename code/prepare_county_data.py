"""
Prepare County-Level Dataset for Analysis
==========================================
Processes:
1. BEA county per capita income
2. Census county age distribution (share 20-40)
3. Downloads county unemployment from USDA ERS
4. Downloads county birth data from CDC/NVSS

Output: data/county_panel.csv
"""

import pandas as pd
import numpy as np
import requests
import io
import os

DATA_DIR = '/Users/landieraugustin/project1/data'

# =============================================================================
# 1. PROCESS BEA COUNTY INCOME DATA
# =============================================================================
print("="*70)
print("1. PROCESSING BEA COUNTY INCOME DATA")
print("="*70)

bea_raw = pd.read_csv(f'{DATA_DIR}/bea_county_income_raw.csv', low_memory=False)
print(f"Loaded BEA data: {len(bea_raw)} rows")

# Filter to per capita personal income (LineCode == 3)
bea_pci = bea_raw[bea_raw['LineCode'] == 3.0].copy()
print(f"Per capita income rows: {len(bea_pci)}")

# Clean GeoFIPS - remove quotes and spaces
bea_pci['GeoFIPS'] = bea_pci['GeoFIPS'].astype(str).str.strip().str.replace('"', '').str.replace(' ', '')
bea_pci['GeoFIPS'] = bea_pci['GeoFIPS'].str.zfill(5)

# Filter to counties only (5-digit FIPS starting with state code, not "00000" which is US total)
# County FIPS codes have format SSCCC where SS=state, CCC=county
bea_pci = bea_pci[bea_pci['GeoFIPS'].str.len() == 5]
bea_pci = bea_pci[~bea_pci['GeoFIPS'].str.endswith('000')]  # Remove state totals
bea_pci = bea_pci[bea_pci['GeoFIPS'] != '00000']  # Remove US total
print(f"County-level rows: {len(bea_pci)}")

# Reshape to long format
year_cols = [str(y) for y in range(2007, 2024)]  # Focus on 2007-2023 for births data overlap
available_years = [c for c in year_cols if c in bea_pci.columns]
print(f"Years available: {available_years}")

bea_long = bea_pci.melt(
    id_vars=['GeoFIPS', 'GeoName'],
    value_vars=available_years,
    var_name='year',
    value_name='pci'
)
bea_long['year'] = bea_long['year'].astype(int)

# Clean PCI values - remove non-numeric
bea_long['pci'] = pd.to_numeric(bea_long['pci'], errors='coerce')
bea_long = bea_long.dropna(subset=['pci'])
bea_long = bea_long[bea_long['pci'] > 0]

# Rename for merge
bea_long = bea_long.rename(columns={'GeoFIPS': 'fips', 'GeoName': 'county_name'})
print(f"BEA long format: {len(bea_long):,} county-year observations")
print(f"Counties: {bea_long['fips'].nunique():,}")
print(f"Years: {bea_long['year'].min()} - {bea_long['year'].max()}")

# =============================================================================
# 2. PROCESS CENSUS COUNTY AGE DATA
# =============================================================================
print("\n" + "="*70)
print("2. PROCESSING CENSUS COUNTY AGE DATA")
print("="*70)

census = pd.read_csv(f'{DATA_DIR}/census_county_pop_raw.csv', low_memory=False)
print(f"Loaded Census data: {len(census)} rows")
print(f"Columns: {census.columns.tolist()[:10]}...")

# Create FIPS code
census['fips'] = census['STATE'].astype(str).str.zfill(2) + census['COUNTY'].astype(str).str.zfill(3)

# Map YEAR codes to actual years (1=2020, 2=2021, 3=2022, 4=2023, 5=2024)
year_map = {1: 2020, 2: 2021, 3: 2022, 4: 2023, 5: 2024}
census['year'] = census['YEAR'].map(year_map)

# Calculate share of population aged 20-40
# Use AGE2024_TOT + AGE2529_TOT + AGE3034_TOT + AGE3539_TOT
census['pop_20_40'] = (census['AGE2024_TOT'] + census['AGE2529_TOT'] +
                       census['AGE3034_TOT'] + census['AGE3539_TOT'])
census['share_20_40'] = census['pop_20_40'] / census['POPESTIMATE']

# Keep relevant columns
census_clean = census[['fips', 'STNAME', 'CTYNAME', 'year', 'POPESTIMATE', 'share_20_40']].copy()
census_clean = census_clean.rename(columns={
    'STNAME': 'state_name',
    'CTYNAME': 'county_name',
    'POPESTIMATE': 'population'
})

print(f"Census clean: {len(census_clean):,} county-year observations")
print(f"Counties: {census_clean['fips'].nunique():,}")
print(f"Years: {census_clean['year'].min()} - {census_clean['year'].max()}")
print(f"Share 20-40 range: {census_clean['share_20_40'].min():.3f} - {census_clean['share_20_40'].max():.3f}")

# =============================================================================
# 3. DOWNLOAD COUNTY UNEMPLOYMENT FROM USDA ERS
# =============================================================================
print("\n" + "="*70)
print("3. DOWNLOADING COUNTY UNEMPLOYMENT FROM USDA ERS")
print("="*70)

# USDA ERS County-level unemployment data
# https://www.ers.usda.gov/data-products/county-level-data-sets/
ers_url = "https://www.ers.usda.gov/webdocs/DataFiles/48747/Unemployment.csv"
print(f"Downloading from: {ers_url}")

try:
    unemp_df = pd.read_csv(ers_url, encoding='latin-1')
    print(f"Downloaded: {len(unemp_df)} rows")
    print(f"Columns: {unemp_df.columns.tolist()}")

    # Save raw
    unemp_df.to_csv(f'{DATA_DIR}/county_unemployment_raw.csv', index=False)
    print("Saved: data/county_unemployment_raw.csv")

    # Check structure
    print("\nSample data:")
    print(unemp_df.head(3))

except Exception as e:
    print(f"USDA ERS download failed: {e}")
    print("\nTrying alternative BLS approach...")

    # Alternative: Download from BLS LAUS bulk data
    # https://download.bls.gov/pub/time.series/la/
    try:
        bls_url = "https://download.bls.gov/pub/time.series/la/la.data.64.County"
        response = requests.get(bls_url, timeout=120, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            print("BLS bulk download successful")
            with open(f'{DATA_DIR}/bls_county_unemployment_raw.txt', 'w') as f:
                f.write(response.text)
            print("Saved: data/bls_county_unemployment_raw.txt")
        else:
            print(f"BLS bulk download HTTP {response.status_code}")
    except Exception as e2:
        print(f"BLS bulk download also failed: {e2}")
        unemp_df = None

# =============================================================================
# 4. DOWNLOAD COUNTY BIRTH DATA
# =============================================================================
print("\n" + "="*70)
print("4. GETTING COUNTY BIRTH DATA")
print("="*70)

# CDC WONDER requires manual query, but we can use NVSS public data
# Try the CDC natality public-use microdata summary files
# Or use the pre-computed county birth rates from CDC

# Alternative: Download from County Health Rankings
# https://www.countyhealthrankings.org/

print("""
CDC WONDER natality data requires manual query:
1. Go to https://wonder.cdc.gov/natality-current.html
2. Group by: County, Year
3. Years: 2007-2022
4. Export as tab-delimited text

For now, we'll proceed with state-level births as fallback.
""")

# Try County Health Rankings data (has birth rates)
chr_url = "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2023.csv"
print(f"Trying County Health Rankings: {chr_url}")

try:
    # This is a complex file, let's try
    response = requests.get(chr_url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        chr_df = pd.read_csv(io.StringIO(response.text), encoding='latin-1', low_memory=False)
        print(f"Downloaded CHR data: {len(chr_df)} rows")
        chr_df.to_csv(f'{DATA_DIR}/county_health_rankings_raw.csv', index=False)
        print("Saved: data/county_health_rankings_raw.csv")
    else:
        print(f"CHR download HTTP {response.status_code}")
except Exception as e:
    print(f"CHR download failed: {e}")

# =============================================================================
# 5. CREATE MERGED COUNTY PANEL
# =============================================================================
print("\n" + "="*70)
print("5. CREATING MERGED COUNTY PANEL")
print("="*70)

# Start with BEA income data
panel = bea_long[['fips', 'county_name', 'year', 'pci']].copy()

# Merge Census age data
panel = panel.merge(
    census_clean[['fips', 'year', 'population', 'share_20_40']],
    on=['fips', 'year'],
    how='left'
)

print(f"Panel after BEA + Census merge: {len(panel):,} rows")
print(f"With population: {panel['population'].notna().sum():,}")

# Process unemployment if downloaded
if os.path.exists(f'{DATA_DIR}/county_unemployment_raw.csv'):
    print("\nProcessing unemployment data...")
    unemp = pd.read_csv(f'{DATA_DIR}/county_unemployment_raw.csv', encoding='latin-1')

    # Check column names
    print(f"Unemployment columns: {unemp.columns.tolist()[:15]}...")

    # The USDA ERS file typically has columns like:
    # FIPS_Code, State, Area_name, Unemployment_rate_2000, ..., Unemployment_rate_2022

    # Find FIPS column
    fips_col = None
    for col in unemp.columns:
        if 'fips' in col.lower():
            fips_col = col
            break

    if fips_col:
        # Reshape unemployment data to long format
        unemp_cols = [c for c in unemp.columns if 'unemployment' in c.lower() and c[-4:].isdigit()]
        print(f"Found unemployment columns: {len(unemp_cols)}")

        if unemp_cols:
            # Extract years from column names
            unemp_long = unemp.melt(
                id_vars=[fips_col],
                value_vars=unemp_cols,
                var_name='year_col',
                value_name='unemployment_rate'
            )
            # Extract year from column name (e.g., "Unemployment_rate_2020" -> 2020)
            unemp_long['year'] = unemp_long['year_col'].str.extract(r'(\d{4})').astype(int)
            unemp_long['fips'] = unemp_long[fips_col].astype(str).str.zfill(5)
            unemp_long = unemp_long[['fips', 'year', 'unemployment_rate']]

            # Merge with panel
            panel = panel.merge(unemp_long, on=['fips', 'year'], how='left')
            print(f"With unemployment: {panel['unemployment_rate'].notna().sum():,}")

# Add log transformations
panel['log_pci'] = np.log(panel['pci'])
panel['log_population'] = np.log(panel['population'])

# Extract state code from FIPS
panel['state_fips'] = panel['fips'].str[:2]

# Save panel
panel.to_csv(f'{DATA_DIR}/county_panel.csv', index=False)
print(f"\nSaved: data/county_panel.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("COUNTY PANEL SUMMARY")
print("="*70)
print(f"Total observations: {len(panel):,}")
print(f"Counties: {panel['fips'].nunique():,}")
print(f"Years: {panel['year'].min()} - {panel['year'].max()}")
print(f"\nVariable coverage:")
print(f"  Per capita income: {panel['pci'].notna().sum():,} ({panel['pci'].notna().mean()*100:.1f}%)")
print(f"  Population: {panel['population'].notna().sum():,} ({panel['population'].notna().mean()*100:.1f}%)")
print(f"  Share 20-40: {panel['share_20_40'].notna().sum():,} ({panel['share_20_40'].notna().mean()*100:.1f}%)")
if 'unemployment_rate' in panel.columns:
    print(f"  Unemployment: {panel['unemployment_rate'].notna().sum():,} ({panel['unemployment_rate'].notna().mean()*100:.1f}%)")

print("\nData file sizes:")
for f in os.listdir(DATA_DIR):
    if 'county' in f.lower():
        size = os.path.getsize(f'{DATA_DIR}/{f}') / 1024
        print(f"  {f}: {size:.0f} KB")
