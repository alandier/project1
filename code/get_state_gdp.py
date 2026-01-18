"""
Download state-level GDP per capita from FRED
"""

import pandas as pd
import requests
import time

# FRED API (free, no key needed for CSV download)
# State Real GDP per Capita series: [STATE]RQGSP or similar

# State abbreviations and their FRED codes for Real GDP per capita
# Using Real Gross Domestic Product: All Industry Total (RGDP) series

state_codes = {
    'AL': 'ALRGSP', 'AK': 'AKRGSP', 'AZ': 'AZRGSP', 'AR': 'ARRGSP', 'CA': 'CARGSP',
    'CO': 'CORGSP', 'CT': 'CTRGSP', 'DE': 'DERGSP', 'DC': 'DCRGSP', 'FL': 'FLRGSP',
    'GA': 'GARGSP', 'HI': 'HIRGSP', 'ID': 'IDRGSP', 'IL': 'ILRGSP', 'IN': 'INRGSP',
    'IA': 'IARGSP', 'KS': 'KSRGSP', 'KY': 'KYRGSP', 'LA': 'LARGSP', 'ME': 'MERGSP',
    'MD': 'MDRGSP', 'MA': 'MARGSP', 'MI': 'MIRGSP', 'MN': 'MNRGSP', 'MS': 'MSRGSP',
    'MO': 'MORGSP', 'MT': 'MTRGSP', 'NE': 'NERGSP', 'NV': 'NVRGSP', 'NH': 'NHRGSP',
    'NJ': 'NJRGSP', 'NM': 'NMRGSP', 'NY': 'NYRGSP', 'NC': 'NCRGSP', 'ND': 'NDRGSP',
    'OH': 'OHRGSP', 'OK': 'OKRGSP', 'OR': 'ORRGSP', 'PA': 'PARGSP', 'RI': 'RIRGSP',
    'SC': 'SCRGSP', 'SD': 'SDRGSP', 'TN': 'TNRGSP', 'TX': 'TXRGSP', 'UT': 'UTRGSP',
    'VT': 'VTRGSP', 'VA': 'VARGSP', 'WA': 'WARGSP', 'WV': 'WVRGSP', 'WI': 'WIRGSP',
    'WY': 'WYRGSP'
}

# Also get population to compute per capita
pop_codes = {
    'AL': 'ALPOP', 'AK': 'AKPOP', 'AZ': 'AZPOP', 'AR': 'ARPOP', 'CA': 'CAPOP',
    'CO': 'COPOP', 'CT': 'CTPOP', 'DE': 'DEPOP', 'DC': 'DCPOP', 'FL': 'FLPOP',
    'GA': 'GAPOP', 'HI': 'HIPOP', 'ID': 'IDPOP', 'IL': 'ILPOP', 'IN': 'INPOP',
    'IA': 'IAPOP', 'KS': 'KSPOP', 'KY': 'KYPOP', 'LA': 'LAPOP', 'ME': 'MEPOP',
    'MD': 'MDPOP', 'MA': 'MAPOP', 'MI': 'MIPOP', 'MN': 'MNPOP', 'MS': 'MSPOP',
    'MO': 'MOPOP', 'MT': 'MTPOP', 'NE': 'NEPOP', 'NV': 'NVPOP', 'NH': 'NHPOP',
    'NJ': 'NJPOP', 'NM': 'NMPOP', 'NY': 'NYPOP', 'NC': 'NCPOP', 'ND': 'NDPOP',
    'OH': 'OHPOP', 'OK': 'OKPOP', 'OR': 'ORPOP', 'PA': 'PAPOP', 'RI': 'RIPOP',
    'SC': 'SCPOP', 'SD': 'SDPOP', 'TN': 'TNPOP', 'TX': 'TXPOP', 'UT': 'UTPOP',
    'VT': 'VTPOP', 'VA': 'VAPOP', 'WA': 'WAPOP', 'WV': 'WVPOP', 'WI': 'WIPOP',
    'WY': 'WYPOP'
}

def fetch_fred_series(series_id):
    """Fetch a FRED series as CSV"""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        df = pd.read_csv(url)
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        return df[['year', 'value']].dropna()
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return None

print("Fetching state GDP data from FRED...")

all_data = []

for state, gdp_code in state_codes.items():
    print(f"  {state}...", end=" ")

    # Get GDP
    gdp_df = fetch_fred_series(gdp_code)
    if gdp_df is None:
        print("failed")
        continue
    gdp_df = gdp_df.rename(columns={'value': 'real_gdp'})

    # Get population
    pop_code = pop_codes[state]
    pop_df = fetch_fred_series(pop_code)
    if pop_df is None:
        print("failed (pop)")
        continue
    pop_df = pop_df.rename(columns={'value': 'population'})

    # Merge
    merged = gdp_df.merge(pop_df, on='year', how='inner')
    merged['state'] = state

    # GDP is in millions, population is in thousands
    # GDP per capita in thousands of dollars
    merged['gdp_per_capita'] = (merged['real_gdp'] * 1000) / merged['population']

    all_data.append(merged[['state', 'year', 'real_gdp', 'population', 'gdp_per_capita']])
    print("OK")

    time.sleep(0.1)  # Be nice to FRED

# Combine all states
state_gdp = pd.concat(all_data, ignore_index=True)

# Filter to years we need
state_gdp = state_gdp[(state_gdp['year'] >= 1990) & (state_gdp['year'] <= 2024)]

print(f"\nGot {len(state_gdp)} state-year observations")
print(f"Years: {state_gdp['year'].min()} - {state_gdp['year'].max()}")
print(f"States: {state_gdp['state'].nunique()}")

# Save
state_gdp.to_csv('/Users/landieraugustin/project1/data/state_gdp.csv', index=False)
print("\nSaved to data/state_gdp.csv")

print("\nSummary:")
print(state_gdp[['real_gdp', 'population', 'gdp_per_capita']].describe())
