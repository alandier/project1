"""
County-Level Regression Analysis
================================
Birth rates ~ Unemployment + Controls
With placebo tests

Uses:
- Census county birth counts (2010-2023)
- BEA county per capita income
- State-level unemployment (county unemployment blocked)
- Census age distribution (2020-2024)
- Compustat firms mapped to counties
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/landieraugustin/project1/data'

# =============================================================================
# 1. PROCESS COUNTY BIRTH DATA
# =============================================================================
print("="*70)
print("1. PROCESSING COUNTY BIRTH DATA")
print("="*70)

# Load 2010-2019 data
df_2019 = pd.read_csv(f'{DATA_DIR}/census_county_estimates_2010-2019.csv', encoding='latin-1')
print(f"2010-2019 data: {len(df_2019)} counties")

# Load 2020-2023 data
df_2023 = pd.read_csv(f'{DATA_DIR}/census_county_estimates_raw.csv', encoding='latin-1')
print(f"2020-2023 data: {len(df_2023)} counties")

# Create FIPS codes
def make_fips(df):
    return df['STATE'].astype(str).str.zfill(2) + df['COUNTY'].astype(str).str.zfill(3)

df_2019['fips'] = make_fips(df_2019)
df_2023['fips'] = make_fips(df_2023)

# Extract birth data for 2010-2019
birth_cols_2019 = [c for c in df_2019.columns if c.startswith('BIRTHS20')]
births_2019 = df_2019[['fips', 'STNAME', 'CTYNAME'] + birth_cols_2019].copy()

# Reshape to long format
births_long_2019 = births_2019.melt(
    id_vars=['fips', 'STNAME', 'CTYNAME'],
    value_vars=birth_cols_2019,
    var_name='year_col',
    value_name='births'
)
births_long_2019['year'] = births_long_2019['year_col'].str.extract(r'(\d{4})').astype(int)
births_long_2019 = births_long_2019[['fips', 'STNAME', 'CTYNAME', 'year', 'births']]

# Extract birth data for 2020-2023
birth_cols_2023 = [c for c in df_2023.columns if c.startswith('BIRTHS20')]
births_2023 = df_2023[['fips', 'STNAME', 'CTYNAME'] + birth_cols_2023].copy()

births_long_2023 = births_2023.melt(
    id_vars=['fips', 'STNAME', 'CTYNAME'],
    value_vars=birth_cols_2023,
    var_name='year_col',
    value_name='births'
)
births_long_2023['year'] = births_long_2023['year_col'].str.extract(r'(\d{4})').astype(int)
births_long_2023 = births_long_2023[['fips', 'STNAME', 'CTYNAME', 'year', 'births']]

# Combine
births = pd.concat([births_long_2019, births_long_2023], ignore_index=True)
births = births.drop_duplicates(subset=['fips', 'year'])
births = births.rename(columns={'STNAME': 'state_name', 'CTYNAME': 'county_name'})

# Get population data for birth rate calculation
pop_cols_2019 = [c for c in df_2019.columns if c.startswith('POPESTIMATE20')]
pop_2019 = df_2019[['fips'] + pop_cols_2019].copy()
pop_long_2019 = pop_2019.melt(
    id_vars=['fips'],
    value_vars=pop_cols_2019,
    var_name='year_col',
    value_name='population'
)
pop_long_2019['year'] = pop_long_2019['year_col'].str.extract(r'(\d{4})').astype(int)

pop_cols_2023 = [c for c in df_2023.columns if c.startswith('POPESTIMATE20')]
pop_2023 = df_2023[['fips'] + pop_cols_2023].copy()
pop_long_2023 = pop_2023.melt(
    id_vars=['fips'],
    value_vars=pop_cols_2023,
    var_name='year_col',
    value_name='population'
)
pop_long_2023['year'] = pop_long_2023['year_col'].str.extract(r'(\d{4})').astype(int)

population = pd.concat([pop_long_2019[['fips', 'year', 'population']],
                        pop_long_2023[['fips', 'year', 'population']]], ignore_index=True)
population = population.drop_duplicates(subset=['fips', 'year'])

# Merge births with population
births = births.merge(population, on=['fips', 'year'], how='left')

# Calculate birth rate (per 1000)
births['birth_rate'] = births['births'] / births['population'] * 1000

# Remove counties with 0 births or population
births = births[(births['births'] > 0) & (births['population'] > 0)]
births = births[births['birth_rate'].notna()]

print(f"Birth observations: {len(births):,}")
print(f"Counties: {births['fips'].nunique():,}")
print(f"Years: {births['year'].min()} - {births['year'].max()}")
print(f"Birth rate range: {births['birth_rate'].min():.1f} - {births['birth_rate'].max():.1f}")

# =============================================================================
# 2. PROCESS BEA COUNTY INCOME DATA
# =============================================================================
print("\n" + "="*70)
print("2. PROCESSING BEA COUNTY INCOME DATA")
print("="*70)

bea_raw = pd.read_csv(f'{DATA_DIR}/bea_county_income_raw.csv', low_memory=False)
bea_pci = bea_raw[bea_raw['LineCode'] == 3.0].copy()

# Clean FIPS
bea_pci['fips'] = bea_pci['GeoFIPS'].astype(str).str.strip().str.replace('"', '').str.replace(' ', '')
bea_pci['fips'] = bea_pci['fips'].str.zfill(5)
bea_pci = bea_pci[bea_pci['fips'].str.len() == 5]
bea_pci = bea_pci[~bea_pci['fips'].str.endswith('000')]
bea_pci = bea_pci[bea_pci['fips'] != '00000']

# Reshape to long
year_cols = [str(y) for y in range(2010, 2024)]
available_years = [c for c in year_cols if c in bea_pci.columns]

bea_long = bea_pci.melt(
    id_vars=['fips'],
    value_vars=available_years,
    var_name='year',
    value_name='pci'
)
bea_long['year'] = bea_long['year'].astype(int)
bea_long['pci'] = pd.to_numeric(bea_long['pci'], errors='coerce')
bea_long = bea_long.dropna(subset=['pci'])
bea_long = bea_long[bea_long['pci'] > 0]

print(f"Income observations: {len(bea_long):,}")
print(f"Counties: {bea_long['fips'].nunique():,}")

# =============================================================================
# 3. LOAD STATE-LEVEL UNEMPLOYMENT
# =============================================================================
print("\n" + "="*70)
print("3. LOADING STATE UNEMPLOYMENT (as proxy for county)")
print("="*70)

state_unemp = pd.read_csv(f'{DATA_DIR}/state_unemployment.csv')
print(f"State unemployment: {len(state_unemp):,} obs")

# Map state names to codes
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

# =============================================================================
# 4. CREATE COUNTY PANEL
# =============================================================================
print("\n" + "="*70)
print("4. CREATING COUNTY PANEL")
print("="*70)

# Start with births
panel = births.copy()

# Merge with per capita income
panel = panel.merge(bea_long, on=['fips', 'year'], how='left')

# Add state abbreviation for unemployment merge
panel['state'] = panel['state_name'].map(state_abbrev)

# Merge with state unemployment
panel = panel.merge(state_unemp[['state', 'year', 'unemployment_rate']], on=['state', 'year'], how='left')

# Create variables
panel['log_birth_rate'] = np.log(panel['birth_rate'])
panel['log_pci'] = np.log(panel['pci'])
panel['log_population'] = np.log(panel['population'])

# Add lagged unemployment for placebo test
panel = panel.sort_values(['fips', 'year'])
panel['unemployment_lead1'] = panel.groupby('fips')['unemployment_rate'].shift(-1)

print(f"Panel observations: {len(panel):,}")
print(f"Counties: {panel['fips'].nunique():,}")
print(f"Years: {panel['year'].min()} - {panel['year'].max()}")
print(f"With unemployment: {panel['unemployment_rate'].notna().sum():,}")
print(f"With income: {panel['pci'].notna().sum():,}")

# Filter to complete observations
panel_complete = panel.dropna(subset=['log_birth_rate', 'unemployment_rate', 'log_pci'])
print(f"\nComplete observations: {len(panel_complete):,}")
print(f"Counties: {panel_complete['fips'].nunique():,}")

# =============================================================================
# 5. REGRESSIONS WITH COUNTY AND YEAR FIXED EFFECTS
# =============================================================================
print("\n" + "="*70)
print("5. COUNTY-LEVEL REGRESSIONS")
print("="*70)

def run_county_regression(data, y_var, x_vars, name, show_fe=False):
    """Run regression with county and year FE"""
    subset = data.dropna(subset=[y_var] + x_vars)

    if len(subset) < 100:
        print(f"{name}: Insufficient observations ({len(subset)})")
        return None

    # County fixed effects
    county_dum = pd.get_dummies(subset['fips'], prefix='cty', drop_first=True).astype(float)
    # Year fixed effects
    year_dum = pd.get_dummies(subset['year'].astype(int), prefix='yr', drop_first=True).astype(float)

    X = pd.concat([subset[x_vars], county_dum, year_dum], axis=1)
    X = sm.add_constant(X)
    y = subset[y_var]

    # Cluster standard errors at state level
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': subset['state']})

    print(f"\n{name}")
    print(f"N = {len(subset):,}, Counties = {subset['fips'].nunique():,}, R² = {model.rsquared:.4f}")
    for var in x_vars:
        if var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            t = model.tvalues[var]
            p = model.pvalues[var]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {var:25s}: {coef:10.5f} (t={t:6.2f}){sig}")

    return model

# Model 1: Unemployment only
run_county_regression(panel_complete, 'log_birth_rate', ['unemployment_rate'],
                     "Model 1: Unemployment only")

# Model 2: Add log per capita income
run_county_regression(panel_complete, 'log_birth_rate', ['unemployment_rate', 'log_pci'],
                     "Model 2: Add log per capita income")

# Model 3: Add log population
run_county_regression(panel_complete, 'log_birth_rate', ['unemployment_rate', 'log_pci', 'log_population'],
                     "Model 3: Add log population")

# =============================================================================
# 6. PLACEBO TEST: FUTURE UNEMPLOYMENT
# =============================================================================
print("\n" + "="*70)
print("6. PLACEBO TEST: FUTURE UNEMPLOYMENT (t+1)")
print("="*70)

# Subset with lead unemployment
panel_placebo = panel_complete.dropna(subset=['unemployment_lead1'])
print(f"Observations with t+1 unemployment: {len(panel_placebo):,}")

# Current unemployment only
run_county_regression(panel_placebo, 'log_birth_rate', ['unemployment_rate'],
                     "Placebo baseline: Current unemployment")

# Future unemployment only
run_county_regression(panel_placebo, 'log_birth_rate', ['unemployment_lead1'],
                     "Placebo: Future unemployment (t+1) only")

# Both current and future
run_county_regression(panel_placebo, 'log_birth_rate', ['unemployment_rate', 'unemployment_lead1'],
                     "Placebo: Current + Future unemployment")

# =============================================================================
# 7. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("7. SUMMARY STATISTICS")
print("="*70)

summary_vars = ['birth_rate', 'births', 'population', 'unemployment_rate', 'pci']
print("\nVariable             Mean      Std      Min      Max        N")
print("-" * 70)
for var in summary_vars:
    if var in panel_complete.columns:
        s = panel_complete[var].describe()
        print(f"{var:20s} {s['mean']:9.2f} {s['std']:8.2f} {s['min']:8.0f} {s['max']:9.0f} {s['count']:8.0f}")

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("8. SAVING PANEL DATA")
print("="*70)

panel.to_csv(f'{DATA_DIR}/county_panel_full.csv', index=False)
print(f"Saved: data/county_panel_full.csv ({len(panel):,} rows)")
