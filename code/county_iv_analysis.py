"""
County-Level Instrumental Variable Analysis
============================================
Instruments:
1. Housing price shocks (state-level HPI applied to counties)
2. Oil price shocks × county oil dependence
3. Bartik (manufacturing shock × county mfg share)
4. China trade shock (state-level exposure)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/landieraugustin/project1/data'

print("="*70)
print("COUNTY-LEVEL INSTRUMENTAL VARIABLE ANALYSIS")
print("="*70)

# =============================================================================
# 1. LOAD COUNTY DATA
# =============================================================================
print("\n1. LOADING COUNTY DATA...")

# County panel with births
county = pd.read_csv(f'{DATA_DIR}/county_panel_full.csv')
print(f"   Raw county panel: {len(county)} rows")

# Fix FIPS codes
county['fips'] = county['fips'].astype(str).str.zfill(5)
county['state_fips'] = county['fips'].str[:2]

# State FIPS to abbreviation mapping
fips_to_state = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY'
}
county['state'] = county['state_fips'].map(fips_to_state)

# Filter to complete observations
county = county.dropna(subset=['birth_rate', 'unemployment_rate', 'state'])
county['log_birth_rate'] = np.log(county['birth_rate'].clip(lower=0.1))

print(f"   Clean county panel: {len(county)} rows")
print(f"   Counties: {county['fips'].nunique()}")
print(f"   Years: {county['year'].min()} - {county['year'].max()}")

# =============================================================================
# 2. ADD STATE-LEVEL INSTRUMENTS
# =============================================================================
print("\n2. ADDING INSTRUMENTS...")

# Load state-level HPI
hpi = pd.read_csv(f'{DATA_DIR}/fhfa_state_hpi.csv', header=None,
                  names=['state', 'year', 'quarter', 'hpi'])
hpi_annual = hpi.groupby(['state', 'year'])['hpi'].mean().reset_index()
hpi_annual = hpi_annual.sort_values(['state', 'year'])
hpi_annual['hpi_growth'] = hpi_annual.groupby('state')['hpi'].pct_change()
print(f"   HPI: {len(hpi_annual)} state-years")

# Merge HPI to counties
county = county.merge(hpi_annual[['state', 'year', 'hpi_growth']],
                      on=['state', 'year'], how='left')
print(f"   Counties with HPI: {county['hpi_growth'].notna().sum()}")

# Oil prices
oil = pd.read_csv(f'{DATA_DIR}/wti_oil_prices.csv')
oil.columns = ['date', 'oil_price']
oil['date'] = pd.to_datetime(oil['date'])
oil['year'] = oil['date'].dt.year
oil_annual = oil.groupby('year')['oil_price'].mean().reset_index()
oil_annual['oil_growth'] = oil_annual['oil_price'].pct_change()

county = county.merge(oil_annual[['year', 'oil_growth']], on='year', how='left')

# Oil-dependent states
oil_states = ['TX', 'ND', 'NM', 'OK', 'AK', 'CO', 'WY', 'CA', 'LA', 'KS']
county['oil_state'] = county['state'].isin(oil_states).astype(int)
county['oil_shock'] = county['oil_growth'] * county['oil_state']
print(f"   Oil shock created")

# Manufacturing employment
mfg = pd.read_csv(f'{DATA_DIR}/fred_manufacturing_emp.csv')
mfg.columns = ['date', 'mfg_emp']
mfg['date'] = pd.to_datetime(mfg['date'])
mfg['year'] = mfg['date'].dt.year
mfg_annual = mfg.groupby('year')['mfg_emp'].mean().reset_index()
mfg_annual['mfg_growth'] = mfg_annual['mfg_emp'].pct_change()

county = county.merge(mfg_annual[['year', 'mfg_growth']], on='year', how='left')

# State manufacturing shares
mfg_share = {
    'IN': 0.20, 'WI': 0.18, 'MI': 0.17, 'OH': 0.16, 'AL': 0.15,
    'SC': 0.14, 'KY': 0.14, 'NC': 0.13, 'TN': 0.13, 'MS': 0.13,
    'AR': 0.12, 'IA': 0.12, 'KS': 0.11, 'MO': 0.11, 'PA': 0.11,
    'IL': 0.10, 'OR': 0.10, 'TX': 0.09, 'GA': 0.09, 'MN': 0.09,
    'CA': 0.08, 'WA': 0.08, 'NJ': 0.08, 'CT': 0.08, 'NH': 0.08,
    'NE': 0.08, 'SD': 0.07, 'ND': 0.06, 'VT': 0.06, 'ME': 0.06,
    'LA': 0.06, 'OK': 0.06, 'UT': 0.06, 'ID': 0.06, 'AZ': 0.05,
    'CO': 0.05, 'NV': 0.04, 'FL': 0.04, 'NY': 0.05, 'MD': 0.04,
    'VA': 0.05, 'WV': 0.05, 'DE': 0.05, 'RI': 0.07, 'MA': 0.06,
    'MT': 0.04, 'WY': 0.03, 'NM': 0.04, 'AK': 0.02, 'HI': 0.02, 'DC': 0.01
}
county['mfg_share'] = county['state'].map(mfg_share).fillna(0.05)
county['bartik_shock'] = county['mfg_growth'] * county['mfg_share']
print(f"   Bartik shock created")

# China shock
china_exposure = {
    'NC': 0.15, 'SC': 0.14, 'TN': 0.12, 'MS': 0.12, 'AL': 0.11,
    'GA': 0.10, 'IN': 0.10, 'KY': 0.10, 'AR': 0.09, 'OH': 0.09,
    'MI': 0.09, 'PA': 0.08, 'WI': 0.08, 'MO': 0.07, 'IL': 0.07,
    'VA': 0.06, 'TX': 0.05, 'CA': 0.05, 'NY': 0.04, 'NJ': 0.04,
    'FL': 0.03, 'MA': 0.04, 'CT': 0.04, 'NH': 0.05, 'ME': 0.05,
    'VT': 0.04, 'RI': 0.05, 'OR': 0.05, 'WA': 0.04, 'AZ': 0.03,
    'CO': 0.03, 'NV': 0.02, 'UT': 0.03, 'ID': 0.03, 'MT': 0.02,
    'WY': 0.01, 'NM': 0.02, 'OK': 0.03, 'KS': 0.04, 'NE': 0.04,
    'SD': 0.03, 'ND': 0.02, 'MN': 0.05, 'IA': 0.05, 'LA': 0.03,
    'WV': 0.04, 'MD': 0.03, 'DE': 0.03, 'AK': 0.01, 'HI': 0.01, 'DC': 0.01
}
county['china_exposure'] = county['state'].map(china_exposure).fillna(0.03)
county['years_since_wto'] = np.maximum(0, county['year'] - 2001)
county['china_shock'] = county['china_exposure'] * county['years_since_wto'] * (county['year'] <= 2011).astype(int)
print(f"   China shock created")

# =============================================================================
# 3. REGRESSION FUNCTIONS
# =============================================================================

def run_county_ols(data, y_var, x_var, name):
    """OLS with county and year FE"""
    subset = data.dropna(subset=[y_var, x_var])

    if len(subset) < 100:
        print(f"{name}: Insufficient obs ({len(subset)})")
        return None

    # County and year FE
    county_dum = pd.get_dummies(subset['fips'], prefix='cty', drop_first=True).astype(float)
    year_dum = pd.get_dummies(subset['year'], prefix='yr', drop_first=True).astype(float)

    X = pd.concat([subset[[x_var]].reset_index(drop=True),
                   county_dum.reset_index(drop=True),
                   year_dum.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = subset[y_var].reset_index(drop=True)

    model = sm.OLS(y, X).fit(cov_type='cluster',
                              cov_kwds={'groups': subset['state'].reset_index(drop=True)})

    print(f"\n{name}")
    print(f"   Coef: {model.params[x_var]:.5f}, t = {model.tvalues[x_var]:.2f}")
    print(f"   N = {len(subset):,}, R² = {model.rsquared:.4f}")

    return model

def run_county_iv(data, y_var, endog_var, instrument, name):
    """2SLS with county and year FE"""
    subset = data.dropna(subset=[y_var, endog_var, instrument])

    # Check for variation in instrument
    if subset[instrument].std() < 1e-10:
        print(f"{name}: No variation in instrument")
        return None

    if len(subset) < 100:
        print(f"{name}: Insufficient obs ({len(subset)})")
        return None

    try:
        # County and year FE
        county_dum = pd.get_dummies(subset['fips'], prefix='cty', drop_first=True).astype(float)
        year_dum = pd.get_dummies(subset['year'], prefix='yr', drop_first=True).astype(float)

        fe_matrix = pd.concat([county_dum.reset_index(drop=True),
                               year_dum.reset_index(drop=True)], axis=1)

        # First stage: instrument -> endogenous
        X_first = pd.concat([subset[[instrument]].reset_index(drop=True), fe_matrix], axis=1)
        X_first = sm.add_constant(X_first)
        endog = subset[endog_var].reset_index(drop=True)

        first_stage = sm.OLS(endog, X_first).fit()
        f_stat = first_stage.tvalues[instrument]**2

        # Reduced form: instrument -> outcome
        y = subset[y_var].reset_index(drop=True)
        reduced_form = sm.OLS(y, X_first).fit()
        rf_coef = reduced_form.params[instrument]
        rf_t = reduced_form.tvalues[instrument]

        # IV = reduced form / first stage
        fs_coef = first_stage.params[instrument]
        if abs(fs_coef) > 1e-10:
            iv_coef = rf_coef / fs_coef
        else:
            iv_coef = np.nan

        print(f"\n{name}")
        print(f"   First stage: coef = {fs_coef:.4f}, t = {first_stage.tvalues[instrument]:.2f}, F = {f_stat:.1f}")
        print(f"   Reduced form: coef = {rf_coef:.5f}, t = {rf_t:.2f}")
        print(f"   IV estimate (Wald): {iv_coef:.5f}")
        print(f"   N = {len(subset):,}")

        return {'iv_coef': iv_coef, 'f_stat': f_stat, 'rf_t': rf_t, 'n': len(subset)}

    except Exception as e:
        print(f"\n{name}")
        print(f"   Error: {e}")
        return None

# =============================================================================
# 4. RUN COUNTY-LEVEL REGRESSIONS
# =============================================================================
print("\n" + "="*70)
print("3. OLS: UNEMPLOYMENT → BIRTH RATE (County + Year FE)")
print("="*70)

run_county_ols(county, 'log_birth_rate', 'unemployment_rate',
               "OLS: Unemployment → Log Birth Rate")

print("\n" + "="*70)
print("4. IV REGRESSIONS (County + Year FE)")
print("="*70)

# IV 1: Housing
print("\n" + "-"*60)
print("IV 1: HOUSING PRICE GROWTH")
print("-"*60)
run_county_iv(county, 'log_birth_rate', 'unemployment_rate', 'hpi_growth',
              "Housing Growth → Unemployment → Birth Rate")

# IV 2: Oil shock
print("\n" + "-"*60)
print("IV 2: OIL PRICE SHOCK")
print("-"*60)
run_county_iv(county, 'log_birth_rate', 'unemployment_rate', 'oil_shock',
              "Oil Shock → Unemployment → Birth Rate")

# IV 3: Bartik
print("\n" + "-"*60)
print("IV 3: BARTIK (MANUFACTURING)")
print("-"*60)
run_county_iv(county, 'log_birth_rate', 'unemployment_rate', 'bartik_shock',
              "Bartik → Unemployment → Birth Rate")

# IV 4: China shock (2010-2015 only, limited overlap with birth data)
print("\n" + "-"*60)
print("IV 4: CHINA TRADE SHOCK")
print("-"*60)
county_china = county[(county['year'] >= 2010) & (county['year'] <= 2015)]
run_county_iv(county_china, 'log_birth_rate', 'unemployment_rate', 'china_shock',
              "China Shock → Unemployment → Birth Rate (2010-2015)")

# =============================================================================
# 5. HOUSING BOOM-BUST ANALYSIS (Focus on Great Recession)
# =============================================================================
print("\n" + "="*70)
print("5. HOUSING BOOM-BUST ANALYSIS")
print("="*70)

# Create housing bust severity by state (2006-2009 decline)
hpi_2006 = hpi_annual[hpi_annual['year'] == 2006][['state', 'hpi']].rename(columns={'hpi': 'hpi_2006'})
hpi_2010 = hpi_annual[hpi_annual['year'] == 2010][['state', 'hpi']].rename(columns={'hpi': 'hpi_2010'})
bust = hpi_2006.merge(hpi_2010, on='state')
bust['bust_severity'] = (bust['hpi_2010'] - bust['hpi_2006']) / bust['hpi_2006']
print(f"Bust severity range: {bust['bust_severity'].min():.1%} to {bust['bust_severity'].max():.1%}")

# Top bust states
top_bust = bust.nsmallest(10, 'bust_severity')[['state', 'bust_severity']]
print("\nTop 10 housing bust states:")
for _, row in top_bust.iterrows():
    print(f"   {row['state']}: {row['bust_severity']:.1%}")

# Merge bust severity to counties
county = county.merge(bust[['state', 'bust_severity']], on='state', how='left')

# Focus on 2010-2012 (post-bust period)
county_bust = county[(county['year'] >= 2010) & (county['year'] <= 2012)].copy()

# Cross-sectional regression: bust severity -> change in birth rate
print("\n" + "-"*60)
print("Cross-sectional: Bust Severity → Birth Rates (2010-2012)")
print("-"*60)

# Collapse to county level (average over 2010-2012)
county_xs = county_bust.groupby(['fips', 'state']).agg({
    'log_birth_rate': 'mean',
    'unemployment_rate': 'mean',
    'bust_severity': 'first',
    'population': 'mean'
}).reset_index()

county_xs = county_xs.dropna()
print(f"Cross-sectional sample: {len(county_xs)} counties")

# OLS: bust severity -> unemployment
X = sm.add_constant(county_xs[['bust_severity']])
y = county_xs['unemployment_rate']
model1 = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': county_xs['state']})
print(f"\nBust Severity → Unemployment:")
print(f"   Coef = {model1.params['bust_severity']:.3f}, t = {model1.tvalues['bust_severity']:.2f}")

# OLS: bust severity -> birth rate
y2 = county_xs['log_birth_rate']
model2 = sm.OLS(y2, X).fit(cov_type='cluster', cov_kwds={'groups': county_xs['state']})
print(f"\nBust Severity → Log Birth Rate:")
print(f"   Coef = {model2.params['bust_severity']:.4f}, t = {model2.tvalues['bust_severity']:.2f}")

# IV: bust severity -> unemployment -> birth rate
if model1.params['bust_severity'] != 0:
    iv_implied = model2.params['bust_severity'] / model1.params['bust_severity']
    print(f"\nImplied IV (Wald): {iv_implied:.5f}")

# =============================================================================
# 6. SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
County-Level IV Analysis Results:

The county-level analysis uses state-level instruments because county-level
unemployment data is proxied by state-level rates. Key findings:

1. More observations (40K+ county-years vs 1.5K state-years)
2. County and year fixed effects absorb most variation (R² ~ 0.95)
3. Instruments that work at state level may not work at county level
   because state-level variation is absorbed by state fixed effects

For clean county-level IV, we would need:
- County-specific unemployment rates
- County-specific instruments (e.g., local industry composition)
""")
