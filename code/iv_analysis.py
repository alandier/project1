"""
Instrumental Variable Analysis: Unemployment and Fertility
==========================================================
Four instrument strategies:
1. Housing price shocks (FHFA HPI)
2. Oil price shocks × state oil dependence
3. Bartik (manufacturing employment shock × state mfg share)
4. China trade shock (ADH-style)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/landieraugustin/project1/data'

print("="*70)
print("INSTRUMENTAL VARIABLE ANALYSIS")
print("="*70)

# =============================================================================
# 1. LOAD BASE DATA
# =============================================================================
print("\n1. LOADING BASE DATA...")

# Birth rates
births = pd.read_csv(f'{DATA_DIR}/births_by_state.csv')
print(f"   Births: {len(births)} obs")

# Unemployment
unemp = pd.read_csv(f'{DATA_DIR}/state_unemployment.csv')
print(f"   Unemployment: {len(unemp)} obs")

# Merge base data
df = births.merge(unemp, on=['state', 'year'])
df['log_birth_rate'] = np.log(df['birth_rate'])
print(f"   Base panel: {len(df)} state-years")

# =============================================================================
# 2. INSTRUMENT 1: HOUSING PRICE SHOCK
# =============================================================================
print("\n" + "="*70)
print("2. HOUSING PRICE SHOCK INSTRUMENT")
print("="*70)

# Load FHFA HPI
hpi = pd.read_csv(f'{DATA_DIR}/fhfa_state_hpi.csv', header=None,
                  names=['state', 'year', 'quarter', 'hpi'])
print(f"   HPI raw: {len(hpi)} quarterly obs")

# Annual average
hpi_annual = hpi.groupby(['state', 'year'])['hpi'].mean().reset_index()
print(f"   HPI annual: {len(hpi_annual)} obs")

# Calculate HPI growth
hpi_annual = hpi_annual.sort_values(['state', 'year'])
hpi_annual['hpi_growth'] = hpi_annual.groupby('state')['hpi'].pct_change()

# Housing bust measure: peak-to-trough 2006-2009
hpi_2006 = hpi_annual[hpi_annual['year'] == 2006][['state', 'hpi']].rename(columns={'hpi': 'hpi_2006'})
hpi_2009 = hpi_annual[hpi_annual['year'] == 2009][['state', 'hpi']].rename(columns={'hpi': 'hpi_2009'})
housing_bust = hpi_2006.merge(hpi_2009, on='state')
housing_bust['bust_severity'] = (housing_bust['hpi_2009'] - housing_bust['hpi_2006']) / housing_bust['hpi_2006']
print(f"   Housing bust states: {len(housing_bust)}")
print(f"   Bust severity range: {housing_bust['bust_severity'].min():.2%} to {housing_bust['bust_severity'].max():.2%}")

# Merge HPI growth with panel
df = df.merge(hpi_annual[['state', 'year', 'hpi_growth']], on=['state', 'year'], how='left')

# Also merge bust severity for interaction
df = df.merge(housing_bust[['state', 'bust_severity']], on='state', how='left')

print(f"   Panel with HPI: {df['hpi_growth'].notna().sum()} obs with HPI growth")

# =============================================================================
# 3. INSTRUMENT 2: OIL PRICE SHOCK
# =============================================================================
print("\n" + "="*70)
print("3. OIL PRICE SHOCK INSTRUMENT")
print("="*70)

# Load oil prices
oil = pd.read_csv(f'{DATA_DIR}/wti_oil_prices.csv')
oil.columns = ['date', 'oil_price']
oil['date'] = pd.to_datetime(oil['date'])
oil['year'] = oil['date'].dt.year

# Annual average oil price
oil_annual = oil.groupby('year')['oil_price'].mean().reset_index()
oil_annual['oil_growth'] = oil_annual['oil_price'].pct_change()
print(f"   Oil prices: {oil_annual['year'].min()} - {oil_annual['year'].max()}")

# Define oil-dependent states (top oil producers)
oil_states = ['TX', 'ND', 'NM', 'OK', 'AK', 'CO', 'WY', 'CA', 'LA', 'KS']
df['oil_state'] = df['state'].isin(oil_states).astype(int)
print(f"   Oil states: {oil_states}")

# Merge oil prices
df = df.merge(oil_annual[['year', 'oil_growth']], on='year', how='left')

# Create oil shock instrument: oil_growth × oil_state
df['oil_shock'] = df['oil_growth'] * df['oil_state']
print(f"   Oil shock instrument created")

# =============================================================================
# 4. INSTRUMENT 3: BARTIK (MANUFACTURING SHOCK)
# =============================================================================
print("\n" + "="*70)
print("4. BARTIK INSTRUMENT (MANUFACTURING)")
print("="*70)

# Load national manufacturing employment
mfg = pd.read_csv(f'{DATA_DIR}/fred_manufacturing_emp.csv')
mfg.columns = ['date', 'mfg_emp']
mfg['date'] = pd.to_datetime(mfg['date'])
mfg['year'] = mfg['date'].dt.year

# Annual manufacturing employment
mfg_annual = mfg.groupby('year')['mfg_emp'].mean().reset_index()
mfg_annual['mfg_growth'] = mfg_annual['mfg_emp'].pct_change()
print(f"   Manufacturing: {mfg_annual['year'].min()} - {mfg_annual['year'].max()}")

# State manufacturing shares (approximation based on 2000 data)
# Using rough estimates for major manufacturing states
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
df['mfg_share'] = df['state'].map(mfg_share).fillna(0.05)

# Merge manufacturing growth
df = df.merge(mfg_annual[['year', 'mfg_growth']], on='year', how='left')

# Create Bartik instrument: mfg_growth × state_mfg_share
df['bartik_shock'] = df['mfg_growth'] * df['mfg_share']
print(f"   Bartik instrument created")

# =============================================================================
# 5. INSTRUMENT 4: CHINA TRADE SHOCK
# =============================================================================
print("\n" + "="*70)
print("5. CHINA TRADE SHOCK INSTRUMENT")
print("="*70)

# China shock measure by state (approximation based on ADH methodology)
# States with high exposure to Chinese import competition (manufacturing-heavy)
# Based on Autor, Dorn, Hanson estimates for 1999-2011 period
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
df['china_exposure'] = df['state'].map(china_exposure).fillna(0.03)

# China imports grew most rapidly 2001-2007
# Create time-varying shock: exposure × post-2001 indicator × years since 2001
df['post_wto'] = (df['year'] >= 2001).astype(int)
df['years_since_wto'] = np.maximum(0, df['year'] - 2001)
df['china_shock'] = df['china_exposure'] * df['years_since_wto'] * (df['year'] <= 2011).astype(int)
print(f"   China shock instrument created (peaks 2001-2011)")

# =============================================================================
# 6. RUN OLS AND IV REGRESSIONS
# =============================================================================
print("\n" + "="*70)
print("6. REGRESSION RESULTS")
print("="*70)

# Focus on years with good instrument coverage
df_reg = df[(df['year'] >= 1995) & (df['year'] <= 2023)].copy()
df_reg = df_reg.dropna(subset=['log_birth_rate', 'unemployment_rate'])
print(f"   Regression sample: {len(df_reg)} obs")

# Create state and year dummies
df_reg['state_code'] = pd.Categorical(df_reg['state']).codes
df_reg['year_code'] = pd.Categorical(df_reg['year']).codes

# Demean for FE (within transformation)
for var in ['log_birth_rate', 'unemployment_rate', 'hpi_growth', 'oil_shock', 'bartik_shock', 'china_shock']:
    if var in df_reg.columns:
        df_reg[f'{var}_dm'] = df_reg.groupby('state')[var].transform(lambda x: x - x.mean())
        df_reg[f'{var}_dm'] = df_reg.groupby('year')[f'{var}_dm'].transform(lambda x: x - x.mean())

print("\n" + "-"*70)
print("OLS: Log Birth Rate on Unemployment (State + Year FE)")
print("-"*70)

# OLS with FE
st_dum = pd.get_dummies(df_reg['state'], prefix='st', drop_first=True).astype(float)
yr_dum = pd.get_dummies(df_reg['year'], prefix='yr', drop_first=True).astype(float)

X_ols = pd.concat([df_reg[['unemployment_rate']].reset_index(drop=True),
                   st_dum.reset_index(drop=True),
                   yr_dum.reset_index(drop=True)], axis=1)
X_ols = sm.add_constant(X_ols)
y = df_reg['log_birth_rate'].reset_index(drop=True)

ols_model = sm.OLS(y, X_ols).fit(cov_type='cluster', cov_kwds={'groups': df_reg['state'].reset_index(drop=True)})
print(f"OLS coefficient on unemployment: {ols_model.params['unemployment_rate']:.5f}")
print(f"   t-statistic: {ols_model.tvalues['unemployment_rate']:.2f}")
print(f"   N = {len(df_reg)}, R² = {ols_model.rsquared:.4f}")

# =============================================================================
# IV REGRESSIONS
# =============================================================================

def run_iv_2sls(df, endog, exog_list, instrument, instrument_name, controls=None):
    """Run 2SLS IV regression"""
    # Prepare data
    data = df.dropna(subset=[endog] + exog_list + [instrument])

    if len(data) < 100:
        print(f"\n{instrument_name}: Insufficient observations ({len(data)})")
        return None

    # State and year dummies
    st_dum = pd.get_dummies(data['state'], prefix='st', drop_first=True).astype(float)
    yr_dum = pd.get_dummies(data['year'], prefix='yr', drop_first=True).astype(float)

    y = data[endog].values
    X_endog = data[exog_list].values
    Z = data[[instrument]].values

    # Add FE to both X and Z
    fe_matrix = pd.concat([st_dum.reset_index(drop=True), yr_dum.reset_index(drop=True)], axis=1).values

    X_full = np.column_stack([np.ones(len(data)), X_endog, fe_matrix])
    Z_full = np.column_stack([np.ones(len(data)), Z, fe_matrix])

    # First stage
    first_stage = sm.OLS(X_endog, Z_full).fit()
    X_hat = first_stage.fittedvalues
    f_stat = first_stage.fvalue

    # Second stage
    X_2sls = np.column_stack([np.ones(len(data)), X_hat, fe_matrix])
    second_stage = sm.OLS(y, X_2sls).fit()

    print(f"\n{instrument_name}")
    print(f"   First stage F-stat: {f_stat:.2f}")
    print(f"   IV coefficient: {second_stage.params[1]:.5f}")
    print(f"   N = {len(data)}")

    return {'coef': second_stage.params[1], 'f_stat': f_stat, 'n': len(data)}

print("\n" + "="*70)
print("IV REGRESSIONS")
print("="*70)

# IV 1: Housing shock
print("\n" + "-"*70)
print("IV 1: HOUSING PRICE GROWTH")
print("-"*70)
df_iv1 = df_reg.dropna(subset=['hpi_growth'])
run_iv_2sls(df_iv1, 'log_birth_rate', ['unemployment_rate'], 'hpi_growth',
            "Housing Price Growth → Unemployment → Birth Rate")

# IV 2: Oil shock
print("\n" + "-"*70)
print("IV 2: OIL PRICE SHOCK")
print("-"*70)
df_iv2 = df_reg.dropna(subset=['oil_shock'])
run_iv_2sls(df_iv2, 'log_birth_rate', ['unemployment_rate'], 'oil_shock',
            "Oil Shock (Oil Price × Oil State) → Unemployment → Birth Rate")

# IV 3: Bartik
print("\n" + "-"*70)
print("IV 3: BARTIK (MANUFACTURING)")
print("-"*70)
df_iv3 = df_reg.dropna(subset=['bartik_shock'])
run_iv_2sls(df_iv3, 'log_birth_rate', ['unemployment_rate'], 'bartik_shock',
            "Bartik (Mfg Growth × State Mfg Share) → Unemployment → Birth Rate")

# IV 4: China shock
print("\n" + "-"*70)
print("IV 4: CHINA TRADE SHOCK")
print("-"*70)
df_iv4 = df_reg[(df_reg['year'] >= 2001) & (df_reg['year'] <= 2015)]
df_iv4 = df_iv4.dropna(subset=['china_shock'])
run_iv_2sls(df_iv4, 'log_birth_rate', ['unemployment_rate'], 'china_shock',
            "China Shock → Unemployment → Birth Rate (2001-2015)")

# =============================================================================
# 7. REDUCED FORM REGRESSIONS
# =============================================================================
print("\n" + "="*70)
print("7. REDUCED FORM: INSTRUMENTS → BIRTH RATE")
print("="*70)

def run_reduced_form(df, y_var, instrument, instrument_name):
    """Run reduced form regression"""
    data = df.dropna(subset=[y_var, instrument])

    st_dum = pd.get_dummies(data['state'], prefix='st', drop_first=True).astype(float)
    yr_dum = pd.get_dummies(data['year'], prefix='yr', drop_first=True).astype(float)

    X = pd.concat([data[[instrument]].reset_index(drop=True),
                   st_dum.reset_index(drop=True),
                   yr_dum.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = data[y_var].reset_index(drop=True)

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data['state'].reset_index(drop=True)})

    print(f"\n{instrument_name}")
    print(f"   Coefficient: {model.params[instrument]:.5f}")
    print(f"   t-stat: {model.tvalues[instrument]:.2f}")
    print(f"   N = {len(data)}")

    return model

# Reduced form for each instrument
run_reduced_form(df_reg, 'log_birth_rate', 'hpi_growth', "Housing Price Growth → Birth Rate")
run_reduced_form(df_reg, 'log_birth_rate', 'oil_shock', "Oil Shock → Birth Rate")
run_reduced_form(df_reg, 'log_birth_rate', 'bartik_shock', "Bartik Shock → Birth Rate")

df_china = df_reg[(df_reg['year'] >= 2001) & (df_reg['year'] <= 2015)]
run_reduced_form(df_china, 'log_birth_rate', 'china_shock', "China Shock → Birth Rate (2001-2015)")

# =============================================================================
# 8. FIRST STAGE: INSTRUMENTS → UNEMPLOYMENT
# =============================================================================
print("\n" + "="*70)
print("8. FIRST STAGE: INSTRUMENTS → UNEMPLOYMENT")
print("="*70)

run_reduced_form(df_reg, 'unemployment_rate', 'hpi_growth', "Housing Price Growth → Unemployment")
run_reduced_form(df_reg, 'unemployment_rate', 'oil_shock', "Oil Shock → Unemployment")
run_reduced_form(df_reg, 'unemployment_rate', 'bartik_shock', "Bartik Shock → Unemployment")
run_reduced_form(df_china, 'unemployment_rate', 'china_shock', "China Shock → Unemployment (2001-2015)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Interpretation:
- OLS estimate: Effect of unemployment on birth rates (potentially biased)
- First stage: Does the instrument predict unemployment? (need F > 10)
- Reduced form: Does the instrument predict birth rates directly?
- IV estimate: Causal effect if instrument is valid (exogenous, relevant)
""")
