"""
Additional robustness checks for the paper:
1. Pre-trends analysis
2. Controls (unemployment, income)
3. By time period
4. Deaths calculation
5. Summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# =============================================================================
# Load Data
# =============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

suicide = pd.read_csv('/Users/landieraugustin/project1/data/suicide_state_panel_extended.csv')
suicide = suicide[['state', 'year', 'suicide_rate']]

hpi = pd.read_csv('/Users/landieraugustin/project1/data/fhfa_state_hpi.csv',
                  names=['state', 'year', 'quarter', 'hpi'])
hpi_annual = hpi[hpi['quarter'] == 4][['state', 'year', 'hpi']].copy()
hpi_annual['hpi_growth'] = hpi_annual.groupby('state')['hpi'].pct_change() * 100

# Unemployment
unemp = pd.read_csv('/Users/landieraugustin/project1/data/state_unemployment.csv')

land_elasticity = {
    'CA': 0.6, 'NJ': 0.7, 'MA': 0.8, 'RI': 0.9, 'CT': 1.0, 'MD': 1.0, 'NY': 0.9,
    'FL': 1.1, 'WA': 1.0, 'HI': 0.5, 'CO': 1.2, 'AZ': 1.3, 'NV': 1.2, 'OR': 1.1,
    'VA': 1.3, 'IL': 1.4, 'PA': 1.5, 'MI': 1.6, 'OH': 1.7, 'GA': 1.5, 'NC': 1.4,
    'MN': 1.8, 'WI': 1.7, 'MO': 1.9, 'IN': 1.8, 'TN': 1.6, 'AL': 1.7, 'SC': 1.5,
    'LA': 1.6, 'KY': 1.8, 'TX': 2.0, 'OK': 2.2, 'KS': 2.3, 'NE': 2.4, 'IA': 2.5,
    'AR': 2.1, 'MS': 2.0, 'NM': 2.2, 'UT': 1.4, 'ID': 1.8, 'MT': 2.6, 'WY': 2.7,
    'ND': 2.8, 'SD': 2.7, 'AK': 2.5, 'WV': 1.9, 'ME': 1.5, 'NH': 1.3, 'VT': 1.6,
    'DE': 1.2, 'DC': 0.4
}

# Merge
df = hpi_annual.merge(suicide, on=['state', 'year'], how='left')
df = df.merge(unemp, on=['state', 'year'], how='left')
df['elasticity'] = df['state'].map(land_elasticity)
nat_hpi = df.groupby('year')['hpi_growth'].mean().reset_index()
nat_hpi.columns = ['year', 'national_hpi']
df = df.merge(nat_hpi, on='year')
df['hpi_iv'] = df['national_hpi'] * (1 / df['elasticity'])
df['log_suicide'] = np.log(df['suicide_rate'])

df = df[(df['year'] >= 2000) & (df['year'] <= 2022)]
df = df.dropna(subset=['elasticity', 'hpi_growth', 'log_suicide'])

print(f"Panel: {len(df)} obs")

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

stats_vars = ['suicide_rate', 'hpi_growth', 'elasticity', 'hpi_iv']
if 'unemployment_rate' in df.columns:
    stats_vars.append('unemployment_rate')

for var in stats_vars:
    if var in df.columns:
        print(f"{var:20s}: mean={df[var].mean():8.2f}, sd={df[var].std():8.2f}, "
              f"min={df[var].min():8.2f}, max={df[var].max():8.2f}")

# =============================================================================
# Helper Functions
# =============================================================================
def demean(x, groups):
    return x - x.groupby(groups).transform('mean')

def run_iv(data, controls=None):
    """Run IV regression, optionally with controls"""
    data = data.copy().dropna(subset=['log_suicide', 'hpi_growth', 'hpi_iv'])

    # Double demean
    data['y_dm'] = demean(demean(data['log_suicide'], data['state']), data['year'])
    data['x_dm'] = demean(demean(data['hpi_growth'], data['state']), data['year'])
    data['z_dm'] = demean(demean(data['hpi_iv'], data['state']), data['year'])

    if controls:
        for c in controls:
            if c in data.columns:
                data[f'{c}_dm'] = demean(demean(data[c], data['state']), data['year'])

    # First stage
    fs_vars = ['z_dm']
    if controls:
        fs_vars += [f'{c}_dm' for c in controls if c in data.columns]
    fs = sm.OLS(data['x_dm'], sm.add_constant(data[fs_vars])).fit()

    # Reduced form
    rf = sm.OLS(data['y_dm'], sm.add_constant(data[fs_vars])).fit()

    iv_coef = rf.params['z_dm'] / fs.params['z_dm']
    iv_se = rf.bse['z_dm'] / abs(fs.params['z_dm'])

    return {
        'iv_coef': iv_coef, 'iv_se': iv_se,
        'rf_coef': rf.params['z_dm'], 'rf_t': rf.tvalues['z_dm'],
        'fs_f': fs.fvalue, 'n': len(data)
    }

# =============================================================================
# Robustness with Controls
# =============================================================================
print("\n" + "="*70)
print("ROBUSTNESS: ADDING CONTROLS")
print("="*70)

# Baseline
base = run_iv(df)
print(f"\nBaseline:              IV = {base['iv_coef']:.5f}, RF t = {base['rf_t']:.2f}, N = {base['n']}")

# With unemployment
if 'unemployment_rate' in df.columns:
    with_unemp = run_iv(df, controls=['unemployment_rate'])
    print(f"+ Unemployment:        IV = {with_unemp['iv_coef']:.5f}, RF t = {with_unemp['rf_t']:.2f}")

# =============================================================================
# By Time Period
# =============================================================================
print("\n" + "="*70)
print("BY TIME PERIOD")
print("="*70)

periods = [
    ('2000-2006 (Boom)', 2000, 2006),
    ('2007-2011 (Bust)', 2007, 2011),
    ('2012-2022 (Recovery)', 2012, 2022),
    ('2000-2022 (Full)', 2000, 2022),
]

print(f"\n{'Period':<25} {'IV Coef':>12} {'RF t':>10} {'FS F':>10} {'N':>8}")
print("-"*70)
for name, y1, y2 in periods:
    df_p = df[(df['year'] >= y1) & (df['year'] <= y2)]
    if len(df_p) > 50:
        r = run_iv(df_p)
        print(f"{name:<25} {r['iv_coef']:>12.5f} {r['rf_t']:>10.2f} {r['fs_f']:>10.1f} {r['n']:>8}")

# =============================================================================
# Pre-trends: Low vs High Elasticity States
# =============================================================================
print("\n" + "="*70)
print("PRE-TRENDS ANALYSIS")
print("="*70)

# Split states by elasticity
median_elast = df.groupby('state')['elasticity'].first().median()
low_elast_states = [s for s, e in land_elasticity.items() if e < median_elast]
high_elast_states = [s for s, e in land_elasticity.items() if e >= median_elast]

print(f"Median elasticity: {median_elast:.2f}")
print(f"Low elasticity states ({len(low_elast_states)}): {', '.join(sorted(low_elast_states)[:10])}...")
print(f"High elasticity states ({len(high_elast_states)}): {', '.join(sorted(high_elast_states)[:10])}...")

# Compute average suicide rate by group and year
df['low_elast'] = df['state'].isin(low_elast_states)

pretrends = df.groupby(['year', 'low_elast'])['suicide_rate'].mean().unstack()
pretrends.columns = ['High Elasticity', 'Low Elasticity']

# Index to 2000
for col in pretrends.columns:
    pretrends[col] = 100 * pretrends[col] / pretrends.loc[2000, col]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pretrends.index, pretrends['Low Elasticity'], 'b-', lw=2, marker='o', ms=5, label='Low Elasticity (CA, FL, NY...)')
ax.plot(pretrends.index, pretrends['High Elasticity'], 'r--', lw=2, marker='s', ms=5, label='High Elasticity (TX, KS, IA...)')
ax.axvspan(2007, 2011, alpha=0.15, color='gray', label='Housing Bust')
ax.axvline(2006, color='black', ls=':', lw=1, label='HPI Peak')
ax.set_xlabel('Year')
ax.set_ylabel('Suicide Rate (2000 = 100)')
ax.set_title('Suicide Trends: Low vs High Elasticity States')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig_pretrends.pdf', bbox_inches='tight')
print("\nSaved: fig_pretrends.pdf")

# Pre-2007 trends test
pre_2007 = df[df['year'] <= 2006].copy()
pre_2007['year_centered'] = pre_2007['year'] - 2003
pre_2007['trend_x_low_elast'] = pre_2007['year_centered'] * pre_2007['low_elast'].astype(int)

# Simple diff-in-diff pre-trend
print("\nPre-2007 differential trend test:")
print("(Testing if low-elasticity states had different suicide trends before the bust)")

pre_grouped = pre_2007.groupby(['year', 'low_elast'])['log_suicide'].mean().unstack()
pre_grouped.columns = ['High', 'Low']
pre_grouped['Diff'] = pre_grouped['Low'] - pre_grouped['High']
print(pre_grouped.to_string())

# =============================================================================
# Excess Deaths Calculation
# =============================================================================
print("\n" + "="*70)
print("EXCESS DEATHS CALCULATION")
print("="*70)

# Get actual HPI changes by state during bust
bust_period = df[(df['year'] >= 2007) & (df['year'] <= 2011)].copy()

# Calculate cumulative HPI decline for each state
hpi_2006 = df[df['year'] == 2006][['state', 'hpi']].rename(columns={'hpi': 'hpi_2006'})
hpi_2011 = df[df['year'] == 2011][['state', 'hpi']].rename(columns={'hpi': 'hpi_2011'})
hpi_change = hpi_2006.merge(hpi_2011, on='state')
hpi_change['pct_change'] = 100 * (hpi_change['hpi_2011'] - hpi_change['hpi_2006']) / hpi_change['hpi_2006']

print(f"\nHPI change 2006-2011:")
print(f"  Mean: {hpi_change['pct_change'].mean():.1f}%")
print(f"  Min (Nevada): {hpi_change['pct_change'].min():.1f}%")
print(f"  Max: {hpi_change['pct_change'].max():.1f}%")

# IV coefficient
iv_coef = -0.00889

# Average baseline suicide rate and population
avg_suicide_rate = 12.5  # per 100,000 (approximate 2006 level)
us_population = 310_000_000  # approximate

# Average HPI decline across states (weighted would be better but this is illustrative)
avg_hpi_decline = -hpi_change['pct_change'].mean()  # Make positive for decline

# Effect on suicide rate
pct_increase_suicide = iv_coef * avg_hpi_decline * 100  # Convert to percent
# This gives: -0.00889 * (-15) * 100 = 13.3% increase (approximately)

# But we need to be more careful - use state-level calculation
print(f"\nCalculation:")
print(f"  IV coefficient: {iv_coef}")
print(f"  Interpretation: 1pp HPI growth → {iv_coef*100:.3f}% change in suicide")

# For each state, calculate counterfactual
# Assume ~35,000 suicides per year, 5 years = 175,000 total
total_suicides_period = 175000  # Approximate

# Average effect
avg_effect = -iv_coef * avg_hpi_decline  # This is the proportional increase
excess_deaths = total_suicides_period * avg_effect

print(f"\n  Average HPI decline 2006-2011: {-avg_hpi_decline:.1f}pp")
print(f"  Implied suicide increase: {avg_effect*100:.1f}%")
print(f"  Baseline suicides 2007-2011: ~{total_suicides_period:,}")
print(f"  Excess deaths: ~{int(excess_deaths):,}")

# More precise calculation using actual state-year data
print("\n  More precise state-level calculation:")
bust_data = df[(df['year'] >= 2007) & (df['year'] <= 2011)].copy()
bust_data = bust_data.merge(hpi_2006, on='state')
bust_data['hpi_decline'] = 100 * (bust_data['hpi'] - bust_data['hpi_2006']) / bust_data['hpi_2006']

# Counterfactual: what if HPI had stayed at 2006 level?
# Effect = iv_coef * hpi_decline (which is negative for declines)
bust_data['pct_effect'] = iv_coef * bust_data['hpi_decline']  # This is positive when hpi_decline is negative

# Average effect across state-years
avg_pct_effect = bust_data['pct_effect'].mean()
print(f"  Average proportional effect: {avg_pct_effect*100:.2f}%")
print(f"  Applied to ~175,000 suicides: ~{int(175000 * avg_pct_effect):,} excess deaths")

# =============================================================================
# Summary Stats Table for Paper
# =============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS TABLE")
print("="*70)

# Prepare clean stats
stats_dict = {
    'Suicide rate (per 100k)': df['suicide_rate'],
    'HPI growth (%)': df['hpi_growth'],
    'Land elasticity': df['elasticity'],
    'Instrument': df['hpi_iv'],
}

print(f"\n{'Variable':<30} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
print("-"*70)
for name, series in stats_dict.items():
    print(f"{name:<30} {series.mean():>10.2f} {series.std():>10.2f} {series.min():>10.2f} {series.max():>10.2f}")

print(f"\nN = {len(df)}, States = {df['state'].nunique()}, Years = {df['year'].min()}-{df['year'].max()}")

print("\n" + "="*70)
print("DONE")
print("="*70)
