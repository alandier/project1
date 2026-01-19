"""
HPI and Suicide: State-level IV Analysis
=========================================
Extended analysis 2000-2022 with land elasticity IV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('/Users/landieraugustin/project1/output/figures', exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# =============================================================================
# Load Data
# =============================================================================
print("="*70)
print("LOADING DATA (2000-2022)")
print("="*70)

suicide = pd.read_csv('/Users/landieraugustin/project1/data/suicide_state_panel_extended.csv')
suicide = suicide[['state', 'year', 'suicide_rate']]
print(f"Suicide: {len(suicide)} obs, {suicide['year'].min()}-{suicide['year'].max()}")

hpi = pd.read_csv('/Users/landieraugustin/project1/data/fhfa_state_hpi.csv',
                  names=['state', 'year', 'quarter', 'hpi'])
hpi_annual = hpi[hpi['quarter'] == 4][['state', 'year', 'hpi']].copy()
hpi_annual['hpi_growth'] = hpi_annual.groupby('state')['hpi'].pct_change() * 100
print(f"HPI: {len(hpi_annual)} obs, {hpi_annual['year'].min()}-{hpi_annual['year'].max()}")

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

df = hpi_annual.merge(suicide, on=['state', 'year'], how='left')
df['elasticity'] = df['state'].map(land_elasticity)
nat_hpi = df.groupby('year')['hpi_growth'].mean().reset_index()
nat_hpi.columns = ['year', 'national_hpi']
df = df.merge(nat_hpi, on='year')
df['hpi_iv'] = df['national_hpi'] * (1 / df['elasticity'])
df['log_suicide'] = np.log(df['suicide_rate'])

df = df[(df['year'] >= 2000) & (df['year'] <= 2022)]
df = df.dropna(subset=['elasticity', 'hpi_growth', 'log_suicide'])

print(f"\nPanel: {len(df)} obs, {df['year'].min()}-{df['year'].max()}, {df['state'].nunique()} states")

# =============================================================================
# Functions
# =============================================================================
def demean(x, groups):
    return x - x.groupby(groups).transform('mean')

def run_fe_regression(df, y_var, x_vars, cluster_var='state'):
    df_reg = df.dropna(subset=[y_var] + x_vars).copy()
    state_dummies = pd.get_dummies(df_reg['state'], prefix='s', drop_first=True)
    year_dummies = pd.get_dummies(df_reg['year'], prefix='y', drop_first=True)
    X = pd.concat([df_reg[x_vars], state_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(df_reg[y_var], X.astype(float)).fit(
        cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]})
    return model, len(df_reg)

def run_iv(df, y_var, x_var, z_var):
    df_reg = df.dropna(subset=[y_var, x_var, z_var]).copy()
    df_reg['y_dm'] = demean(demean(df_reg[y_var], df_reg['state']), df_reg['year'])
    df_reg['x_dm'] = demean(demean(df_reg[x_var], df_reg['state']), df_reg['year'])
    df_reg['z_dm'] = demean(demean(df_reg[z_var], df_reg['state']), df_reg['year'])

    fs = sm.OLS(df_reg['x_dm'], sm.add_constant(df_reg['z_dm'])).fit()
    rf = sm.OLS(df_reg['y_dm'], sm.add_constant(df_reg['z_dm'])).fit()

    iv_coef = rf.params['z_dm'] / fs.params['z_dm']
    iv_se = rf.bse['z_dm'] / abs(fs.params['z_dm'])

    return {
        'fs_coef': fs.params['z_dm'], 'fs_f': fs.fvalue,
        'rf_coef': rf.params['z_dm'], 'rf_t': rf.tvalues['z_dm'],
        'iv_coef': iv_coef, 'iv_se': iv_se, 'n': len(df_reg)
    }

# =============================================================================
# Main Results
# =============================================================================
print("\n" + "="*70)
print("MAIN RESULTS: HPI → Suicide")
print("="*70)

# OLS
model, n = run_fe_regression(df, 'log_suicide', ['hpi_growth'])
ols_coef = model.params['hpi_growth']
ols_se = model.bse['hpi_growth']
ols_t = model.tvalues['hpi_growth']
print(f"\nOLS: {ols_coef:.5f} (SE={ols_se:.5f}, t={ols_t:.2f}), N={n}")

# IV
iv = run_iv(df, 'log_suicide', 'hpi_growth', 'hpi_iv')
print(f"\nIV:")
print(f"  First Stage F: {iv['fs_f']:.1f}")
print(f"  Reduced Form: {iv['rf_coef']:.5f} (t={iv['rf_t']:.2f})")
print(f"  IV estimate: {iv['iv_coef']:.5f} (SE={iv['iv_se']:.5f})")

print(f"\nInterpretation:")
print(f"  OLS: 10pp HPI decline → {-10*ols_coef*100:+.1f}% suicide")
print(f"  IV:  10pp HPI decline → {-10*iv['iv_coef']*100:+.1f}% suicide")

# =============================================================================
# Figure 1: Time Series
# =============================================================================
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

df_ts = df.groupby('year').agg({'hpi': 'mean', 'suicide_rate': 'mean'}).reset_index()
for col in ['hpi', 'suicide_rate']:
    base = df_ts.loc[df_ts['year'] == 2000, col].values[0]
    df_ts[f'{col}_idx'] = 100 * df_ts[col] / base

# HPI
ax = axes[0]
ax.plot(df_ts['year'], df_ts['hpi_idx'], 'b-', lw=2, marker='o', ms=5)
ax.axvspan(2007, 2009, alpha=0.15, color='gray', label='GFC')
ax.set_title('House Price Index', fontweight='bold', fontsize=12)
ax.set_ylabel('Index (2000 = 100)')
ax.set_xlabel('Year')
ax.grid(True, alpha=0.3)

# Suicide
ax = axes[1]
ax.plot(df_ts['year'], df_ts['suicide_rate_idx'], 'r-', lw=2, marker='o', ms=5)
ax.axvspan(2007, 2009, alpha=0.15, color='gray', label='GFC')
ax.set_title('Suicide Rate', fontweight='bold', fontsize=12)
ax.set_ylabel('Index (2000 = 100)')
ax.set_xlabel('Year')
ax.grid(True, alpha=0.3)

plt.suptitle('National Trends, 2000-2022', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig1_time_series.pdf', bbox_inches='tight')
print("Saved: fig1_time_series.pdf")

# =============================================================================
# Figure 2: Binscatter
# =============================================================================
fig2, ax = plt.subplots(figsize=(8, 6))

df_plot = df.copy()
df_plot['resid_y'] = demean(demean(df_plot['log_suicide'], df_plot['state']), df_plot['year'])
df_plot['resid_x'] = demean(demean(df_plot['hpi_growth'], df_plot['state']), df_plot['year'])

df_plot['x_bin'] = pd.qcut(df_plot['resid_x'], 20, duplicates='drop')
binned = df_plot.groupby('x_bin', observed=True).agg({
    'resid_x': 'mean', 'resid_y': 'mean'
}).reset_index()

ax.scatter(df_plot['resid_x'], df_plot['resid_y'], alpha=0.15, s=20, c='gray')
ax.scatter(binned['resid_x'], binned['resid_y'], s=80, c='#B2182B',
           edgecolors='white', linewidths=1.5, zorder=5)

slope, intercept, r, p, se = stats.linregress(df_plot['resid_x'], df_plot['resid_y'])
x_line = np.linspace(df_plot['resid_x'].min(), df_plot['resid_x'].max(), 100)
ax.plot(x_line, intercept + slope * x_line, 'k-', lw=2.5)

ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.axvline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel('Residualized HPI Growth (pp)', fontsize=11)
ax.set_ylabel('Residualized log(Suicide Rate)', fontsize=11)
ax.set_title(f'HPI Growth and Suicide (OLS)\nβ = {slope:.4f} (t = {slope/se:.2f})', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig2_binscatter.pdf', bbox_inches='tight')
print("Saved: fig2_binscatter.pdf")

# =============================================================================
# Figure 3: OLS vs IV
# =============================================================================
fig3, ax = plt.subplots(figsize=(6, 5))

x_pos = [0, 1]
coefs = [ols_coef, iv['iv_coef']]
ses = [ols_se, iv['iv_se']]
colors = ['#2166AC', '#B2182B']

bars = ax.bar(x_pos, coefs, width=0.6, color=colors, alpha=0.8)
ax.errorbar(x_pos, coefs, yerr=[1.96*s for s in ses], fmt='none', c='black', capsize=6, lw=2)

ax.axhline(0, color='gray', lw=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(['OLS', 'IV'], fontsize=11)
ax.set_ylabel('Effect on log(Suicide Rate)', fontsize=11)
ax.set_title('Effect of 1pp HPI Growth on Suicide', fontsize=12)

for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.text(i, c - 0.002, f'{c:.4f}', ha='center', va='top', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig3_ols_iv.pdf', bbox_inches='tight')
print("Saved: fig3_ols_iv.pdf")

# =============================================================================
# Robustness
# =============================================================================
print("\n" + "="*70)
print("ROBUSTNESS CHECKS")
print("="*70)

# By period
periods = [
    ('2000-2022', 2000, 2022),
    ('2000-2010', 2000, 2010),
    ('2008-2012 (GFC)', 2008, 2012),
    ('2011-2022', 2011, 2022),
]

print(f"\n{'Period':<20} {'OLS coef':>10} {'OLS t':>8} {'IV coef':>10} {'RF t':>8}")
print("-"*60)
for name, y1, y2 in periods:
    df_p = df[(df['year'] >= y1) & (df['year'] <= y2)]
    if len(df_p) > 100:
        m, _ = run_fe_regression(df_p, 'log_suicide', ['hpi_growth'])
        iv_p = run_iv(df_p, 'log_suicide', 'hpi_growth', 'hpi_iv')
        print(f"{name:<20} {m.params['hpi_growth']:>10.5f} {m.tvalues['hpi_growth']:>8.2f} "
              f"{iv_p['iv_coef']:>10.5f} {iv_p['rf_t']:>8.2f}")

# Excluding states
print(f"\n{'Sample':<25} {'IV coef':>10} {'RF t':>8} {'N':>8}")
print("-"*55)

samples = [
    ('Full sample', df),
    ('Excl. DC', df[df['state'] != 'DC']),
    ('Excl. CA', df[df['state'] != 'CA']),
    ('Excl. AK, WY, MT', df[~df['state'].isin(['AK', 'WY', 'MT'])]),
]

for name, sample in samples:
    iv_s = run_iv(sample, 'log_suicide', 'hpi_growth', 'hpi_iv')
    print(f"{name:<25} {iv_s['iv_coef']:>10.5f} {iv_s['rf_t']:>8.2f} {iv_s['n']:>8}")

print("\nDone!")
