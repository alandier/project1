"""
HPI and Life Outcomes: Suicide & Fertility
===========================================
State-level IV analysis 2000-2022.
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

# Fertility
fertility = pd.read_csv('/Users/landieraugustin/project1/data/births_by_state.csv')
fertility = fertility.rename(columns={'birth_rate': 'fertility_rate'})
print(f"Fertility: {len(fertility)} obs, {fertility['year'].min()}-{fertility['year'].max()}")

# Suicide
suicide = pd.read_csv('/Users/landieraugustin/project1/data/suicide_state_panel_extended.csv')
suicide = suicide[['state', 'year', 'suicide_rate']]
print(f"Suicide: {len(suicide)} obs, {suicide['year'].min()}-{suicide['year'].max()}")

# HPI
hpi = pd.read_csv('/Users/landieraugustin/project1/data/fhfa_state_hpi.csv',
                  names=['state', 'year', 'quarter', 'hpi'])
hpi_annual = hpi[hpi['quarter'] == 4][['state', 'year', 'hpi']].copy()
hpi_annual['hpi_growth'] = hpi_annual.groupby('state')['hpi'].pct_change() * 100
print(f"HPI: {len(hpi_annual)} obs, {hpi_annual['year'].min()}-{hpi_annual['year'].max()}")

# Land elasticity (Saiz 2010)
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
df = hpi_annual.merge(fertility[['state', 'year', 'fertility_rate']], on=['state', 'year'], how='left')
df = df.merge(suicide, on=['state', 'year'], how='left')

df['elasticity'] = df['state'].map(land_elasticity)
nat_hpi = df.groupby('year')['hpi_growth'].mean().reset_index()
nat_hpi.columns = ['year', 'national_hpi']
df = df.merge(nat_hpi, on='year')
df['hpi_iv'] = df['national_hpi'] * (1 / df['elasticity'])

df['log_fertility'] = np.log(df['fertility_rate'])
df['log_suicide'] = np.log(df['suicide_rate'])

df = df[(df['year'] >= 2000) & (df['year'] <= 2022)]
df = df.dropna(subset=['elasticity', 'hpi_growth', 'log_suicide', 'log_fertility'])

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
        'rf_coef': rf.params['z_dm'], 'rf_se': rf.bse['z_dm'], 'rf_t': rf.tvalues['z_dm'],
        'iv_coef': iv_coef, 'iv_se': iv_se, 'n': len(df_reg)
    }

# =============================================================================
# Main Results
# =============================================================================
print("\n" + "="*70)
print("MAIN RESULTS")
print("="*70)

results = {}

for outcome, log_var in [('Suicide', 'log_suicide'), ('Fertility', 'log_fertility')]:
    model, n = run_fe_regression(df, log_var, ['hpi_growth'])
    iv = run_iv(df, log_var, 'hpi_growth', 'hpi_iv')

    results[outcome] = {
        'ols_coef': model.params['hpi_growth'],
        'ols_se': model.bse['hpi_growth'],
        'ols_t': model.tvalues['hpi_growth'],
        'iv_coef': iv['iv_coef'],
        'iv_se': iv['iv_se'],
        'rf_t': iv['rf_t'],
        'fs_f': iv['fs_f'],
        'n': n
    }

    print(f"\n{outcome}:")
    print(f"  OLS: {model.params['hpi_growth']:.5f} (t={model.tvalues['hpi_growth']:.2f})")
    print(f"  IV:  {iv['iv_coef']:.5f} (RF t={iv['rf_t']:.2f})")
    print(f"  First Stage F: {iv['fs_f']:.1f}")

# =============================================================================
# Figure 1: Time Series
# =============================================================================
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))

df_ts = df.groupby('year').agg({
    'hpi': 'mean', 'suicide_rate': 'mean', 'fertility_rate': 'mean'
}).reset_index()

for col in ['hpi', 'suicide_rate', 'fertility_rate']:
    base = df_ts.loc[df_ts['year'] == 2000, col].values[0]
    df_ts[f'{col}_idx'] = 100 * df_ts[col] / base

def add_gfc(ax):
    ax.axvspan(2007, 2009, alpha=0.15, color='gray')

# HPI
ax = axes[0]
ax.plot(df_ts['year'], df_ts['hpi_idx'], 'b-', lw=2, marker='o', ms=4)
add_gfc(ax)
ax.set_title('House Price Index', fontweight='bold')
ax.set_ylabel('Index (2000 = 100)')
ax.grid(True, alpha=0.3)

# Suicide
ax = axes[1]
ax.plot(df_ts['year'], df_ts['suicide_rate_idx'], 'r-', lw=2, marker='o', ms=4)
add_gfc(ax)
ax.set_title('Suicide Rate', fontweight='bold')
ax.set_ylabel('Index (2000 = 100)')
ax.grid(True, alpha=0.3)

# Fertility
ax = axes[2]
ax.plot(df_ts['year'], df_ts['fertility_rate_idx'], 'g-', lw=2, marker='o', ms=4)
add_gfc(ax)
ax.set_title('Fertility Rate', fontweight='bold')
ax.set_ylabel('Index (2000 = 100)')
ax.grid(True, alpha=0.3)

plt.suptitle('National Trends, 2000-2022 (Indexed to 2000 = 100)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig1_time_series.pdf', bbox_inches='tight')
print("Saved: fig1_time_series.pdf")

# =============================================================================
# Figure 2: Binscatters
# =============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (log_var, title, color) in zip(axes, [('log_suicide', 'Suicide', '#B2182B'),
                                               ('log_fertility', 'Fertility', '#2166AC')]):
    df_plot = df.copy()
    df_plot['resid_y'] = demean(demean(df_plot[log_var], df_plot['state']), df_plot['year'])
    df_plot['resid_x'] = demean(demean(df_plot['hpi_growth'], df_plot['state']), df_plot['year'])

    df_plot['x_bin'] = pd.qcut(df_plot['resid_x'], 20, duplicates='drop')
    binned = df_plot.groupby('x_bin', observed=True).agg({
        'resid_x': 'mean', 'resid_y': 'mean'
    }).reset_index()

    ax.scatter(df_plot['resid_x'], df_plot['resid_y'], alpha=0.12, s=15, c='gray')
    ax.scatter(binned['resid_x'], binned['resid_y'], s=70, c=color,
               edgecolors='white', linewidths=1.5, zorder=5)

    slope, intercept, r, p, se = stats.linregress(df_plot['resid_x'], df_plot['resid_y'])
    x_line = np.linspace(df_plot['resid_x'].min(), df_plot['resid_x'].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'k-', lw=2)

    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Residualized HPI Growth (pp)')
    ax.set_ylabel(f'Residualized log({title} Rate)')

    stars = '***' if abs(slope/se) > 2.58 else ('**' if abs(slope/se) > 1.96 else ('*' if abs(slope/se) > 1.64 else ''))
    ax.set_title(f'{title}\nβ = {slope:.4f}{stars} (t = {slope/se:.2f})')

plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig2_binscatter.pdf', bbox_inches='tight')
print("Saved: fig2_binscatter.pdf")

# =============================================================================
# Figure 3: OLS vs IV
# =============================================================================
fig3, ax = plt.subplots(figsize=(8, 5))

outcomes = ['Suicide', 'Fertility']
x_pos = np.arange(len(outcomes))
width = 0.35

ols_coefs = [results[o]['ols_coef'] for o in outcomes]
iv_coefs = [results[o]['iv_coef'] for o in outcomes]
ols_ses = [results[o]['ols_se'] for o in outcomes]
iv_ses = [results[o]['iv_se'] for o in outcomes]

bars1 = ax.bar(x_pos - width/2, ols_coefs, width, label='OLS', color='#2166AC', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, iv_coefs, width, label='IV', color='#B2182B', alpha=0.8)

ax.errorbar(x_pos - width/2, ols_coefs, yerr=[1.96*s for s in ols_ses], fmt='none', c='black', capsize=5)
ax.errorbar(x_pos + width/2, iv_coefs, yerr=[1.96*s for s in iv_ses], fmt='none', c='black', capsize=5)

ax.axhline(0, color='gray', lw=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(outcomes)
ax.set_ylabel('Effect of 1pp HPI Growth on log(Outcome)')
ax.set_title('OLS vs IV Estimates')
ax.legend()

for i, (ols, iv) in enumerate(zip(ols_coefs, iv_coefs)):
    offset = 0.001
    ax.text(i - width/2, ols - offset if ols < 0 else ols + offset, f'{ols:.4f}',
            ha='center', va='top' if ols < 0 else 'bottom', fontsize=9)
    ax.text(i + width/2, iv - offset if iv < 0 else iv + offset, f'{iv:.4f}',
            ha='center', va='top' if iv < 0 else 'bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig3_ols_iv.pdf', bbox_inches='tight')
print("Saved: fig3_ols_iv.pdf")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n{'Outcome':<12} {'OLS':>10} {'OLS t':>8} {'IV':>10} {'RF t':>8} {'FS F':>8}")
print("-"*60)
for o in outcomes:
    r = results[o]
    print(f"{o:<12} {r['ols_coef']:>10.5f} {r['ols_t']:>8.2f} {r['iv_coef']:>10.5f} {r['rf_t']:>8.2f} {r['fs_f']:>8.1f}")

print("\nEffect of 10pp HPI decline:")
for o in outcomes:
    r = results[o]
    print(f"  {o}: OLS {-10*r['ols_coef']*100:+.2f}%, IV {-10*r['iv_coef']*100:+.2f}%")

print("\nDone!")
