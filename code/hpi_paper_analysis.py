"""
Housing Wealth and Life Outcomes: Complete Analysis
====================================================
Generates all tables and figures for the paper.

Outcomes:
1. Suicide (log) - main finding
2. Mental Health Days - BRFSS
3. Frequent Mental Distress - BRFSS
4. Fair/Poor Health - BRFSS
5. Fertility (log) - null result for comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import os

BASE_PATH = '/Users/landieraugustin/project1'
os.makedirs(f'{BASE_PATH}/output/figures', exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8

# =============================================================================
# Load Data
# =============================================================================
print("="*70)
print("HOUSING WEALTH AND LIFE OUTCOMES: COMPLETE ANALYSIS")
print("="*70)

# HPI data
hpi = pd.read_csv(f'{BASE_PATH}/data/fhfa_state_hpi.csv',
                  names=['state', 'year', 'quarter', 'hpi'])
hpi_annual = hpi[hpi['quarter'] == 4][['state', 'year', 'hpi']].copy()
hpi_annual['hpi_growth'] = hpi_annual.groupby('state')['hpi'].pct_change() * 100
print(f"HPI: {len(hpi_annual)} obs")

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

# State FIPS to USPS mapping
fips_to_usps = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE',
    11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN',
    19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA',
    26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV',
    33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH',
    40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN',
    48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY'
}

# Wellbeing data
wellbeing = pd.read_parquet(f'{BASE_PATH}/data/processed/wellbeing/state_year_wellbeing.parquet')
wellbeing['state'] = wellbeing['state_fips'].map(fips_to_usps)
print(f"Wellbeing: {len(wellbeing)} obs, {wellbeing['year'].min()}-{wellbeing['year'].max()}")

# Suicide
suicide = pd.read_csv(f'{BASE_PATH}/data/suicide_state_panel_extended.csv')
suicide = suicide[['state', 'year', 'suicide_rate']]
print(f"Suicide: {len(suicide)} obs")

# Fertility
fertility = pd.read_csv(f'{BASE_PATH}/data/births_by_state.csv')
fertility = fertility.rename(columns={'birth_rate': 'fertility_rate'})
print(f"Fertility: {len(fertility)} obs")

# =============================================================================
# Build Panels
# =============================================================================
print("\n" + "-"*70)
print("BUILDING ANALYSIS PANELS")
print("-"*70)

# Full panel with all outcomes
df_full = hpi_annual.copy()
df_full = df_full.merge(suicide, on=['state', 'year'], how='left')
df_full = df_full.merge(fertility[['state', 'year', 'fertility_rate']], on=['state', 'year'], how='left')
df_full = df_full.merge(wellbeing[['state', 'year', 'brfss_menthlth_mean_days',
                                    'brfss_freq_mental_distress_share',
                                    'brfss_fair_poor_health_share']],
                        on=['state', 'year'], how='left')

# Add elasticity and IV
df_full['elasticity'] = df_full['state'].map(land_elasticity)
nat_hpi = df_full.groupby('year')['hpi_growth'].mean().reset_index()
nat_hpi.columns = ['year', 'national_hpi']
df_full = df_full.merge(nat_hpi, on='year')
df_full['hpi_iv'] = df_full['national_hpi'] * (1 / df_full['elasticity'])

# Log transforms
df_full['log_suicide'] = np.log(df_full['suicide_rate'])
df_full['log_fertility'] = np.log(df_full['fertility_rate'])

# Filter to main sample
df = df_full[(df_full['year'] >= 2000) & (df_full['year'] <= 2022)]
df = df.dropna(subset=['elasticity', 'hpi_growth', 'hpi_iv'])

print(f"Main panel (2000-2022): {len(df)} obs, {df['state'].nunique()} states")

# Wellbeing sample (2005-2022)
df_wb = df[df['year'] >= 2005].copy()
print(f"Wellbeing panel (2005-2022): {len(df_wb)} obs")

# =============================================================================
# IV Functions
# =============================================================================
def demean(x, groups):
    return x - x.groupby(groups).transform('mean')

def run_fe_regression(data, y_var, x_vars, cluster_var='state'):
    """OLS with state and year FE, clustered SEs."""
    df_reg = data.dropna(subset=[y_var] + x_vars).copy()
    if len(df_reg) < 50:
        return None, 0
    state_dummies = pd.get_dummies(df_reg['state'], prefix='s', drop_first=True)
    year_dummies = pd.get_dummies(df_reg['year'], prefix='y', drop_first=True)
    X = pd.concat([df_reg[x_vars], state_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(df_reg[y_var], X.astype(float)).fit(
        cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]})
    return model, len(df_reg)

def run_iv(data, y_var, x_var='hpi_growth', z_var='hpi_iv'):
    """IV with two-way FE via demeaning."""
    df_reg = data.dropna(subset=[y_var, x_var, z_var]).copy()
    if len(df_reg) < 50:
        return None

    df_reg['y_dm'] = demean(demean(df_reg[y_var], df_reg['state']), df_reg['year'])
    df_reg['x_dm'] = demean(demean(df_reg[x_var], df_reg['state']), df_reg['year'])
    df_reg['z_dm'] = demean(demean(df_reg[z_var], df_reg['state']), df_reg['year'])

    fs = sm.OLS(df_reg['x_dm'], sm.add_constant(df_reg['z_dm'])).fit()
    rf = sm.OLS(df_reg['y_dm'], sm.add_constant(df_reg['z_dm'])).fit()

    iv_coef = rf.params['z_dm'] / fs.params['z_dm']
    iv_se = rf.bse['z_dm'] / abs(fs.params['z_dm'])

    return {
        'fs_coef': fs.params['z_dm'],
        'fs_se': fs.bse['z_dm'],
        'fs_t': fs.tvalues['z_dm'],
        'fs_f': fs.fvalue,
        'rf_coef': rf.params['z_dm'],
        'rf_se': rf.bse['z_dm'],
        'rf_t': rf.tvalues['z_dm'],
        'iv_coef': iv_coef,
        'iv_se': iv_se,
        'n': len(df_reg),
        'n_states': df_reg['state'].nunique()
    }

# =============================================================================
# Main Results
# =============================================================================
print("\n" + "="*70)
print("MAIN RESULTS")
print("="*70)

outcomes = [
    ('Suicide', 'log_suicide', df, 'log', 'deaths'),
    ('Mental Health Days', 'brfss_menthlth_mean_days', df_wb, 'level', 'days'),
    ('Frequent Distress', 'brfss_freq_mental_distress_share', df_wb, 'share', 'share'),
    ('Fair/Poor Health', 'brfss_fair_poor_health_share', df_wb, 'share', 'share'),
    ('Fertility', 'log_fertility', df, 'log', 'rate'),
]

results = {}

print(f"\n{'Outcome':<20} {'OLS':>10} {'OLS t':>8} {'IV':>10} {'RF t':>8} {'FS F':>7} {'N':>6}")
print("-"*75)

for name, var, data, scale, unit in outcomes:
    model, n = run_fe_regression(data, var, ['hpi_growth'])
    iv = run_iv(data, var)

    if model is not None and iv is not None:
        results[name] = {
            'var': var,
            'scale': scale,
            'unit': unit,
            'ols_coef': model.params['hpi_growth'],
            'ols_se': model.bse['hpi_growth'],
            'ols_t': model.tvalues['hpi_growth'],
            'iv_coef': iv['iv_coef'],
            'iv_se': iv['iv_se'],
            'rf_coef': iv['rf_coef'],
            'rf_se': iv['rf_se'],
            'rf_t': iv['rf_t'],
            'fs_coef': iv['fs_coef'],
            'fs_f': iv['fs_f'],
            'n': iv['n']
        }

        ols_stars = '***' if abs(model.tvalues['hpi_growth']) > 2.58 else ('**' if abs(model.tvalues['hpi_growth']) > 1.96 else ('*' if abs(model.tvalues['hpi_growth']) > 1.64 else ''))
        rf_stars = '***' if abs(iv['rf_t']) > 2.58 else ('**' if abs(iv['rf_t']) > 1.96 else ('*' if abs(iv['rf_t']) > 1.64 else ''))

        print(f"{name:<20} {model.params['hpi_growth']:>9.5f}{ols_stars} {model.tvalues['hpi_growth']:>7.2f} {iv['iv_coef']:>9.5f}{rf_stars} {iv['rf_t']:>7.2f} {iv['fs_f']:>7.1f} {iv['n']:>6}")

# =============================================================================
# Figure 1: National Time Series
# =============================================================================
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

# Time series data
df_ts = df_full.groupby('year').agg({
    'hpi': 'mean',
    'suicide_rate': 'mean',
    'fertility_rate': 'mean',
    'brfss_menthlth_mean_days': 'mean',
    'brfss_freq_mental_distress_share': 'mean',
    'brfss_fair_poor_health_share': 'mean'
}).reset_index()

def add_gfc(ax):
    ax.axvspan(2007, 2009, alpha=0.12, color='gray')

# Row 1: HPI, Suicide, Fertility
ax = axes[0, 0]
ts = df_ts[(df_ts['year'] >= 2000) & (df_ts['year'] <= 2022)]
ax.plot(ts['year'], ts['hpi'], 'b-', lw=2.5, marker='o', ms=4)
add_gfc(ax)
ax.set_title('A. House Price Index', fontweight='bold', fontsize=11)
ax.set_ylabel('HPI Level')
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(1999, 2023)

ax = axes[0, 1]
ax.plot(ts['year'], ts['suicide_rate'], color='#B2182B', lw=2.5, marker='o', ms=4)
add_gfc(ax)
ax.set_title('B. Suicide Rate', fontweight='bold', fontsize=11)
ax.set_ylabel('Per 100,000')
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(1999, 2023)

ax = axes[0, 2]
ax.plot(ts['year'], ts['fertility_rate'], color='#2166AC', lw=2.5, marker='o', ms=4)
add_gfc(ax)
ax.set_title('C. Fertility Rate', fontweight='bold', fontsize=11)
ax.set_ylabel('Births per 1,000')
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(1999, 2023)

# Row 2: Mental Health Days, Frequent Distress, Fair/Poor Health
ts_wb = df_ts[(df_ts['year'] >= 2005) & (df_ts['year'] <= 2023)]

ax = axes[1, 0]
ax.plot(ts_wb['year'], ts_wb['brfss_menthlth_mean_days'], color='#D6604D', lw=2.5, marker='o', ms=4)
add_gfc(ax)
ax.set_title('D. Mental Health Days', fontweight='bold', fontsize=11)
ax.set_ylabel('Days (0-30)')
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(2004, 2024)

ax = axes[1, 1]
ax.plot(ts_wb['year'], ts_wb['brfss_freq_mental_distress_share'] * 100, color='#8073AC', lw=2.5, marker='o', ms=4)
add_gfc(ax)
ax.set_title('E. Frequent Mental Distress', fontweight='bold', fontsize=11)
ax.set_ylabel('Share (%)')
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(2004, 2024)

ax = axes[1, 2]
ax.plot(ts_wb['year'], ts_wb['brfss_fair_poor_health_share'] * 100, color='#E08214', lw=2.5, marker='o', ms=4)
add_gfc(ax)
ax.set_title('F. Fair/Poor Health', fontweight='bold', fontsize=11)
ax.set_ylabel('Share (%)')
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(2004, 2024)

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/output/figures/fig1_trends.pdf', bbox_inches='tight')
plt.savefig(f'{BASE_PATH}/output/figures/fig1_trends.png', dpi=150, bbox_inches='tight')
print("Saved: fig1_trends.pdf")

# =============================================================================
# Figure 2: OLS vs IV Comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Coefficients
ax = axes[0]
names = list(results.keys())
x = np.arange(len(names))
width = 0.35

ols_coefs = [results[n]['ols_coef'] for n in names]
iv_coefs = [results[n]['iv_coef'] for n in names]
ols_ses = [results[n]['ols_se'] for n in names]
iv_ses = [results[n]['iv_se'] for n in names]

bars1 = ax.bar(x - width/2, ols_coefs, width, label='OLS', color='#2166AC', alpha=0.85)
bars2 = ax.bar(x + width/2, iv_coefs, width, label='IV', color='#B2182B', alpha=0.85)

ax.errorbar(x - width/2, ols_coefs, yerr=[1.96*s for s in ols_ses], fmt='none', c='black', capsize=4, lw=1.5)
ax.errorbar(x + width/2, iv_coefs, yerr=[1.96*s for s in iv_ses], fmt='none', c='black', capsize=4, lw=1.5)

ax.axhline(0, color='gray', lw=1)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax.set_ylabel('Coefficient on HPI Growth')
ax.set_title('A. OLS vs IV Estimates', fontweight='bold')
ax.legend(loc='upper right')

# Panel B: Reduced Form t-statistics
ax = axes[1]
rf_ts = [results[n]['rf_t'] for n in names]
colors = ['#B2182B' if t < -1.96 else ('#2166AC' if t > 1.96 else '#888888') for t in rf_ts]

bars = ax.barh(range(len(names)), rf_ts, color=colors, alpha=0.85)
ax.axvline(-1.96, color='black', ls='--', lw=1.2, alpha=0.7)
ax.axvline(1.96, color='black', ls='--', lw=1.2, alpha=0.7)
ax.axvline(0, color='gray', lw=1)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Reduced Form t-statistic')
ax.set_title('B. Statistical Significance', fontweight='bold')

# Add labels
for i, t in enumerate(rf_ts):
    ax.text(t + 0.2 if t >= 0 else t - 0.2, i, f'{t:.2f}',
            va='center', ha='left' if t >= 0 else 'right', fontsize=9)

ax.text(-1.96, len(names)-0.3, 't = -1.96', ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/output/figures/fig2_ols_iv.pdf', bbox_inches='tight')
plt.savefig(f'{BASE_PATH}/output/figures/fig2_ols_iv.png', dpi=150, bbox_inches='tight')
print("Saved: fig2_ols_iv.pdf")

# =============================================================================
# Figure 3: Binscatter for key outcomes
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

def make_binscatter(ax, data, y_var, title, color, ylabel):
    df_plot = data.dropna(subset=[y_var, 'hpi_growth']).copy()
    df_plot['resid_y'] = demean(demean(df_plot[y_var], df_plot['state']), df_plot['year'])
    df_plot['resid_x'] = demean(demean(df_plot['hpi_growth'], df_plot['state']), df_plot['year'])

    df_plot['x_bin'] = pd.qcut(df_plot['resid_x'], 20, duplicates='drop')
    binned = df_plot.groupby('x_bin', observed=True).agg({
        'resid_x': 'mean', 'resid_y': 'mean'
    }).reset_index()

    ax.scatter(df_plot['resid_x'], df_plot['resid_y'], alpha=0.08, s=12, c='gray')
    ax.scatter(binned['resid_x'], binned['resid_y'], s=80, c=color,
               edgecolors='white', linewidths=1.5, zorder=5)

    slope, intercept, r, p, se = stats.linregress(df_plot['resid_x'], df_plot['resid_y'])
    x_line = np.linspace(df_plot['resid_x'].min(), df_plot['resid_x'].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, 'k-', lw=2)

    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Residualized HPI Growth (pp)')
    ax.set_ylabel(ylabel)

    stars = '***' if abs(slope/se) > 2.58 else ('**' if abs(slope/se) > 1.96 else ('*' if abs(slope/se) > 1.64 else ''))
    ax.set_title(f'{title}\nβ = {slope:.5f}{stars}', fontweight='bold')

    return slope, se

make_binscatter(axes[0,0], df, 'log_suicide', 'A. Suicide (log)', '#B2182B', 'Residualized log(Suicide Rate)')
make_binscatter(axes[0,1], df_wb, 'brfss_menthlth_mean_days', 'B. Mental Health Days', '#D6604D', 'Residualized Mental Health Days')
make_binscatter(axes[1,0], df_wb, 'brfss_freq_mental_distress_share', 'C. Frequent Mental Distress', '#8073AC', 'Residualized Distress Share')
make_binscatter(axes[1,1], df, 'log_fertility', 'D. Fertility (log)', '#2166AC', 'Residualized log(Fertility Rate)')

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/output/figures/fig3_binscatter.pdf', bbox_inches='tight')
plt.savefig(f'{BASE_PATH}/output/figures/fig3_binscatter.png', dpi=150, bbox_inches='tight')
print("Saved: fig3_binscatter.pdf")

# =============================================================================
# Figure 4: Pre-trends (Low vs High Elasticity)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Classify states by elasticity
low_elast_states = [s for s, e in land_elasticity.items() if e < 1.2]
high_elast_states = [s for s, e in land_elasticity.items() if e > 1.8]

df_elast = df.copy()
df_elast['elast_group'] = df_elast['state'].apply(
    lambda x: 'Low' if x in low_elast_states else ('High' if x in high_elast_states else 'Medium'))

# Panel A: HPI by elasticity group
ax = axes[0]
for group, color, label in [('Low', '#B2182B', 'Low Elasticity'), ('High', '#2166AC', 'High Elasticity')]:
    ts = df_elast[df_elast['elast_group'] == group].groupby('year')['hpi'].mean()
    base = ts.loc[2000] if 2000 in ts.index else ts.iloc[0]
    ts_idx = 100 * ts / base
    ax.plot(ts_idx.index, ts_idx.values, lw=2.5, color=color, label=label, marker='o', ms=4)

add_gfc(ax)
ax.axhline(100, color='gray', ls=':', lw=1)
ax.set_title('A. House Price Index by Supply Elasticity', fontweight='bold')
ax.set_ylabel('Index (2000 = 100)')
ax.legend()
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(1999, 2023)

# Panel B: Suicide by elasticity group
ax = axes[1]
for group, color, label in [('Low', '#B2182B', 'Low Elasticity'), ('High', '#2166AC', 'High Elasticity')]:
    ts = df_elast[df_elast['elast_group'] == group].groupby('year')['suicide_rate'].mean()
    base = ts.loc[2000] if 2000 in ts.index else ts.iloc[0]
    ts_idx = 100 * ts / base
    ax.plot(ts_idx.index, ts_idx.values, lw=2.5, color=color, label=label, marker='o', ms=4)

add_gfc(ax)
ax.axhline(100, color='gray', ls=':', lw=1)
ax.set_title('B. Suicide Rate by Supply Elasticity', fontweight='bold')
ax.set_ylabel('Index (2000 = 100)')
ax.legend()
ax.grid(True, alpha=0.3, ls=':')
ax.set_xlim(1999, 2023)

plt.tight_layout()
plt.savefig(f'{BASE_PATH}/output/figures/fig4_pretrends.pdf', bbox_inches='tight')
plt.savefig(f'{BASE_PATH}/output/figures/fig4_pretrends.png', dpi=150, bbox_inches='tight')
print("Saved: fig4_pretrends.pdf")

# =============================================================================
# Robustness: By Time Period
# =============================================================================
print("\n" + "="*70)
print("ROBUSTNESS: BY TIME PERIOD")
print("="*70)

periods = [
    ('Full (2000-2022)', 2000, 2022),
    ('Boom (2000-2006)', 2000, 2006),
    ('Bust (2007-2011)', 2007, 2011),
    ('Recovery (2012-2022)', 2012, 2022),
]

print(f"\n{'Period':<22} {'Suicide IV':>12} {'RF t':>8} {'Mental Dist IV':>14} {'RF t':>8}")
print("-"*70)

for name, y1, y2 in periods:
    df_p = df[(df['year'] >= y1) & (df['year'] <= y2)]
    df_wb_p = df_wb[(df_wb['year'] >= y1) & (df_wb['year'] <= y2)]

    iv_sui = run_iv(df_p, 'log_suicide')
    iv_md = run_iv(df_wb_p, 'brfss_freq_mental_distress_share')

    sui_str = f"{iv_sui['iv_coef']:.5f}" if iv_sui else "N/A"
    sui_t = f"{iv_sui['rf_t']:.2f}" if iv_sui else "--"
    md_str = f"{iv_md['iv_coef']:.5f}" if iv_md else "N/A"
    md_t = f"{iv_md['rf_t']:.2f}" if iv_md else "--"

    print(f"{name:<22} {sui_str:>12} {sui_t:>8} {md_str:>14} {md_t:>8}")

# =============================================================================
# Export Results Table
# =============================================================================
print("\n" + "="*70)
print("EXPORTING RESULTS")
print("="*70)

results_export = []
for name, r in results.items():
    results_export.append({
        'Outcome': name,
        'OLS_Coef': r['ols_coef'],
        'OLS_SE': r['ols_se'],
        'OLS_t': r['ols_t'],
        'IV_Coef': r['iv_coef'],
        'IV_SE': r['iv_se'],
        'RF_Coef': r['rf_coef'],
        'RF_SE': r['rf_se'],
        'RF_t': r['rf_t'],
        'FS_Coef': r['fs_coef'],
        'FS_F': r['fs_f'],
        'N': r['n']
    })

results_df = pd.DataFrame(results_export)
results_df.to_csv(f'{BASE_PATH}/output/hpi_paper_results.csv', index=False)
print("Saved: hpi_paper_results.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nEffect of 10pp HPI DECLINE (IV estimates):")
for name, r in results.items():
    effect = -10 * r['iv_coef']
    if r['scale'] == 'log':
        print(f"  {name}: {effect*100:+.1f}%")
    elif r['scale'] == 'share':
        print(f"  {name}: {effect*100:+.2f} pp")
    else:
        print(f"  {name}: {effect:+.3f} {r['unit']}")

print("\nStatistically significant IV effects:")
for name, r in results.items():
    if abs(r['rf_t']) > 1.96:
        print(f"  ✓ {name} (RF t = {r['rf_t']:.2f})")
    else:
        print(f"  ✗ {name} (RF t = {r['rf_t']:.2f}) - NOT SIGNIFICANT")

print("\nDone!")
