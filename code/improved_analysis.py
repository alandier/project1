"""
Improved Analysis for Birth Rates Paper
=======================================
- State + Year FE only
- Show R²
- Add controls: GDP per capita, mean ROA, ROA dispersion
- Simplified placebo (t+1 only)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# =============================================================================
# 1. LOAD AND MERGE ALL DATA
# =============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)

# Birth data
births = pd.read_csv('/Users/landieraugustin/project1/data/births_by_state.csv')
print(f"Births: {len(births)} obs, years {births['year'].min()}-{births['year'].max()}")

# Unemployment
unemp = pd.read_csv('/Users/landieraugustin/project1/data/state_unemployment.csv')
print(f"Unemployment: {len(unemp)} obs, years {unemp['year'].min()}-{unemp['year'].max()}")

# State GDP
gdp = pd.read_csv('/Users/landieraugustin/project1/data/state_gdp.csv')
gdp['log_gdp_pc'] = np.log(gdp['gdp_per_capita'])
print(f"GDP: {len(gdp)} obs, years {gdp['year'].min()}-{gdp['year'].max()}")

# State-level ROA stats (pre-computed)
roa_stats = pd.read_csv('/Users/landieraugustin/project1/data/state_roa_stats.csv')
print(f"ROA stats: {len(roa_stats)} obs, years {roa_stats['year'].min()}-{roa_stats['year'].max()}")

# Merge all
df = births.merge(unemp, on=['state', 'year'], how='inner')
df = df.merge(gdp[['state', 'year', 'gdp_per_capita', 'log_gdp_pc']], on=['state', 'year'], how='left')
df = df.merge(roa_stats[['state', 'year', 'mean_roa', 'roa_std']], on=['state', 'year'], how='left')

df['log_birth_rate'] = np.log(df['birth_rate'])

print(f"\nMerged dataset: {len(df)} obs")
print(f"  With GDP: {df['gdp_per_capita'].notna().sum()}")
print(f"  With ROA: {df['mean_roa'].notna().sum()}")
print(f"  Years: {df['year'].min()}-{df['year'].max()}")

# =============================================================================
# 2. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

summary_vars = ['unemployment_rate', 'birth_rate', 'gdp_per_capita', 'mean_roa', 'roa_std']
summary = df[summary_vars].describe().T[['count', 'mean', 'std', 'min', 'max']]
print(summary.round(3))

# =============================================================================
# 3. MAIN REGRESSIONS WITH CONTROLS
# =============================================================================
print("\n" + "="*70)
print("MAIN REGRESSIONS (State + Year FE)")
print("="*70)

def run_panel_reg(data, dep_var, indep_vars, show_name=None):
    """Run panel regression with state + year FE"""
    required = [dep_var] + indep_vars
    subset = data.dropna(subset=required).copy()

    # Fixed effects
    st_dum = pd.get_dummies(subset['state'], prefix='st', drop_first=True).astype(float)
    yr_dum = pd.get_dummies(subset['year'].astype(int), prefix='yr', drop_first=True).astype(float)

    X = pd.concat([subset[indep_vars], st_dum, yr_dum], axis=1)
    X = sm.add_constant(X)
    y = subset[dep_var]

    model = sm.OLS(y, X).fit(cov_type='HC1')

    results = {
        'n': len(subset),
        'r2': model.rsquared,
        'r2_adj': model.rsquared_adj
    }
    for var in indep_vars:
        results[f'{var}_coef'] = model.params[var]
        results[f'{var}_se'] = model.bse[var]
        results[f'{var}_t'] = model.tvalues[var]
        results[f'{var}_p'] = model.pvalues[var]

    return results

# Run regressions with different control sets
print("\nDependent Variable: log(Birth Rate)")
print("-" * 90)

specs = [
    ('(1) Baseline', ['unemployment_rate']),
    ('(2) + GDP p.c.', ['unemployment_rate', 'log_gdp_pc']),
    ('(3) + Mean ROA', ['unemployment_rate', 'log_gdp_pc', 'mean_roa']),
    ('(4) + ROA Disp.', ['unemployment_rate', 'log_gdp_pc', 'mean_roa', 'roa_std']),
]

reg_results = []
for name, vars in specs:
    res = run_panel_reg(df, 'log_birth_rate', vars)
    res['spec'] = name
    reg_results.append(res)

# Display results
print(f"\n{'Specification':<20} {'Unemp Coef':>12} {'t-stat':>10} {'R²':>10} {'N':>8}")
print("-" * 70)
for res in reg_results:
    coef = res['unemployment_rate_coef']
    t = res['unemployment_rate_t']
    sig = "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""
    print(f"{res['spec']:<20} {coef:>12.5f} {t:>9.2f}{sig} {res['r2']:>10.3f} {res['n']:>8}")

# Full table
print("\n\nFull Coefficient Table:")
print("-" * 90)
header = f"{'Variable':<20}"
for res in reg_results:
    header += f" {res['spec']:>15}"
print(header)
print("-" * 90)

for var in ['unemployment_rate', 'log_gdp_pc', 'mean_roa', 'roa_std']:
    row = f"{var:<20}"
    for res in reg_results:
        if f'{var}_coef' in res:
            coef = res[f'{var}_coef']
            t = res[f'{var}_t']
            sig = "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""
            row += f" {coef:>13.4f}{sig:2s}"
        else:
            row += f" {'':>15}"
    print(row)

    # t-stats row
    row_t = f"{'':20}"
    for res in reg_results:
        if f'{var}_t' in res:
            row_t += f" ({res[f'{var}_t']:>11.2f})"
        else:
            row_t += f" {'':>15}"
    print(row_t)

print("-" * 90)
row = f"{'R²':<20}"
for res in reg_results:
    row += f" {res['r2']:>15.3f}"
print(row)

row = f"{'Observations':<20}"
for res in reg_results:
    row += f" {res['n']:>15,}"
print(row)

# =============================================================================
# 4. PLACEBO TEST (t+1 only)
# =============================================================================
print("\n" + "="*70)
print("PLACEBO TEST: Future Unemployment (t+1)")
print("="*70)

# Create lead
df_sorted = df.sort_values(['state', 'year'])
df_sorted['unemp_lead1'] = df_sorted.groupby('state')['unemployment_rate'].shift(-1)

print("\nTest: Does unemployment at t+1 predict birth rate at t?")
print("(If causal, the answer should be NO after controlling for current unemployment)")
print("-" * 70)

# Without control
subset1 = df_sorted.dropna(subset=['unemp_lead1', 'log_birth_rate'])
st_dum = pd.get_dummies(subset1['state'], prefix='st', drop_first=True).astype(float)
yr_dum = pd.get_dummies(subset1['year'].astype(int), prefix='yr', drop_first=True).astype(float)
X = pd.concat([subset1[['unemp_lead1']], st_dum, yr_dum], axis=1)
X = sm.add_constant(X)
model1 = sm.OLS(subset1['log_birth_rate'], X).fit(cov_type='HC1')

print(f"Without control for current unemp:")
print(f"  Unemp(t+1): coef = {model1.params['unemp_lead1']:.5f}, t = {model1.tvalues['unemp_lead1']:.2f}")

# With control for current unemployment
subset2 = df_sorted.dropna(subset=['unemp_lead1', 'unemployment_rate', 'log_birth_rate'])
st_dum = pd.get_dummies(subset2['state'], prefix='st', drop_first=True).astype(float)
yr_dum = pd.get_dummies(subset2['year'].astype(int), prefix='yr', drop_first=True).astype(float)
X = pd.concat([subset2[['unemp_lead1', 'unemployment_rate']], st_dum, yr_dum], axis=1)
X = sm.add_constant(X)
model2 = sm.OLS(subset2['log_birth_rate'], X).fit(cov_type='HC1')

print(f"\nControlling for current unemployment:")
print(f"  Unemp(t+1): coef = {model2.params['unemp_lead1']:.5f}, t = {model2.tvalues['unemp_lead1']:.2f}, p = {model2.pvalues['unemp_lead1']:.3f}")
print(f"  Unemp(t):   coef = {model2.params['unemployment_rate']:.5f}, t = {model2.tvalues['unemployment_rate']:.2f}")

placebo_pass = abs(model2.tvalues['unemp_lead1']) < 1.96
print(f"\n→ Placebo {'PASSES' if placebo_pass else 'FAILS'}: Future unemployment {'does NOT' if placebo_pass else 'DOES'} predict current births")

# Store for figure
placebo_results = {
    'lead1_raw': (model1.params['unemp_lead1'], model1.bse['unemp_lead1'], model1.tvalues['unemp_lead1']),
    'lead1_ctrl': (model2.params['unemp_lead1'], model2.bse['unemp_lead1'], model2.tvalues['unemp_lead1']),
    'current': (model2.params['unemployment_rate'], model2.bse['unemployment_rate'], model2.tvalues['unemployment_rate'])
}

# =============================================================================
# 5. LAGGED EFFECTS
# =============================================================================
print("\n" + "="*70)
print("LAGGED EFFECTS: Persistence of unemployment effect")
print("="*70)

# Create lags
for lag in range(1, 6):
    df_sorted[f'unemp_lag{lag}'] = df_sorted.groupby('state')['unemployment_rate'].shift(lag)

lag_results = []

# t=0
res0 = run_panel_reg(df_sorted, 'log_birth_rate', ['unemployment_rate'])
lag_results.append({'lag': 0, 'coef': res0['unemployment_rate_coef'],
                    'se': res0['unemployment_rate_se'], 't': res0['unemployment_rate_t'], 'n': res0['n']})

# t+1 to t+5 (birth rate responds to past unemployment)
for lag in range(1, 6):
    res = run_panel_reg(df_sorted, 'log_birth_rate', [f'unemp_lag{lag}'])
    lag_results.append({'lag': lag, 'coef': res[f'unemp_lag{lag}_coef'],
                        'se': res[f'unemp_lag{lag}_se'], 't': res[f'unemp_lag{lag}_t'], 'n': res['n']})

lag_df = pd.DataFrame(lag_results)

print("\nEffect of Unemployment at t on Birth Rate at t+k:")
print("-" * 60)
print(f"{'Horizon (k)':<12} {'Coefficient':>12} {'Std Err':>10} {'t-stat':>10} {'N':>8}")
print("-" * 60)
for _, row in lag_df.iterrows():
    sig = "***" if abs(row['t']) > 2.576 else "**" if abs(row['t']) > 1.96 else "*" if abs(row['t']) > 1.645 else ""
    print(f"t+{int(row['lag']):<10} {row['coef']:>12.5f} {row['se']:>10.5f} {row['t']:>9.2f}{sig} {int(row['n']):>8}")

# =============================================================================
# 6. FIGURES
# =============================================================================
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: National trends
ax1 = axes[0, 0]
national = df.groupby('year').agg({
    'birth_rate': 'mean',
    'unemployment_rate': 'mean'
}).reset_index()

ax1_twin = ax1.twinx()
l1, = ax1.plot(national['year'], national['birth_rate'], 'b-', linewidth=2, label='Birth Rate')
l2, = ax1_twin.plot(national['year'], national['unemployment_rate'], 'r--', linewidth=2, label='Unemployment')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Birth Rate (per 1,000)', color='blue', fontsize=12)
ax1_twin.set_ylabel('Unemployment Rate (%)', color='red', fontsize=12)
ax1.set_title('A. National Trends', fontsize=13, fontweight='bold')
ax1.legend([l1, l2], ['Birth Rate', 'Unemployment'], loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel B: Binned scatter (residualized)
ax2 = axes[0, 1]

# Residualize
subset_plot = df.dropna(subset=['unemployment_rate', 'log_birth_rate']).copy()
st_dum = pd.get_dummies(subset_plot['state'], prefix='st', drop_first=True).astype(float)
yr_dum = pd.get_dummies(subset_plot['year'].astype(int), prefix='yr', drop_first=True).astype(float)
fe = pd.concat([st_dum, yr_dum], axis=1)
fe = sm.add_constant(fe)

model_u = sm.OLS(subset_plot['unemployment_rate'], fe).fit()
subset_plot['unemp_resid'] = model_u.resid

model_b = sm.OLS(subset_plot['log_birth_rate'], fe).fit()
subset_plot['birth_resid'] = model_b.resid

# Create bins
subset_plot['unemp_bin'] = pd.qcut(subset_plot['unemp_resid'], 20, labels=False, duplicates='drop')
binned = subset_plot.groupby('unemp_bin').agg({
    'unemp_resid': 'mean',
    'birth_resid': 'mean'
}).reset_index()

ax2.scatter(binned['unemp_resid'], binned['birth_resid'], s=80, color='steelblue', edgecolor='black', zorder=5)

# Fit line
slope, intercept, _, _, _ = stats.linregress(subset_plot['unemp_resid'], subset_plot['birth_resid'])
x_line = np.linspace(binned['unemp_resid'].min(), binned['unemp_resid'].max(), 100)
ax2.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2, label=f'Slope: {slope:.4f}')

ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.axvline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('Unemployment (residualized)', fontsize=12)
ax2.set_ylabel('log(Birth Rate) (residualized)', fontsize=12)
ax2.set_title('B. Within State-Year Variation', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Panel C: Lagged effects
ax3 = axes[1, 0]
ax3.errorbar(lag_df['lag'], lag_df['coef'], yerr=1.96*lag_df['se'],
             fmt='o-', capsize=4, capthick=1.5, markersize=8, color='steelblue', linewidth=2)
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Years After Unemployment Shock (k)', fontsize=12)
ax3.set_ylabel('Effect on log(Birth Rate) at t+k', fontsize=12)
ax3.set_title('C. Dynamic Response to Unemployment', fontsize=13, fontweight='bold')
ax3.set_xticks(range(6))
ax3.set_xticklabels(['t', 't+1', 't+2', 't+3', 't+4', 't+5'])
ax3.grid(True, alpha=0.3)

# Panel D: Great Recession
ax4 = axes[1, 1]

# Compute changes 2007-2010
pre = df[df['year'] == 2007][['state', 'unemployment_rate', 'birth_rate']].rename(
    columns={'unemployment_rate': 'unemp_2007', 'birth_rate': 'birth_2007'})
post = df[df['year'] == 2010][['state', 'unemployment_rate', 'birth_rate']].rename(
    columns={'unemployment_rate': 'unemp_2010', 'birth_rate': 'birth_2010'})
gr = pre.merge(post, on='state')
gr['delta_unemp'] = gr['unemp_2010'] - gr['unemp_2007']
gr['delta_birth'] = (gr['birth_2010'] - gr['birth_2007']) / gr['birth_2007'] * 100

# Pretty scatter plot
ax4.scatter(gr['delta_unemp'], gr['delta_birth'], s=80, alpha=0.8,
            color='#2E86AB', edgecolor='white', linewidth=0.8, zorder=3)

# Fit line
slope_gr, intercept_gr, r_val, _, _ = stats.linregress(gr['delta_unemp'], gr['delta_birth'])
x_line = np.linspace(0, 10, 100)  # Clean x range
ax4.plot(x_line, intercept_gr + slope_gr * x_line, color='#E94F37', linewidth=2.5,
         label=f'Slope: {slope_gr:.2f}, R²={r_val**2:.2f}', zorder=2)

# Label key states (housing bust states and outliers)
labels_to_show = {'NV': 'Nevada', 'FL': 'Florida', 'AZ': 'Arizona', 'CA': 'California',
                  'MI': 'Michigan', 'ND': 'N. Dakota', 'UT': 'Utah'}
for _, row in gr.iterrows():
    if row['state'] in labels_to_show:
        # Offset labels to avoid overlap
        offset_x, offset_y = 0.15, 0.3
        if row['state'] == 'CA':
            offset_x, offset_y = -0.8, 0.4
        elif row['state'] == 'ND':
            offset_x, offset_y = 0.15, -0.5
        ax4.annotate(row['state'], (row['delta_unemp'] + offset_x, row['delta_birth'] + offset_y),
                    fontsize=9, fontweight='bold', color='#333333')

# Clean axis limits
ax4.set_xlim(0, 10)
ax4.set_ylim(-16, -4)
ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax4.set_xlabel('Change in Unemployment (pp), 2007–2010', fontsize=12)
ax4.set_ylabel('Change in Birth Rate (%), 2007–2010', fontsize=12)
ax4.set_title('D. Great Recession: Cross-State Variation', fontsize=13, fontweight='bold')
ax4.legend(loc='lower left', framealpha=0.9)
ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('/Users/landieraugustin/project1/output/figures/fig_main_results.png', dpi=150, bbox_inches='tight')
print("Saved: output/figures/fig_main_results.png")

# =============================================================================
# 7. LATEX TABLES
# =============================================================================
print("\n" + "="*70)
print("LATEX TABLE: Main Results")
print("="*70)

latex_table = r"""
\begin{table}[H]
\centering
\caption{Unemployment and Birth Rates}
\label{tab:main}
\begin{tabular}{lcccc}
\toprule
 & (1) & (2) & (3) & (4) \\
 & Baseline & + GDP p.c. & + Mean ROA & + ROA Disp. \\
\midrule
"""

for var, label in [('unemployment_rate', 'Unemployment'),
                   ('log_gdp_pc', 'log(GDP per capita)'),
                   ('mean_roa', 'Mean ROA'),
                   ('roa_std', 'ROA Dispersion')]:
    row = f"{label} "
    for res in reg_results:
        if f'{var}_coef' in res:
            coef = res[f'{var}_coef']
            t = res[f'{var}_t']
            sig = "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""
            row += f"& {coef:.4f}{sig} "
        else:
            row += "& "
    row += r"\\"
    latex_table += row + "\n"

    # t-stat row
    row_t = " "
    for res in reg_results:
        if f'{var}_t' in res:
            row_t += f"& ({res[f'{var}_t']:.2f}) "
        else:
            row_t += "& "
    row_t += r"\\"
    latex_table += row_t + "\n"

latex_table += r"""\midrule
"""

# R² and N
row_r2 = "R-squared "
for res in reg_results:
    row_r2 += f"& {res['r2']:.3f} "
row_r2 += r"\\"
latex_table += row_r2 + "\n"

row_n = "Observations "
for res in reg_results:
    row_n += f"& {res['n']:,} "
row_n += r"\\"
latex_table += row_n + "\n"

latex_table += r"""State FE & Yes & Yes & Yes & Yes \\
Year FE & Yes & Yes & Yes & Yes \\
\bottomrule
\multicolumn{5}{l}{\footnotesize $t$-statistics in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.}
\end{tabular}
\end{table}
"""

print(latex_table)

plt.show()
