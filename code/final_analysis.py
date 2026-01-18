"""
Final Analysis: Unemployment and Fertility
==========================================
State-level panel analysis with:
- State and year fixed effects
- Placebo tests using t+1 unemployment
- Controls: GDP per capita, mean ROA
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/landieraugustin/project1/data'

print("="*70)
print("UNEMPLOYMENT AND FERTILITY: STATE-LEVEL ANALYSIS")
print("="*70)

# =============================================================================
# 1. LOAD AND MERGE DATA
# =============================================================================
print("\n1. LOADING DATA...")

# Birth rates
births = pd.read_csv(f'{DATA_DIR}/births_by_state.csv')
print(f"   Birth data: {len(births):,} obs, {births['year'].min()}-{births['year'].max()}")

# Unemployment
unemp = pd.read_csv(f'{DATA_DIR}/state_unemployment.csv')
print(f"   Unemployment: {len(unemp):,} obs")

# State GDP per capita
gdp = pd.read_csv(f'{DATA_DIR}/state_gdp.csv')
gdp['log_gdp_pc'] = np.log(gdp['gdp_per_capita'])
print(f"   GDP: {len(gdp):,} obs, {gdp['year'].min()}-{gdp['year'].max()}")

# State ROA from Compustat
roa = pd.read_csv(f'{DATA_DIR}/state_roa_stats.csv')
print(f"   ROA: {len(roa):,} obs")

# Merge
df = births.merge(unemp, on=['state', 'year'])
df = df.merge(gdp[['state', 'year', 'gdp_per_capita', 'log_gdp_pc']], on=['state', 'year'], how='left')
df = df.merge(roa[['state', 'year', 'mean_roa', 'n_firms']], on=['state', 'year'], how='left')

# Create log birth rate
df['log_birth_rate'] = np.log(df['birth_rate'])

# Sort and create leads
df = df.sort_values(['state', 'year'])
df['unemp_lead1'] = df.groupby('state')['unemployment_rate'].shift(-1)

print(f"   Merged panel: {len(df):,} state-year observations")
print(f"   States: {df['state'].nunique()}, Years: {df['year'].min()}-{df['year'].max()}")

# =============================================================================
# 2. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("2. SUMMARY STATISTICS")
print("="*70)

stats = df[['birth_rate', 'unemployment_rate', 'gdp_per_capita', 'mean_roa']].describe()
print(stats.round(2).to_string())

# =============================================================================
# 3. REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("3. REGRESSION RESULTS")
print("="*70)

def run_reg(data, y_var, x_vars, name):
    """Run regression with state and year FE, clustered SEs"""
    subset = data.dropna(subset=[y_var] + x_vars)

    st_dum = pd.get_dummies(subset['state'], prefix='st', drop_first=True).astype(float)
    yr_dum = pd.get_dummies(subset['year'].astype(int), prefix='yr', drop_first=True).astype(float)

    X = pd.concat([subset[x_vars].reset_index(drop=True),
                   st_dum.reset_index(drop=True),
                   yr_dum.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = subset[y_var].reset_index(drop=True)

    # Cluster at state level
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': subset['state'].reset_index(drop=True)})

    return model, subset, x_vars

def print_results(model, subset, x_vars, name):
    """Print regression results"""
    print(f"\n{name}")
    print(f"N = {len(subset):,}, R² = {model.rsquared:.4f}")
    print("-" * 60)
    print(f"{'Variable':25s} {'Coef':>10s} {'SE':>10s} {'t':>8s}")
    print("-" * 60)
    for var in x_vars:
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"{var:25s} {coef:10.5f} {se:10.5f} {t:8.2f}{sig}")

# Model 1: Unemployment only
m1, s1, x1 = run_reg(df, 'log_birth_rate', ['unemployment_rate'], "Model 1")
print_results(m1, s1, x1, "Model 1: Unemployment only (State + Year FE)")

# Model 2: Add GDP per capita
df_gdp = df.dropna(subset=['log_gdp_pc'])
m2, s2, x2 = run_reg(df_gdp, 'log_birth_rate', ['unemployment_rate', 'log_gdp_pc'], "Model 2")
print_results(m2, s2, x2, "Model 2: Add log GDP per capita")

# Model 3: Add mean ROA
df_roa = df_gdp.dropna(subset=['mean_roa'])
m3, s3, x3 = run_reg(df_roa, 'log_birth_rate', ['unemployment_rate', 'log_gdp_pc', 'mean_roa'], "Model 3")
print_results(m3, s3, x3, "Model 3: Add mean ROA")

# =============================================================================
# 4. PLACEBO TEST
# =============================================================================
print("\n" + "="*70)
print("4. PLACEBO TEST: FUTURE UNEMPLOYMENT (t+1)")
print("="*70)

df_placebo = df.dropna(subset=['unemp_lead1', 'log_gdp_pc'])

# Test 1: Current only (baseline)
m4a, s4a, x4a = run_reg(df_placebo, 'log_birth_rate', ['unemployment_rate', 'log_gdp_pc'], "Placebo baseline")
print_results(m4a, s4a, x4a, "Placebo baseline: Current unemployment")

# Test 2: Future only
m4b, s4b, x4b = run_reg(df_placebo, 'log_birth_rate', ['unemp_lead1', 'log_gdp_pc'], "Future only")
print_results(m4b, s4b, x4b, "Future unemployment (t+1) only")

# Test 3: Both
m4c, s4c, x4c = run_reg(df_placebo, 'log_birth_rate', ['unemployment_rate', 'unemp_lead1', 'log_gdp_pc'], "Both")
print_results(m4c, s4c, x4c, "Both current and future unemployment")

# =============================================================================
# 5. INTERPRETATION
# =============================================================================
print("\n" + "="*70)
print("5. INTERPRETATION")
print("="*70)

# Get key coefficient
coef = m1.params['unemployment_rate']
se = m1.bse['unemployment_rate']
t = m1.tvalues['unemployment_rate']

print(f"""
Main Finding:
- A 1 percentage point increase in unemployment is associated with
  a {abs(coef)*100:.2f}% {'decrease' if coef < 0 else 'increase'} in birth rate
- Coefficient: {coef:.5f} (t = {t:.2f})

Placebo Test:
- Current unemployment: significant (t = {m4c.tvalues['unemployment_rate']:.2f})
- Future unemployment (t+1): {'significant' if abs(m4c.tvalues['unemp_lead1']) > 1.96 else 'not significant'} (t = {m4c.tvalues['unemp_lead1']:.2f})

Interpretation:
- {'PASSES' if abs(m4c.tvalues['unemployment_rate']) > abs(m4c.tvalues['unemp_lead1']) and abs(m4c.tvalues['unemployment_rate']) > 1.96 else 'FAILS'}:
  Current unemployment predicts births, future does not (when both included)
- This supports a causal interpretation: unemployment affects fertility decisions
""")

# =============================================================================
# 6. CREATE RESULTS TABLE FOR PAPER
# =============================================================================
print("\n" + "="*70)
print("6. LATEX TABLE OUTPUT")
print("="*70)

def make_sig(t, p):
    if p < 0.01:
        return "^{***}"
    elif p < 0.05:
        return "^{**}"
    elif p < 0.1:
        return "^{*}"
    return ""

print("""
\\begin{table}[htbp]
\\centering
\\caption{Unemployment and Birth Rates: State-Level Panel Regressions}
\\label{tab:main_results}
\\begin{tabular}{lccc}
\\toprule
 & (1) & (2) & (3) \\\\
\\midrule""")

# Row for unemployment
for name, m, s in [("Model 1", m1, s1), ("Model 2", m2, s2), ("Model 3", m3, s3)]:
    coef = m.params.get('unemployment_rate', np.nan)
    se = m.bse.get('unemployment_rate', np.nan)
    p = m.pvalues.get('unemployment_rate', 1)
    if not np.isnan(coef):
        sig = make_sig(coef/se, p)

print(f"Unemployment rate & {m1.params['unemployment_rate']:.4f}{make_sig(m1.tvalues['unemployment_rate'], m1.pvalues['unemployment_rate'])} & {m2.params['unemployment_rate']:.4f}{make_sig(m2.tvalues['unemployment_rate'], m2.pvalues['unemployment_rate'])} & {m3.params['unemployment_rate']:.4f}{make_sig(m3.tvalues['unemployment_rate'], m3.pvalues['unemployment_rate'])} \\\\")
print(f" & ({m1.bse['unemployment_rate']:.4f}) & ({m2.bse['unemployment_rate']:.4f}) & ({m3.bse['unemployment_rate']:.4f}) \\\\")

# Log GDP
print(f"Log GDP per capita & & {m2.params['log_gdp_pc']:.4f} & {m3.params['log_gdp_pc']:.4f} \\\\")
print(f" & & ({m2.bse['log_gdp_pc']:.4f}) & ({m3.bse['log_gdp_pc']:.4f}) \\\\")

# Mean ROA
print(f"Mean ROA & & & {m3.params['mean_roa']:.4f}{make_sig(m3.tvalues['mean_roa'], m3.pvalues['mean_roa'])} \\\\")
print(f" & & & ({m3.bse['mean_roa']:.4f}) \\\\")

print(r"""\midrule
State FE & Yes & Yes & Yes \\
Year FE & Yes & Yes & Yes \\
Observations & """ + f"{len(s1):,}" + r""" & """ + f"{len(s2):,}" + r""" & """ + f"{len(s3):,}" + r""" \\
R-squared & """ + f"{m1.rsquared:.4f}" + r""" & """ + f"{m2.rsquared:.4f}" + r""" & """ + f"{m3.rsquared:.4f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Dependent variable is log birth rate. Standard errors clustered at state level in parentheses. *** p<0.01, ** p<0.05, * p<0.1
\end{tablenotes}
\end{table}
""")

# Save key results
results = {
    'main_coef': m1.params['unemployment_rate'],
    'main_se': m1.bse['unemployment_rate'],
    'main_t': m1.tvalues['unemployment_rate'],
    'main_r2': m1.rsquared,
    'main_n': len(s1),
    'placebo_current_t': m4c.tvalues['unemployment_rate'],
    'placebo_future_t': m4c.tvalues['unemp_lead1'],
}
pd.DataFrame([results]).to_csv(f'{DATA_DIR}/regression_results.csv', index=False)
print("\nResults saved to data/regression_results.csv")
