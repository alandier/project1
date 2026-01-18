"""
Pre-compute state-level ROA statistics from Compustat
"""

import pandas as pd
import numpy as np

print("Loading Compustat (only needed columns)...")
comp = pd.read_csv('/Users/landieraugustin/project1/data/compustat.csv',
                   usecols=['gvkey', 'fyear', 'state', 'at', 'ebitda', 'oibdp'])

print(f"Loaded {len(comp)} rows")

# Calculate ROA
comp['roa'] = comp['ebitda'] / comp['at']
# Fill missing EBITDA with operating income
mask = comp['roa'].isna()
comp.loc[mask, 'roa'] = comp.loc[mask, 'oibdp'] / comp.loc[mask, 'at']

# Filter valid observations
comp_valid = comp[(comp['at'] > 0) & (comp['roa'].notna()) & (comp['state'].notna())].copy()
comp_valid = comp_valid[(comp_valid['roa'] > -1) & (comp_valid['roa'] < 1)]

print(f"Valid observations: {len(comp_valid)}")

# Aggregate by state-year
print("Computing state-year ROA statistics...")
roa_stats = comp_valid.groupby(['state', 'fyear']).agg(
    mean_roa=('roa', 'mean'),
    median_roa=('roa', 'median'),
    roa_std=('roa', 'std'),
    roa_iqr=('roa', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    n_firms=('gvkey', 'nunique')
).reset_index()

roa_stats = roa_stats.rename(columns={'fyear': 'year'})

# Require at least 10 firms
roa_stats = roa_stats[roa_stats['n_firms'] >= 10]

print(f"\nFinal: {len(roa_stats)} state-year observations")
print(f"Years: {roa_stats['year'].min()} - {roa_stats['year'].max()}")
print(f"States: {roa_stats['state'].nunique()}")

# Save
roa_stats.to_csv('/Users/landieraugustin/project1/data/state_roa_stats.csv', index=False)
print("\nSaved to data/state_roa_stats.csv")

print("\nSummary:")
print(roa_stats[['mean_roa', 'roa_std', 'n_firms']].describe())
