"""
Wellbeing Data Validation Script
================================
Produces sanity checks, coverage reports, and basic visualizations for the
state-year wellbeing dataset.

Author: Research pipeline
Created: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path('/Users/landieraugustin/project1')
WELLBEING_PATH = BASE_PATH / 'data' / 'processed' / 'wellbeing' / 'state_year_wellbeing.parquet'
OUTPUT_PATH = BASE_PATH / 'data' / 'processed' / 'wellbeing'

# =============================================================================
# Validation Functions
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load the wellbeing dataset."""
    if not WELLBEING_PATH.exists():
        raise FileNotFoundError(f"Wellbeing data not found at {WELLBEING_PATH}")
    return pd.read_parquet(WELLBEING_PATH)

def validate_ranges(df: pd.DataFrame) -> dict:
    """Check that variables are in expected ranges."""
    checks = {}

    # Life satisfaction mean should be in [1, 4]
    if 'brfss_lifesat_mean' in df.columns:
        ls = df['brfss_lifesat_mean'].dropna()
        checks['lifesat_mean_range'] = {
            'expected': '[1, 4]',
            'actual_min': ls.min(),
            'actual_max': ls.max(),
            'pass': (ls.min() >= 1) and (ls.max() <= 4)
        }

    # Life satisfaction shares should sum to ~1
    if 'brfss_lifesat_share_very_satisfied' in df.columns:
        share_cols = [c for c in df.columns if c.startswith('brfss_lifesat_share_')]
        if share_cols:
            share_sum = df[share_cols].sum(axis=1)
            share_sum_valid = share_sum[share_sum > 0]
            checks['lifesat_shares_sum'] = {
                'expected': '~1.0',
                'actual_mean': share_sum_valid.mean(),
                'actual_min': share_sum_valid.min(),
                'actual_max': share_sum_valid.max(),
                'pass': abs(share_sum_valid.mean() - 1.0) < 0.01
            }

    # Mental distress share should be in [0, 1]
    if 'brfss_freq_mental_distress_share' in df.columns:
        md = df['brfss_freq_mental_distress_share'].dropna()
        checks['mental_distress_range'] = {
            'expected': '[0, 1]',
            'actual_min': md.min(),
            'actual_max': md.max(),
            'pass': (md.min() >= 0) and (md.max() <= 1)
        }

    # Mental health days should be in [0, 30]
    if 'brfss_menthlth_mean_days' in df.columns:
        mh = df['brfss_menthlth_mean_days'].dropna()
        checks['menthlth_days_range'] = {
            'expected': '[0, 30]',
            'actual_min': mh.min(),
            'actual_max': mh.max(),
            'pass': (mh.min() >= 0) and (mh.max() <= 30)
        }

    # State FIPS should be valid US states
    valid_fips = set(range(1, 57)) | {66, 72, 78}  # US states + territories
    actual_fips = set(df['state_fips'].unique())
    checks['state_fips_valid'] = {
        'expected': 'Valid US FIPS codes',
        'n_states': len(actual_fips),
        'invalid_codes': actual_fips - valid_fips,
        'pass': actual_fips.issubset(valid_fips)
    }

    return checks

def compute_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute coverage summary by year."""
    years = sorted(df['year'].unique())

    coverage_data = []
    for year in years:
        year_df = df[df['year'] == year]
        row = {
            'year': year,
            'n_states': len(year_df),
            'total_respondents': year_df['n_unweighted'].sum()
        }

        # Variable-specific coverage
        for var in ['brfss_lifesat_mean', 'brfss_menthlth_mean_days',
                   'brfss_freq_mental_distress_share', 'brfss_genhlth_mean']:
            if var in year_df.columns:
                row[f'{var}_n_valid'] = year_df[var].notna().sum()

        coverage_data.append(row)

    return pd.DataFrame(coverage_data)

def compute_missingness_matrix(df: pd.DataFrame, var: str = 'brfss_lifesat_mean') -> pd.DataFrame:
    """Compute year x state missingness matrix for a variable."""
    if var not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index='state_usps',
        columns='year',
        values=var,
        aggfunc='first'
    )

    # Convert to 1 (present) / 0 (missing)
    missingness = pivot.notna().astype(int)
    return missingness

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for wellbeing variables."""
    vars_of_interest = [c for c in df.columns if c.startswith('brfss_')]

    stats = []
    for var in vars_of_interest:
        if var.endswith('_n'):
            continue
        series = df[var].dropna()
        if len(series) > 0:
            stats.append({
                'variable': var,
                'n_valid': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'p25': series.quantile(0.25),
                'median': series.median(),
                'p75': series.quantile(0.75),
                'max': series.max()
            })

    return pd.DataFrame(stats)

def compute_time_series_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute national time series averages."""
    vars_of_interest = ['brfss_lifesat_mean', 'brfss_menthlth_mean_days',
                        'brfss_freq_mental_distress_share', 'brfss_fair_poor_health_share']

    ts_data = []
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        row = {'year': year}

        for var in vars_of_interest:
            if var in year_df.columns:
                valid = year_df[var].dropna()
                if len(valid) > 0:
                    row[var] = valid.mean()

        ts_data.append(row)

    return pd.DataFrame(ts_data)

def compute_cross_state_distribution(df: pd.DataFrame, year: int = 2020,
                                     var: str = 'brfss_freq_mental_distress_share') -> pd.DataFrame:
    """Show distribution of a variable across states for a given year."""
    year_df = df[df['year'] == year]
    if var not in year_df.columns:
        return pd.DataFrame()

    result = year_df[['state_usps', var]].dropna()
    result = result.sort_values(var, ascending=False)
    return result

# =============================================================================
# Main Validation Report
# =============================================================================

def run_validation_report():
    """Run full validation and produce reports."""
    print("="*70)
    print("WELLBEING DATA VALIDATION REPORT")
    print("="*70)

    # Load data
    try:
        df = load_data()
        print(f"\nLoaded {len(df):,} state-year observations")
        print(f"Years: {df['year'].min()}-{df['year'].max()}")
        print(f"States: {df['state_fips'].nunique()}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # 1. Range validation
    print("\n" + "-"*70)
    print("1. RANGE VALIDATION")
    print("-"*70)
    checks = validate_ranges(df)
    for check_name, result in checks.items():
        status = "PASS" if result['pass'] else "FAIL"
        print(f"  {check_name}: {status}")
        if not result['pass']:
            print(f"    Details: {result}")

    # 2. Coverage summary
    print("\n" + "-"*70)
    print("2. COVERAGE BY YEAR")
    print("-"*70)
    coverage = compute_coverage_summary(df)
    print(coverage.to_string(index=False))

    # Save coverage
    coverage.to_csv(OUTPUT_PATH / 'coverage_by_year.csv', index=False)
    print(f"\nSaved: coverage_by_year.csv")

    # 3. Descriptive statistics
    print("\n" + "-"*70)
    print("3. DESCRIPTIVE STATISTICS")
    print("-"*70)
    stats = compute_descriptive_stats(df)
    # Format for display
    stats_display = stats.copy()
    for col in ['mean', 'std', 'min', 'p25', 'median', 'p75', 'max']:
        if col in stats_display.columns:
            stats_display[col] = stats_display[col].round(4)
    print(stats_display.to_string(index=False))

    # Save stats
    stats.to_csv(OUTPUT_PATH / 'descriptive_stats.csv', index=False)
    print(f"\nSaved: descriptive_stats.csv")

    # 4. Time series
    print("\n" + "-"*70)
    print("4. NATIONAL TIME SERIES (Cross-State Means)")
    print("-"*70)
    ts = compute_time_series_summary(df)
    ts_display = ts.copy()
    for col in ts_display.columns:
        if col != 'year':
            ts_display[col] = ts_display[col].round(4)
    print(ts_display.to_string(index=False))

    # Save time series
    ts.to_csv(OUTPUT_PATH / 'national_time_series.csv', index=False)
    print(f"\nSaved: national_time_series.csv")

    # 5. Missingness matrix for life satisfaction
    print("\n" + "-"*70)
    print("5. LIFE SATISFACTION MISSINGNESS MATRIX")
    print("-"*70)
    missingness = compute_missingness_matrix(df, 'brfss_lifesat_mean')
    if len(missingness) > 0:
        # Summary
        years_with_data = missingness.sum()
        states_with_data = missingness.sum(axis=1)
        print(f"States with any life satisfaction data: {(states_with_data > 0).sum()}")
        print(f"Years with any life satisfaction data: {(years_with_data > 0).sum()}")
        print(f"\nCoverage by year:")
        print(years_with_data.to_string())

        # Save full matrix
        missingness.to_csv(OUTPUT_PATH / 'lifesat_missingness_matrix.csv')
        print(f"\nSaved: lifesat_missingness_matrix.csv")
    else:
        print("Life satisfaction variable not available.")

    # 6. Cross-state distribution example
    print("\n" + "-"*70)
    print("6. CROSS-STATE DISTRIBUTION (Frequent Mental Distress, Recent Year)")
    print("-"*70)
    recent_year = df['year'].max()
    dist = compute_cross_state_distribution(df, recent_year, 'brfss_freq_mental_distress_share')
    if len(dist) > 0:
        dist_display = dist.copy()
        dist_display['brfss_freq_mental_distress_share'] = dist_display['brfss_freq_mental_distress_share'].round(4)
        print(f"Year: {recent_year}")
        print("Top 10 (highest mental distress):")
        print(dist_display.head(10).to_string(index=False))
        print("\nBottom 10 (lowest mental distress):")
        print(dist_display.tail(10).to_string(index=False))

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"Output files saved to: {OUTPUT_PATH}")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    run_validation_report()
