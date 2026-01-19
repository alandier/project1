"""
Wellbeing Data Merge Utility
============================
Utilities for merging state-year wellbeing data into the main analysis panel.

Author: Research pipeline
Created: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path('/Users/landieraugustin/project1')
WELLBEING_PATH = BASE_PATH / 'data' / 'processed' / 'wellbeing' / 'state_year_wellbeing.parquet'

# State FIPS to USPS mapping (for convenience)
STATE_FIPS_TO_USPS = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE',
    11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN',
    19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA',
    26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV',
    33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH',
    40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN',
    48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI',
    56: 'WY', 66: 'GU', 72: 'PR', 78: 'VI'
}

STATE_USPS_TO_FIPS = {v: k for k, v in STATE_FIPS_TO_USPS.items()}

# =============================================================================
# Loading Functions
# =============================================================================

def load_wellbeing_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the state-year wellbeing dataset.

    Args:
        path: Path to wellbeing parquet file (default: standard location)

    Returns:
        DataFrame with wellbeing variables
    """
    path = path or WELLBEING_PATH
    if not path.exists():
        raise FileNotFoundError(f"Wellbeing data not found at {path}. "
                               "Run ingest_brfss_wellbeing.py first.")
    return pd.read_parquet(path)

# =============================================================================
# Merge Functions
# =============================================================================

def merge_wellbeing(
    panel: pd.DataFrame,
    state_col: str = 'state',
    year_col: str = 'year',
    state_type: str = 'auto',
    wellbeing_vars: Optional[list] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge wellbeing data into a state-year panel.

    Args:
        panel: Main panel DataFrame with state and year columns
        state_col: Name of state column in panel
        year_col: Name of year column in panel
        state_type: Type of state identifier ('fips', 'usps', or 'auto')
        wellbeing_vars: List of wellbeing variables to include (None = all)
        verbose: Print merge diagnostics

    Returns:
        Tuple of (merged DataFrame, merge diagnostics dict)
    """
    # Load wellbeing data
    wellbeing = load_wellbeing_data()

    # Determine state type
    if state_type == 'auto':
        sample_val = panel[state_col].dropna().iloc[0] if len(panel) > 0 else None
        if isinstance(sample_val, str):
            state_type = 'usps'
        else:
            state_type = 'fips'

    # Prepare merge keys
    panel_copy = panel.copy()

    if state_type == 'usps':
        # Map USPS to FIPS for merge
        panel_copy['_merge_state_fips'] = panel_copy[state_col].map(STATE_USPS_TO_FIPS)
    else:
        panel_copy['_merge_state_fips'] = panel_copy[state_col].astype(int)

    panel_copy['_merge_year'] = panel_copy[year_col].astype(int)

    # Select wellbeing variables
    if wellbeing_vars:
        keep_cols = ['state_fips', 'year'] + [c for c in wellbeing_vars if c in wellbeing.columns]
    else:
        keep_cols = wellbeing.columns.tolist()

    wellbeing_subset = wellbeing[keep_cols].copy()

    # Pre-merge stats
    n_panel = len(panel_copy)
    n_wellbeing = len(wellbeing_subset)
    panel_years = set(panel_copy['_merge_year'].unique())
    wellbeing_years = set(wellbeing_subset['year'].unique())
    year_overlap = panel_years & wellbeing_years

    # Merge
    merged = panel_copy.merge(
        wellbeing_subset,
        left_on=['_merge_state_fips', '_merge_year'],
        right_on=['state_fips', 'year'],
        how='left',
        suffixes=('', '_wellbeing')
    )

    # Clean up merge columns
    merge_cols = ['_merge_state_fips', '_merge_year']
    if 'state_fips_wellbeing' in merged.columns:
        merge_cols.append('state_fips_wellbeing')
    if 'year_wellbeing' in merged.columns:
        merge_cols.append('year_wellbeing')
    merged = merged.drop(columns=merge_cols, errors='ignore')

    # Diagnostics
    wb_vars = [c for c in wellbeing_subset.columns if c.startswith('brfss_')]
    n_matched = merged[wb_vars[0]].notna().sum() if wb_vars else 0
    match_rate = n_matched / n_panel if n_panel > 0 else 0

    diagnostics = {
        'n_panel_rows': n_panel,
        'n_wellbeing_rows': n_wellbeing,
        'n_matched': n_matched,
        'match_rate': match_rate,
        'panel_years': sorted(panel_years),
        'wellbeing_years': sorted(wellbeing_years),
        'year_overlap': sorted(year_overlap),
        'wellbeing_vars_merged': wb_vars
    }

    if verbose:
        print("="*60)
        print("WELLBEING DATA MERGE DIAGNOSTICS")
        print("="*60)
        print(f"Panel rows: {n_panel:,}")
        print(f"Wellbeing state-years: {n_wellbeing}")
        print(f"Matched rows: {n_matched:,} ({100*match_rate:.1f}%)")
        print(f"Panel years: {min(panel_years)}-{max(panel_years)}" if panel_years else "N/A")
        print(f"Wellbeing years: {min(wellbeing_years)}-{max(wellbeing_years)}" if wellbeing_years else "N/A")
        print(f"Year overlap: {len(year_overlap)} years")
        print(f"Variables merged: {len(wb_vars)}")

        # Missing by year
        if year_overlap:
            print("\nMissingness by year (for matched years):")
            for year in sorted(year_overlap)[:5]:
                year_data = merged[merged[year_col] == year]
                if len(year_data) > 0 and wb_vars:
                    missing_pct = 100 * year_data[wb_vars[0]].isna().mean()
                    print(f"  {year}: {missing_pct:.1f}% missing")

    return merged, diagnostics

# =============================================================================
# Utility Functions
# =============================================================================

def get_wellbeing_coverage() -> pd.DataFrame:
    """Get summary of wellbeing data coverage by year."""
    wellbeing = load_wellbeing_data()

    summary = wellbeing.groupby('year').agg({
        'state_fips': 'nunique',
        'n_unweighted': 'sum'
    }).reset_index()
    summary.columns = ['year', 'n_states', 'total_respondents']

    # Add variable coverage
    for var in ['brfss_lifesat_mean', 'brfss_menthlth_mean_days', 'brfss_freq_mental_distress_share']:
        if var in wellbeing.columns:
            coverage = wellbeing.groupby('year')[var].apply(lambda x: x.notna().sum())
            summary[f'{var}_n_states'] = summary['year'].map(coverage)

    return summary

def create_state_mapping_table() -> pd.DataFrame:
    """Create a reference table mapping state FIPS to USPS codes."""
    return pd.DataFrame([
        {'state_fips': fips, 'state_usps': usps}
        for fips, usps in sorted(STATE_FIPS_TO_USPS.items())
    ])

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    # Example: merge with a simple panel
    print("Creating example panel...")

    # Create dummy panel
    years = range(2005, 2024)
    states = [1, 6, 12, 36, 48]  # AL, CA, FL, NY, TX
    panel = pd.DataFrame([
        {'state': s, 'year': y, 'dummy_outcome': np.random.randn()}
        for y in years for s in states
    ])

    print(f"Example panel: {len(panel)} rows")

    # Merge wellbeing
    try:
        merged, diag = merge_wellbeing(panel, state_col='state', year_col='year')
        print(f"\nMerged panel: {len(merged)} rows")
        print(f"Sample columns: {merged.columns.tolist()[:10]}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Run the BRFSS ingestion first to create wellbeing data.")

    # Show coverage
    print("\n" + "="*60)
    print("WELLBEING DATA COVERAGE")
    print("="*60)
    try:
        coverage = get_wellbeing_coverage()
        print(coverage.to_string(index=False))
    except FileNotFoundError:
        print("Wellbeing data not yet available.")
