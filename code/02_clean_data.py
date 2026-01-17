"""
02_clean_data.py
Clean and prepare Compustat data for leverage analysis

Author: Research collaboration
Project: Determinants of Firm Leverage in the US (1985-present)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
INPUT_FILE = "../data/raw/compustat_annual.csv"  # Adjust filename as needed
OUTPUT_FILE = "../data/processed/leverage_panel.csv"
START_YEAR = 1985
END_YEAR = 2024

def load_data(filepath):
      """Load Compustat data from CSV or Stata file"""
      if filepath.endswith('.dta'):
                df = pd.read_stata(filepath)
else:
        df = pd.read_csv(filepath, low_memory=False)
      print(f"Loaded {len(df):,} observations")
    return df

def filter_us_firms(df):
      """Keep only US-incorporated firms"""
      # fic = foreign incorporation code, blank or 'USA' for US firms
      if 'fic' in df.columns:
                df = df[df['fic'].isin(['USA', '']) | df['fic'].isna()]
            print(f"After US filter: {len(df):,} observations")
    return df

def filter_industries(df):
      """Remove financial firms and utilities (SIC 6000-6999, 4900-4999)"""
    if 'sich' in df.columns:
              sic = df['sich'].fillna(0).astype(int)
              df = df[~((sic >= 6000) & (sic <= 6999))]  # Financials
        df = df[~((sic >= 4900) & (sic <= 4999))]  # Utilities
    print(f"After industry filter: {len(df):,} observations")
    return df

def create_leverage_variables(df):
      """
          Create leverage ratios and determinant variables

                  Leverage measures:
                      - book_leverage = (dltt + dlc) / at
                          - market_leverage = (dltt + dlc) / (dltt + dlc + mkvalt)

                                  Determinants (following Rajan & Zingales 1995, Frank & Goyal 2009):
                                      - size = log(at)
                                          - profitability = ebitda / at
                                              - tangibility = ppent / at
                                                  - market_to_book = mkvalt / at
                                                      - growth = (at - at_lag) / at_lag
                                                          """

    # Total debt
    df['total_debt'] = df['dltt'].fillna(0) + df['dlc'].fillna(0)

    # Book leverage
    df['book_leverage'] = df['total_debt'] / df['at']

    # Market value of equity (if not available, compute from price * shares)
    if 'mkvalt' not in df.columns or df['mkvalt'].isna().all():
              df['mkvalt'] = df['prcc_f'] * df['csho']

    # Market leverage
    df['market_leverage'] = df['total_debt'] / (df['total_debt'] + df['mkvalt'])

    # Size (log of total assets)
    df['size'] = np.log(df['at'])

    # Profitability (EBITDA / Total Assets)
    if 'ebitda' in df.columns:
              df['profitability'] = df['ebitda'] / df['at']
elif 'oibdp' in df.columns:
        df['profitability'] = df['oibdp'] / df['at']
else:
        # Compute from components: operating income + depreciation
          df['profitability'] = (df['oiadp'].fillna(0) + df['dp'].fillna(0)) / df['at']

    # Tangibility (PP&E / Total Assets)
      df['tangibility'] = df['ppent'] / df['at']

    # Market-to-Book ratio
    df['market_to_book'] = df['mkvalt'] / df['at']

    # R&D intensity
    df['rd_intensity'] = df['xrd'].fillna(0) / df['at']

    # Cash holdings
    df['cash_ratio'] = df['che'].fillna(0) / df['at']

    # Capital expenditure intensity
    df['capex_intensity'] = df['capx'].fillna(0) / df['at']

    return df

def create_industry_median_leverage(df):
      """Create industry median leverage (2-digit SIC)"""
    df['sic2'] = (df['sich'].fillna(0) / 100).astype(int)

    # Compute industry-year median leverage
    industry_median = df.groupby(['sic2', 'fyear'])['book_leverage'].transform('median')
    df['industry_median_leverage'] = industry_median

    return df

def winsorize(df, columns, limits=(0.01, 0.99)):
      """Winsorize variables at specified percentiles"""
    for col in columns:
              if col in df.columns:
                            lower = df[col].quantile(limits[0])
                            upper = df[col].quantile(limits[1])
                            df[col] = df[col].clip(lower=lower, upper=upper)
                    return df

def apply_sample_filters(df):
      """Apply standard sample filters"""
    # Require positive assets
    df = df[df['at'] > 0]

    # Require non-missing leverage
    df = df[df['book_leverage'].notna()]

    # Remove extreme leverage values (negative or > 1)
    df = df[(df['book_leverage'] >= 0) & (df['book_leverage'] <= 1)]

    # Require fiscal year in sample period
    df = df[(df['fyear'] >= START_YEAR) & (df['fyear'] <= END_YEAR)]

    print(f"After sample filters: {len(df):,} observations")
    return df

def main():
      """Main data cleaning pipeline"""

    # Create output directory
    Path("../data/processed").mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading Compustat data...")
    df = load_data(INPUT_FILE)

    # Apply filters
    print("\nApplying filters...")
    df = filter_us_firms(df)
    df = filter_industries(df)

    # Create variables
    print("\nCreating variables...")
    df = create_leverage_variables(df)
    df = create_industry_median_leverage(df)

    # Apply sample filters
    print("\nApplying sample filters...")
    df = apply_sample_filters(df)

    # Winsorize continuous variables
    print("\nWinsorizing variables...")
    vars_to_winsorize = [
              'book_leverage', 'market_leverage', 'profitability', 
              'tangibility', 'market_to_book', 'size', 
              'rd_intensity', 'cash_ratio', 'capex_intensity'
    ]
    df = winsorize(df, vars_to_winsorize)

    # Select final columns
    final_cols = [
              'gvkey', 'conm', 'fyear', 'datadate',
              'book_leverage', 'market_leverage', 'total_debt',
              'size', 'profitability', 'tangibility', 'market_to_book',
              'rd_intensity', 'cash_ratio', 'capex_intensity',
              'industry_median_leverage', 'sich', 'sic2'
    ]
    df_final = df[[c for c in final_cols if c in df.columns]]

    # Save processed data
    print(f"\nSaving to {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Sample period: {df_final['fyear'].min()} - {df_final['fyear'].max()}")
    print(f"Number of firm-years: {len(df_final):,}")
    print(f"Number of unique firms: {df_final['gvkey'].nunique():,}")
    print("\nLeverage statistics:")
    print(df_final[['book_leverage', 'market_leverage']].describe())

    return df_final

if __name__ == "__main__":
      df = main()
