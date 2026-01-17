"""
01_extract_compustat.py
Extract Compustat annual fundamentals from WRDS for leverage analysis

Author: Research collaboration
Project: Determinants of Firm Leverage in the US
"""

import wrds
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
START_YEAR = 1980
END_YEAR = 2023
OUTPUT_DIR = Path("../data/raw")

def connect_wrds():
      """Establish WRDS connection"""
      return wrds.Connection()

def extract_compustat_annual(conn):
      """
          Extract annual Compustat data with key variables for leverage analysis

                  Variables:
                      - Leverage: dltt (long-term debt), dlc (debt in current liabilities), at (total assets)
                          - Size: at, sale
                              - Profitability: oibdp (operating income), ebitda
                                  - Tangibility: ppent (net PP&E)
                                      - Market value: prcc_f (price close), csho (shares outstanding)
                                          - Growth: capx (capital expenditures)
                                              - Other: sich (SIC code), fyear (fiscal year)
                                                  """

    query = """
        SELECT 
                gvkey,
                        datadate,
                                fyear,
                                        conm,
                                                sich,
                                                        -- Assets and size
                                                                at,
                                                                        sale,
                                                                                -- Debt variables
                                                                                        dltt,
                                                                                                dlc,
                                                                                                        -- Profitability
                                                                                                                oibdp,
                                                                                                                        ebitda,
                                                                                                                                ni,
                                                                                                                                        -- Tangibility
                                                                                                                                                ppent,
                                                                                                                                                        ppegt,
                                                                                                                                                                -- Market value components
                                                                                                                                                                        prcc_f,
                                                                                                                                                                                csho,
                                                                                                                                                                                        -- Book equity components
                                                                                                                                                                                                seq,
                                                                                                                                                                                                        ceq,
                                                                                                                                                                                                                txditc,
                                                                                                                                                                                                                        -- Investment
                                                                                                                                                                                                                                capx,
                                                                                                                                                                                                                                        -- Cash
                                                                                                                                                                                                                                                che,
                                                                                                                                                                                                                                                        -- Dividends
                                                                                                                                                                                                                                                                dv
                                                                                                                                                                                                                                                                    FROM comp.funda
                                                                                                                                                                                                                                                                        WHERE 
                                                                                                                                                                                                                                                                                indfmt = 'INDL'
                                                                                                                                                                                                                                                                                        AND datafmt = 'STD'
                                                                                                                                                                                                                                                                                                AND popsrc = 'D'
                                                                                                                                                                                                                                                                                                        AND consol = 'C'
                                                                                                                                                                                                                                                                                                                AND fyear BETWEEN {start} AND {end}
                                                                                                                                                                                                                                                                                                                        AND at > 0
                                                                                                                                                                                                                                                                                                                                AND fic = 'USA'
                                                                                                                                                                                                                                                                                                                                    ORDER BY gvkey, fyear
                                                                                                                                                                                                                                                                                                                                        """.format(start=START_YEAR, end=END_YEAR)

    print("Extracting Compustat annual data...")
    df = conn.raw_sql(query)
    print(f"Retrieved {len(df):,} firm-year observations")

    return df

def compute_leverage_variables(df):
      """Compute leverage ratios and control variables"""

    df = df.copy()

    # Book leverage = (Long-term debt + Debt in current liabilities) / Total assets
    df['book_leverage'] = (df['dltt'].fillna(0) + df['dlc'].fillna(0)) / df['at']

    # Market value of equity
    df['mve'] = df['prcc_f'] * df['csho']

    # Market leverage = Total debt / (Total debt + Market value of equity)
    df['total_debt'] = df['dltt'].fillna(0) + df['dlc'].fillna(0)
    df['market_leverage'] = df['total_debt'] / (df['total_debt'] + df['mve'])

    # Size (log of assets)
    df['size'] = np.log(df['at'])

    # Profitability (ROA)
    df['profitability'] = df['oibdp'] / df['at']

    # Tangibility
    df['tangibility'] = df['ppent'] / df['at']

    # Market-to-book ratio
    df['book_equity'] = df['seq'].fillna(df['ceq'] + df['txditc'].fillna(0))
    df['mtb'] = df['mve'] / df['book_equity'].replace(0, np.nan)

    # Growth (Capex / Assets)
    df['investment'] = df['capx'] / df['at']

    return df

def main():
      """Main extraction pipeline"""

    # Create output directory
      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to WRDS
    conn = connect_wrds()

    try:
              # Extract raw data
              df_raw = extract_compustat_annual(conn)

        # Save raw data
              raw_path = OUTPUT_DIR / "compustat_annual_raw.parquet"
              df_raw.to_parquet(raw_path, index=False)
              print(f"Saved raw data to {raw_path}")

        # Compute leverage variables
              df_processed = compute_leverage_variables(df_raw)

        # Save processed data
              processed_path = OUTPUT_DIR / "compustat_leverage_vars.parquet"
              df_processed.to_parquet(processed_path, index=False)
              print(f"Saved processed data to {processed_path}")

        # Summary statistics
              print("\n=== Summary Statistics ===")
              print(df_processed[['book_leverage', 'market_leverage', 'size', 
                                 'profitability', 'tangibility', 'mtb']].describe())

finally:
          conn.close()
          print("\nWRDS connection closed")

if __name__ == "__main__":
      main()
