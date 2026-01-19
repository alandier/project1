"""
BRFSS Wellbeing Data Ingestion Script
=====================================
Downloads and processes CDC BRFSS annual microdata to extract subjective wellbeing
variables at the state-year level.

Author: Research pipeline
Created: 2024
"""

import os
import sys
import json
import logging
import requests
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False
    print("Warning: pyreadstat not available, will try pandas.read_sas")

# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path('/Users/landieraugustin/project1')
RAW_PATH = BASE_PATH / 'data' / 'raw' / 'brfss'
PROCESSED_PATH = BASE_PATH / 'data' / 'processed' / 'wellbeing'
LOG_PATH = BASE_PATH / 'logs'

# Create directories
RAW_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH / 'brfss_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CDC BRFSS base URL
CDC_BASE = "https://www.cdc.gov/brfss/annual_data"

# URL patterns by year
def get_brfss_url(year: int) -> str:
    """Get download URL for BRFSS XPT file by year."""
    if year >= 2011:
        # Combined landline + cell phone data (LLCP format)
        return f"{CDC_BASE}/{year}/files/LLCP{year}XPT.zip"
    else:
        # Older format (CDBRFS)
        yy = str(year)[-2:]
        return f"{CDC_BASE}/{year}/files/CDBRFS{yy}XPT.zip"

# Variable names by year (these can vary - we'll try multiple variants)
LIFE_SATISFACTION_VARS = ['LSATISFY', 'LSATIS', '_LSATISFY']
MENTAL_HEALTH_VARS = ['MENTHLTH', '_MENT14', 'POORMENT']
GENERAL_HEALTH_VARS = ['GENHLTH', '_GENHLTH', 'GENHLTH1']
PHYSICAL_HEALTH_VARS = ['PHYSHLTH', '_PHYS14', 'POORHLTH']
STATE_VAR = '_STATE'
WEIGHT_VARS = ['_LLCPWT', '_FINALWT', '_WT2RAKE', '_LANDWT', 'WT2RAKE', 'FINALWT', '_WTFRESP']
STRATA_VAR = '_STSTR'
PSU_VAR = '_PSU'

# State FIPS to USPS mapping
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

# =============================================================================
# Download Functions
# =============================================================================

def download_brfss_year(year: int, force: bool = False) -> Optional[Path]:
    """
    Download BRFSS XPT file for a given year.

    Args:
        year: Survey year
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded/extracted XPT file, or None if failed
    """
    year_dir = RAW_PATH / str(year)
    year_dir.mkdir(exist_ok=True)

    # Check for existing XPT file
    existing_xpt = list(year_dir.glob('*.XPT')) + list(year_dir.glob('*.xpt'))
    if existing_xpt and not force:
        logger.info(f"Year {year}: Using existing file {existing_xpt[0].name}")
        return existing_xpt[0]

    url = get_brfss_url(year)
    zip_path = year_dir / f"brfss_{year}.zip"

    logger.info(f"Year {year}: Downloading from {url}")

    try:
        # Download with streaming
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                    pct = 100 * downloaded / total_size
                    logger.info(f"  Downloaded {downloaded / 1e6:.1f} MB ({pct:.0f}%)")

        logger.info(f"Year {year}: Download complete ({zip_path.stat().st_size / 1e6:.1f} MB)")

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Handle filenames with trailing spaces or different cases
            xpt_files = [f for f in zf.namelist() if f.strip().upper().endswith('.XPT')]
            if xpt_files:
                zf.extract(xpt_files[0], year_dir)
                extracted_path = year_dir / xpt_files[0]
                # Handle filenames with trailing spaces by renaming
                clean_name = xpt_files[0].strip()
                xpt_path = year_dir / clean_name
                if extracted_path != xpt_path and extracted_path.exists():
                    extracted_path.rename(xpt_path)
                logger.info(f"Year {year}: Extracted {clean_name}")

                # Log metadata
                metadata = {
                    'source_url': url,
                    'download_date': datetime.now().isoformat(),
                    'file_size_mb': xpt_path.stat().st_size / 1e6,
                    'xpt_file': xpt_files[0]
                }
                with open(year_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

                return xpt_path
            else:
                logger.error(f"Year {year}: No XPT file found in ZIP")
                return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Year {year}: Download failed - {e}")
        return None
    except zipfile.BadZipFile as e:
        logger.error(f"Year {year}: Invalid ZIP file - {e}")
        return None

# =============================================================================
# Processing Functions
# =============================================================================

def find_variable(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching variable from a list of candidates."""
    df_cols_upper = {c.upper(): c for c in df.columns}
    for var in candidates:
        if var.upper() in df_cols_upper:
            return df_cols_upper[var.upper()]
    return None

def process_brfss_year(xpt_path: Path, year: int) -> Optional[pd.DataFrame]:
    """
    Process a single year of BRFSS data to extract wellbeing variables.

    Args:
        xpt_path: Path to XPT file
        year: Survey year

    Returns:
        DataFrame with individual-level data, or None if failed
    """
    logger.info(f"Year {year}: Processing {xpt_path.name}")

    try:
        # Read XPT file - use pandas with latin-1 encoding (more robust for BRFSS files)
        df = pd.read_sas(xpt_path, format='xport', encoding='latin-1')
        logger.info(f"Year {year}: Loaded {len(df):,} records, {len(df.columns)} variables")

        # Find required variables
        state_col = find_variable(df, [STATE_VAR])
        weight_col = find_variable(df, WEIGHT_VARS)

        if not state_col:
            logger.warning(f"Year {year}: State variable not found")
            return None

        # Find wellbeing variables
        lifesat_col = find_variable(df, LIFE_SATISFACTION_VARS)
        menthlth_col = find_variable(df, MENTAL_HEALTH_VARS)
        genhlth_col = find_variable(df, GENERAL_HEALTH_VARS)

        logger.info(f"Year {year}: Found variables - state={state_col}, weight={weight_col}, "
                   f"lifesat={lifesat_col}, menthlth={menthlth_col}, genhlth={genhlth_col}")

        # Build output dataframe
        n_rows = len(df)
        result = pd.DataFrame()
        result['year'] = [year] * n_rows  # Explicit list assignment
        result['state_fips'] = df[state_col].astype(float).astype(int).values

        # Weight
        if weight_col:
            result['weight'] = df[weight_col].astype(float)
        else:
            result['weight'] = 1.0
            logger.warning(f"Year {year}: No weight variable found, using uniform weights")

        # Life satisfaction (typically coded 1-4: Very satisfied, Satisfied, Dissatisfied, Very dissatisfied)
        if lifesat_col:
            lifesat = df[lifesat_col].astype(float)
            # Handle missing codes (7=Don't know, 9=Refused, blank)
            lifesat = lifesat.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
            lifesat[(lifesat < 1) | (lifesat > 4)] = np.nan
            result['lifesat'] = lifesat
            result['has_lifesat'] = True
        else:
            result['lifesat'] = np.nan
            result['has_lifesat'] = False

        # Mental health days (0-30, with 88=None, 77=Don't know, 99=Refused)
        if menthlth_col:
            menthlth = df[menthlth_col].astype(float)
            # 88 = None (0 days)
            menthlth = menthlth.replace({88: 0})
            # Handle missing codes
            menthlth = menthlth.replace({77: np.nan, 99: np.nan})
            menthlth[(menthlth < 0) | (menthlth > 30)] = np.nan
            result['menthlth_days'] = menthlth
            result['has_menthlth'] = True
        else:
            result['menthlth_days'] = np.nan
            result['has_menthlth'] = False

        # General health (1-5: Excellent to Poor)
        if genhlth_col:
            genhlth = df[genhlth_col].astype(float)
            genhlth = genhlth.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
            genhlth[(genhlth < 1) | (genhlth > 5)] = np.nan
            result['genhlth'] = genhlth
        else:
            result['genhlth'] = np.nan

        # Filter to valid US states/DC (FIPS 1-56, plus territories)
        result = result[result['state_fips'].isin(STATE_FIPS_TO_USPS.keys())]

        logger.info(f"Year {year}: Processed {len(result):,} valid records")
        return result

    except Exception as e:
        logger.error(f"Year {year}: Processing failed - {e}")
        return None

def aggregate_to_state_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual-level data to state-year level using survey weights.

    Args:
        df: Individual-level DataFrame with weight column

    Returns:
        State-year aggregated DataFrame
    """
    results = []

    for (year, state_fips), group in df.groupby(['year', 'state_fips']):

        row = {
            'year': int(year),
            'state_fips': int(state_fips),
            'state_usps': STATE_FIPS_TO_USPS.get(int(state_fips), ''),
            'n_unweighted': len(group)
        }

        # Get weights
        w = group['weight'].fillna(1.0)
        w_sum = w.sum()

        if w_sum == 0:
            continue

        # Life satisfaction (if available for this year)
        if group['has_lifesat'].any() and group['lifesat'].notna().sum() > 0:
            ls = group['lifesat']
            ls_valid = ls.notna()
            w_ls = w[ls_valid]
            ls = ls[ls_valid]

            if len(ls) > 0 and w_ls.sum() > 0:
                # Weighted mean
                row['brfss_lifesat_mean'] = np.average(ls, weights=w_ls)
                row['brfss_lifesat_n'] = len(ls)

                # Category shares (weighted)
                w_ls_sum = w_ls.sum()
                row['brfss_lifesat_share_very_satisfied'] = w_ls[ls == 1].sum() / w_ls_sum
                row['brfss_lifesat_share_satisfied'] = w_ls[ls == 2].sum() / w_ls_sum
                row['brfss_lifesat_share_dissatisfied'] = w_ls[ls == 3].sum() / w_ls_sum
                row['brfss_lifesat_share_very_dissatisfied'] = w_ls[ls == 4].sum() / w_ls_sum

        # Mental health days
        if group['has_menthlth'].any() and group['menthlth_days'].notna().sum() > 0:
            mh = group['menthlth_days']
            mh_valid = mh.notna()
            w_mh = w[mh_valid]
            mh = mh[mh_valid]

            if len(mh) > 0 and w_mh.sum() > 0:
                # Weighted mean days
                row['brfss_menthlth_mean_days'] = np.average(mh, weights=w_mh)
                row['brfss_menthlth_n'] = len(mh)

                # Frequent mental distress: 14+ days
                distress = (mh >= 14).astype(float)
                row['brfss_freq_mental_distress_share'] = np.average(distress, weights=w_mh)

        # General health
        if group['genhlth'].notna().sum() > 0:
            gh = group['genhlth']
            gh_valid = gh.notna()
            w_gh = w[gh_valid]
            gh = gh[gh_valid]

            if len(gh) > 0 and w_gh.sum() > 0:
                row['brfss_genhlth_mean'] = np.average(gh, weights=w_gh)
                # Fair or poor health (4 or 5)
                poor_health = (gh >= 4).astype(float)
                row['brfss_fair_poor_health_share'] = np.average(poor_health, weights=w_gh)

        results.append(row)

    return pd.DataFrame(results)

# =============================================================================
# Main Pipeline
# =============================================================================

def run_brfss_pipeline(start_year: int = 2000, end_year: int = 2023,
                       force_download: bool = False) -> pd.DataFrame:
    """
    Run the full BRFSS ingestion pipeline.

    Args:
        start_year: First year to process
        end_year: Last year to process
        force_download: If True, re-download all files

    Returns:
        State-year level wellbeing DataFrame
    """
    logger.info(f"Starting BRFSS pipeline for years {start_year}-{end_year}")

    all_individual = []

    for year in range(start_year, end_year + 1):
        # Download
        xpt_path = download_brfss_year(year, force=force_download)
        if xpt_path is None:
            logger.warning(f"Skipping year {year} - download failed")
            continue

        # Process
        year_df = process_brfss_year(xpt_path, year)
        if year_df is not None:
            all_individual.append(year_df)

        # Progress
        logger.info(f"Completed year {year}")

    if not all_individual:
        logger.error("No data processed successfully")
        return pd.DataFrame()

    # Combine all years
    logger.info("Combining all years...")
    individual_df = pd.concat(all_individual, ignore_index=True)
    logger.info(f"Total individual records: {len(individual_df):,}")

    # Aggregate to state-year
    logger.info("Aggregating to state-year level...")
    state_year_df = aggregate_to_state_year(individual_df)
    logger.info(f"State-year observations: {len(state_year_df)}")

    # Sort
    state_year_df = state_year_df.sort_values(['year', 'state_fips']).reset_index(drop=True)

    return state_year_df

# =============================================================================
# Export Functions
# =============================================================================

def export_wellbeing_data(df: pd.DataFrame):
    """Export wellbeing data to multiple formats."""

    # Ensure output directory exists
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    # Parquet
    parquet_path = PROCESSED_PATH / 'state_year_wellbeing.parquet'
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved: {parquet_path}")

    # CSV
    csv_path = PROCESSED_PATH / 'state_year_wellbeing.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Stata
    try:
        dta_path = PROCESSED_PATH / 'state_year_wellbeing.dta'
        df.to_stata(dta_path, write_index=False)
        logger.info(f"Saved: {dta_path}")
    except Exception as e:
        logger.warning(f"Could not save Stata file: {e}")

# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BRFSS Wellbeing Data Ingestion')
    parser.add_argument('--start-year', type=int, default=2005,
                       help='First year to process (default: 2005)')
    parser.add_argument('--end-year', type=int, default=2023,
                       help='Last year to process (default: 2023)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download of all files')

    args = parser.parse_args()

    # Run pipeline
    df = run_brfss_pipeline(args.start_year, args.end_year, args.force)

    if len(df) > 0:
        # Export
        export_wellbeing_data(df)

        # Summary
        print("\n" + "="*70)
        print("BRFSS WELLBEING DATA SUMMARY")
        print("="*70)
        print(f"Years covered: {df['year'].min()}-{df['year'].max()}")
        print(f"Total state-years: {len(df)}")
        print(f"States: {df['state_fips'].nunique()}")
        print(f"\nVariable coverage:")
        for col in df.columns:
            if col.startswith('brfss_'):
                n_valid = df[col].notna().sum()
                print(f"  {col}: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")
        print("\nData saved to:", PROCESSED_PATH)
    else:
        print("No data processed. Check logs for errors.")
