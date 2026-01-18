# Unemployment and Fertility: Correlation Without Causation?

**Authors:** Sylvain Catherine, Augustin Landier, David Thesmar

## Overview

This repository contains code and data for analyzing the relationship between unemployment and birth rates in the United States. We find a negative correlation at the state level, but instrumental variable analysis suggests this does not reflect a causal relationship.

## Key Findings

- **State-level OLS:** 1 percentage point increase in unemployment is associated with a 0.21% decrease in birth rates
- **County-level OLS:** The sign reverses (+0.53%), suggesting ecological fallacy concerns
- **IV Analysis:** Strong first stages but weak reduced forms across all instruments
- **Conclusion:** The correlation does not appear to reflect causation

## Repository Structure

```
├── code/
│   ├── iv_analysis.py           # State-level IV regressions (Tables 1-3)
│   ├── county_iv_analysis.py    # County-level analysis (Tables 4-5)
│   └── ...                      # Additional analysis scripts
├── data/
│   ├── fhfa_state_hpi.csv       # State house price index (FHFA)
│   ├── wti_oil_prices.csv       # Oil prices (FRED)
│   ├── fred_manufacturing_emp.csv
│   ├── county_panel.csv         # County-level birth/population data
│   └── ...
└── paper/
    ├── main.tex                 # LaTeX source
    ├── main.pdf                 # Compiled paper
    └── references.bib           # Bibliography
```

## Instruments

We test four instrumental variable strategies:

1. **Housing Price Shocks** - State-level HPI growth (FHFA)
2. **Oil Price Shocks** - WTI price changes × oil-producing state indicator
3. **Bartik (Manufacturing)** - National manufacturing growth × local manufacturing share
4. **China Trade Shock** - ADH-style import exposure

## Reproducing Results

```bash
# State-level analysis (Tables 1-3)
python code/iv_analysis.py

# County-level analysis (Tables 4-5)
python code/county_iv_analysis.py
```

Requirements: Python 3, pandas, numpy, statsmodels

## Data Sources

- Birth rates: CDC WONDER
- Unemployment: Bureau of Labor Statistics (LAUS)
- House prices: Federal Housing Finance Agency (FHFA)
- Oil prices: FRED (DCOILWTICO)
- Manufacturing employment: FRED (MANEMP)
