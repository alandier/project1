# BRFSS State-Year Wellbeing Data

## Overview

This dataset contains state-year aggregates of subjective wellbeing measures from the CDC Behavioral Risk Factor Surveillance System (BRFSS), the largest continuously conducted health survey in the world.

## Data Files

| File | Format | Description |
|------|--------|-------------|
| `state_year_wellbeing.parquet` | Apache Parquet | Primary format (recommended) |
| `state_year_wellbeing.csv` | CSV | Comma-separated values |
| `state_year_wellbeing.dta` | Stata | For Stata users |

## Coverage

- **Years**: 2005-2023 (19 years)
- **States**: 54 (50 states + DC + GU + PR + VI)
- **Observations**: 1,011 state-years
- **Source records**: 8.3 million individual respondents

## Variables

### Identifiers
| Variable | Description |
|----------|-------------|
| `year` | Survey year (2005-2023) |
| `state_fips` | State FIPS code |
| `state_usps` | State postal abbreviation |
| `n_unweighted` | Number of respondents in cell |

### Life Satisfaction (LSATISFY)
Available: 2005-2010, select states 2013-2017, 2022-2023

| Variable | Description |
|----------|-------------|
| `brfss_lifesat_mean` | Mean life satisfaction (1=Very Satisfied, 4=Very Dissatisfied) |
| `brfss_lifesat_n` | Number of respondents with valid life satisfaction |
| `brfss_lifesat_share_very_satisfied` | Share responding "Very satisfied" |
| `brfss_lifesat_share_satisfied` | Share responding "Satisfied" |
| `brfss_lifesat_share_dissatisfied` | Share responding "Dissatisfied" |
| `brfss_lifesat_share_very_dissatisfied` | Share responding "Very dissatisfied" |

### Mental Health Days (MENTHLTH)
Available: All years (100% coverage)

| Variable | Description |
|----------|-------------|
| `brfss_menthlth_mean_days` | Mean days of poor mental health in past 30 days |
| `brfss_menthlth_n` | Number of respondents with valid mental health days |
| `brfss_freq_mental_distress_share` | Share with frequent mental distress (14+ days) |

### General Health (GENHLTH)
Available: All years (100% coverage)

| Variable | Description |
|----------|-------------|
| `brfss_genhlth_mean` | Mean general health (1=Excellent, 5=Poor) |
| `brfss_fair_poor_health_share` | Share reporting "Fair" or "Poor" health |

## Variable Coverage by Year

```
Year    LifeSat  MentHlth  GenHlth
2005    53/53    53/53     53/53
2006    53/53    53/53     53/53
2007    54/54    54/54     54/54
2008    54/54    54/54     54/54
2009    54/54    54/54     54/54
2010    54/54    54/54     54/54
2011     0/53    53/53     53/53
2012     0/53    53/53     53/53
2013     2/53    53/53     53/53
2014     1/53    53/53     53/53
2015     2/53    53/53     53/53
2016     4/54    54/54     54/54
2017     2/53    53/53     53/53
2018     0/53    53/53     53/53
2019     0/52    52/52     52/52
2020     0/53    53/53     53/53
2021     0/53    53/53     53/53
2022    37/54    54/54     54/54
2023    35/52    52/52     52/52
```

Note: Life satisfaction (LSATISFY) was removed from the BRFSS core questionnaire after 2010 and only asked in select states 2011-2021 via optional modules.

## Data Quality

All validation checks pass:
- Life satisfaction mean in expected range [1, 4]
- Life satisfaction shares sum to ~1.0
- Mental distress share in [0, 1]
- Mental health days in [0, 30]
- All state FIPS codes are valid US codes

## Usage

### Python
```python
import pandas as pd

# Load data
df = pd.read_parquet('data/processed/wellbeing/state_year_wellbeing.parquet')

# Merge with panel data
from src.construct.merge_wellbeing import merge_wellbeing
merged, diagnostics = merge_wellbeing(your_panel, state_col='state', year_col='year')
```

### Stata
```stata
use "data/processed/wellbeing/state_year_wellbeing.dta", clear
```

### R
```r
library(arrow)
df <- read_parquet("data/processed/wellbeing/state_year_wellbeing.parquet")
```

## Source

**CDC Behavioral Risk Factor Surveillance System (BRFSS)**
- Website: https://www.cdc.gov/brfss/
- Data files: https://www.cdc.gov/brfss/annual_data/annual_data.htm

## Methodology

1. Downloaded annual BRFSS XPT files from CDC
2. Extracted wellbeing variables (LSATISFY, MENTHLTH, GENHLTH)
3. Applied survey weights (_LLCPWT for 2011+, _FINALWT for pre-2011)
4. Aggregated to state-year level using weighted means
5. Computed derived measures (frequent mental distress share, life satisfaction shares)

## Scripts

| Script | Purpose |
|--------|---------|
| `src/ingest/ingest_brfss_wellbeing.py` | Download and process BRFSS data |
| `src/construct/merge_wellbeing.py` | Merge wellbeing data into analysis panels |
| `src/construct/validate_wellbeing.py` | Run validation and produce reports |

## Validation Outputs

- `coverage_by_year.csv` - Coverage summary by year
- `descriptive_stats.csv` - Descriptive statistics for all variables
- `national_time_series.csv` - National averages by year
- `lifesat_missingness_matrix.csv` - State x year missingness for life satisfaction

## Citation

If using this data, please cite the original BRFSS:

> Centers for Disease Control and Prevention (CDC). Behavioral Risk Factor Surveillance System Survey Data. Atlanta, Georgia: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, [years used].

## Last Updated

Generated: 2024
