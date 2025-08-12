# Historical Beta Analysis Project

A comprehensive, generalized workflow for analyzing historical factor betas across different market indices.

## Features

- **Multi-Index Support**: Easily switch between different indices
- **Robust Processing**: Handles missing data, outliers, and various data formats
- **Flexible Configuration**: JSON-based configuration system for easy customization
- **Command Line Interface**: Simple CLI for running individual steps or complete workflows

## Project Structure

```
historical_beta_project/
├── config.json # Index and file path configuration
├── README.md # Project documentation
├── .gitignore # Git ignore rules
├── data_marts/ Final processed data for analysis
│   └── SP500_weekly_factor_betas.csv # S&P output sector,weight,beta
├── data_warehouse/ # Intermediate data
│   ├── consolidated_factor_betas.parquet # All factor betas combined
│   ├── SP500_holdings_consolidated.parquet # All SP500 holdings combined
│   └── SP500_symbols_sectors.csv # Manually created sector classification for SP500
├── raw_data/ # Raw input data
│   ├── master_factor_betas_transformed/ # All transformed (bet -1.5 and +1.5) beta files
│   ├── master_sector_files/ # All sector files
│   └── SP500_holdings/ # Raw SP500 holdings files
├── src/
│   └── run_full_update_workflow.py # Main workflow runner (automates all steps except sector classification)
└── util/
    ├── calculate_sector_weekly_avg_betas.py # Aggregates sector-level weekly betas
    ├── consolidate_betas.py # Consolidates factor beta files
    ├── consolidate_holdings.py # Consolidates holdings files for an index
    └── consolidate_master_sector.py # Analyzes sector files, finds conflicts, and produces sector classification outputs
```

## Quick Start

### 1. Setup

Ensure you have Python 3.8+ installed with the following packages:
```bash
pip install pandas numpy matplotlib seaborn plotly pathlib
```

### 2. Prepare Your Data

For any index, create a folder structure like:
```
raw_data/ # From google drive
├── master_factor_betas_tranformed/ # All factor betas
├── master_sector_files/ # All master sector files
└── [INDEX_NAME]_holdings/            # Holdings files for specific index
    ├── [INDEX_NAME]_Holdings_MM_DD_YYYY.csv
    └── ...
```

Holdings files should contain columns for `Symbol` and `Weight` (percentage or decimal).

### 3. Run Complete Workflow

```bash

python util/run_workflow.py

```

## Configuration

### Built-in Indices

The following indices are pre-configured:  
- **SP500**: S&P 500  
- **NASDAQ**: NASDAQ Composite  
- **DOW**: Dow Jones Industrial Average  
- **RUSSELL2000**: Russell 2000  
- **SP500G**: S&P 500 Growth  
- **SP500ESG**: S&P 500 ESG  
- **SP500V**: S&P 500 Value  
- **SP1500**: S&P 1500  
- **SP600**: S&P 600  
- **SP400**: S&P 400  
- **RUSSELL1000**: Russell 1000  
- **RUSSELL3000**: Russell 3000  
- **EAFE**: EAFE  
- **EEM**: EEM  
- **ACWI**: ACWI  
- **QQQ**: NASDAQ 100  


### Adding Custom Indices

1. Edit `config.json` to add your index:
```json
{
  "indices": {
    "CUSTOM_INDEX": {
      "name": "CUSTOM_INDEX",
      "display_name": "My Custom Index",
      "holdings_folder": "CUSTOM_INDEX_holdings",
      "holdings_file_pattern": "CUSTOM_INDEX_Holdings_*.csv",
      "holdings_consolidated_file": "CUSTOM_INDEX_holdings_consolidated.parquet",
      "weighted_betas_file": "CUSTOM_INDEX_weekly_factor_betas.parquet",
      "csv_export_file": "CUSTOM_INDEX_weekly_factor_betas.csv"
    }
  }
}
```

2. Create the holdings folder: `raw_data/CUSTOM_INDEX_holdings/`

3. Add holdings files following the naming pattern: `CUSTOM_INDEX_Holdings_MM_DD_YYYY.csv`

## Individual Script Usage

### Consolidation
```bash
# Consolidate factor betas (common for all indices)
python util/consolidate_betas.py

# Consolidate holdings for specific index
python util/consolidate_holdings.py SP500
```

### Weighted Beta Calculation
```bash
python util/calculate_weighted_betas.py SP500
```

## Data Requirements


### Factor Beta Files
- Location: `raw_data/master_factor_betas_transformed/`
- Format: CSV, columns: `Symbol`, `Company Name`, factor columns `1`-`88`
- Naming: `master_factor_betas_MM_DD_YYYY.csv`

### Sector Files
- Location: `raw_data/master_sector_files/`
- Format: CSV, columns: `Symbol`, `Company Name`, `Sector`
- Naming: `master_sector_MM_DD_YYYY.csv`

### Holdings Files
- Location: `raw_data/[INDEX]_holdings/`
- Format: CSV, columns: `Symbol`, `Weight`
- Naming: `[INDEX]_Holdings_MM_DD_YYYY.csv`
- Weight: decimal (0-1)

### Output Sector Classification Files
- Location: `data_warehouse/`
- Format: CSV, columns: `Symbol`, `Company Name`, `Sector`
- Naming: `[INDEX]_symbols_sectors.csv` (final, after resolving conflicts manually)
- Note: `[INDEX]_symbols_sectors_conflicts.csv` (tickers with sector conflicts), `[INDEX]_symbols_sectors_no_conflicts.csv` (tickers with no sector conflicts)

## Output Files

### Data Warehouse
- `consolidated_factor_betas.parquet`: All factor betas combined
- `[INDEX]_holdings_consolidated.parquet`: All holdings combined

### Data Marts
- `[INDEX]_weekly_factor_betas.parquet`: Weighted average factor betas
- `[INDEX]_weekly_factor_betas.csv`: CSV export of weighted betas

## Development

## Adding a New Week of Betas and Holdings

To add a new week of data before running the workflow:

1. **Add New Factor Beta File**
   - Place the new weekly beta file in `raw_data/master_factor_betas_transformed/`.
   - Name it using the format: `master_factor_betas_MM_DD_YYYY.csv` (e.g., `master_factor_betas_08_12_2025.csv`).
   - Ensure columns include `Date`, `Symbol`, and all required factor columns.

2. **Add New Holdings File**
   - Place the new weekly holdings file in `raw_data/[INDEX]_holdings/` (replace `[INDEX]` with your index, e.g., `SP500`).
   - Name it using the format: `[INDEX]_Holdings_MM_DD_YYYY.csv` (e.g., `SP500_Holdings_08_12_2025.csv`).
   - Ensure columns include `Symbol` and `Weight` (as decimal or percentage).

After adding the files, run the workflow as usual. The new week will be automatically detected and processed.

### Adding New Indices
1. Add configuration to `config.json`
2. Create holdings data folder
3. Run workflow: `python util/run_workflow.py NEW_INDEX`

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure all required files exist in the correct locations
2. **Date parsing errors**: Check that filenames follow the expected format
3. **Weight format issues**: Ensure weight columns contain numeric values
4. **Factor column problems**: Verify factor betas have numeric column names

## License

This project is for internal use and analysis of historical factor beta data.