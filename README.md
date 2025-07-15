# Historical Beta Analysis Project

A comprehensive, generalized workflow for analyzing historical factor betas across different market indices. This project provides tools for data validation, transformation, consolidation, weighted average calculation, and visualization of factor betas for any market index.

## Features

- **Multi-Index Support**: Easily switch between different indices (S&P 500, NASDAQ, Dow Jones, Russell 2000, or custom indices)
- **Data Pipeline**: Complete workflow from raw data validation to final visualization
- **Robust Processing**: Handles missing data, outliers, and various data formats
- **Interactive Visualization**: Plotly-based interactive charts for factor evolution analysis
- **Flexible Configuration**: JSON-based configuration system for easy customization
- **Command Line Interface**: Simple CLI for running individual steps or complete workflows

## Project Structure

```
historical_beta_project/
├── config.json                 # Configuration file for different indices
├── README.md
├── .gitignore
├── data_marts/                 # Final processed data for analysis
├── data_warehouse/             # Consolidated intermediate data
├── raw_data/                   # Raw input data
│   ├── Master_Factor_Betas_Raw/
│   ├── master_factor_betas_transformed/
│   └── [INDEX]_holdings/       # Holdings files for each index
├── src/
│   ├── config.py              # Configuration management
│   ├── generalized_beta_visualizer.py  # Visualization tools
│   └── beta_visualizer.py     # Legacy S&P 500 visualizer
└── util/
    ├── run_workflow.py         # Complete workflow runner
    ├── consolidate_holdings.py # Holdings consolidation
    ├── calculate_weighted_betas.py  # Weighted beta calculation
    ├── consolidate_betas.py    # Factor beta consolidation
    ├── validate_beta_ranges.py # Data validation
    └── batch_transform_raw_betas.py  # Data transformation
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
raw_data/
├── Master_Factor_Betas_Raw/          # Common factor betas (all indices use this)
├── master_factor_betas_transformed/   # Transformed factor betas (generated)
└── [INDEX_NAME]_holdings/            # Holdings files for your index
    ├── [INDEX_NAME]_Holdings_MM_DD_YYYY.csv
    └── ...
```

Holdings files should contain columns for `Symbol` and `Weight` (percentage or decimal).

### 3. Run Complete Workflow

```bash
# For S&P 500 (default)
python util/run_workflow.py SP500

# For NASDAQ
python util/run_workflow.py NASDAQ

# For any other configured index
python util/run_workflow.py [INDEX_NAME]
```

### 4. Visualize Results

```bash
# Show summary statistics
python src/generalized_beta_visualizer.py SP500 --summary

# Plot factor evolution
python src/generalized_beta_visualizer.py SP500

# Analyze specific ticker vs index
python src/generalized_beta_visualizer.py SP500 --ticker AAPL

# Export to CSV
python src/generalized_beta_visualizer.py SP500 --export
```

## Configuration

### Built-in Indices

The following indices are pre-configured:
- **SP500**: S&P 500
- **NASDAQ**: NASDAQ Composite  
- **DOW**: Dow Jones Industrial Average
- **RUSSELL2000**: Russell 2000

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

### Data Validation
```bash
python util/validate_beta_ranges.py
```

### Data Transformation
```bash
python util/batch_transform_raw_betas.py
```

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

### Visualization
```bash
# Basic factor evolution plot
python src/generalized_beta_visualizer.py SP500

# Specific factors
python src/generalized_beta_visualizer.py SP500 --factors 1 2 3

# Date range filtering
python src/generalized_beta_visualizer.py SP500 --start-date 2022-01-01 --end-date 2023-01-01

# Individual stock analysis
python src/generalized_beta_visualizer.py SP500 --ticker AAPL

# Export results
python src/generalized_beta_visualizer.py SP500 --export
```

## Data Requirements

### Factor Beta Files
- Location: `raw_data/Master_Factor_Betas_Raw/`
- Format: CSV with columns `Date`, `Symbol`, and numeric factor columns
- Naming: `master_factor_betas_MM_DD_YYYY.csv`

### Holdings Files
- Location: `raw_data/[INDEX_NAME]_holdings/`
- Format: CSV with columns for symbol and weight
- Naming: `[INDEX_NAME]_Holdings_MM_DD_YYYY.csv`
- Weight: Can be percentage (0-100) or decimal (0-1)

## Output Files

### Data Warehouse
- `consolidated_factor_betas.parquet`: All factor betas combined
- `[INDEX]_holdings_consolidated.parquet`: All holdings combined

### Data Marts
- `[INDEX]_weekly_factor_betas.parquet`: Weighted average factor betas
- `[INDEX]_weekly_factor_betas.csv`: CSV export of weighted betas

## Development

### Configuration Management
The `src/config.py` module provides centralized configuration management:
- `get_config()`: Get global configuration instance
- `set_index(index_name)`: Set current working index
- `get_paths(index_name)`: Get all file paths for an index

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

### Validation Commands
```bash
# Check available indices
python util/run_workflow.py --list-indices

# Validate requirements
python util/run_workflow.py SP500 --validate-only

# Check data availability
python util/calculate_weighted_betas.py SP500 --validate
```

## License

This project is for internal use and analysis of historical factor beta data.