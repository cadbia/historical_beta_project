
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_sector_weekly_avg_betas.py <INDEX_KEY>")
        sys.exit(1)
    index_key = sys.argv[1]
    with open('config.json', 'r') as f:
        config = json.load(f)
    indices = config['indices']
    if index_key not in indices:
        print(f"Index '{index_key}' not found in config.json.")
        sys.exit(1)
    index_cfg = indices[index_key]

    # File paths
    HOLDINGS_PATH = Path(index_cfg['holdings_consolidated_file'])
    BETAS_PATH = Path('data_warehouse/consolidated_factor_betas.parquet')
    SECTOR_MAP_PATH = Path(index_cfg['symbol_sector_file'])
    OUTPUT_PATH = Path(index_cfg['csv_export_file'])

    # Read data
    holdings = pd.read_parquet(HOLDINGS_PATH)
    betas = pd.read_parquet(BETAS_PATH)
    sector_map = pd.read_csv(SECTOR_MAP_PATH)

    # Standardize column names
    holdings.rename(columns={"week": "Date", "symbol": "Symbol"}, inplace=True)
    betas.rename(columns={"week": "Date", "symbol": "Symbol"}, inplace=True)

    # Merge sector info if not present in betas
    def get_sector_map():
        return dict(zip(sector_map['Symbol'], sector_map['Sector']))

    if 'Sector' not in betas.columns:
        sector_dict = get_sector_map()
        betas['Sector'] = betas['Symbol'].map(sector_dict)

    # Merge holdings and betas on Date and Symbol
    merged = holdings.merge(betas, on=['Date', 'Symbol'], how='inner')

    # Identify all factor columns (1-88 as strings)
    factor_cols = [str(i) for i in range(1, 89)]

    # Group by Date and Sector, calculate weighted average beta and sector total weight
    def sector_weighted_avg(group):
        weights = group['Weight']
        total_weight = weights.sum()
        result = {'Sector_Weight': total_weight}
        if total_weight == 0:
            for col in factor_cols:
                result[col] = np.nan
            return pd.Series(result)
        for col in factor_cols:
            if col in group:
                betas = group[col]
                result[col] = (betas * weights).sum() / total_weight
            else:
                result[col] = np.nan
        return pd.Series(result)

    # Only operate on factor columns and Weight to avoid FutureWarning
    sector_weekly = (
        merged.groupby(['Date', 'Sector'])
        [factor_cols + ['Weight']]
        .apply(sector_weighted_avg)
        .reset_index()
    )

    # Normalize sector weights for each week
    sector_weekly['Sector_Weight'] = sector_weekly.groupby('Date')['Sector_Weight'].transform(lambda x: x / x.sum())

    # Sort for readability
    sector_weekly = sector_weekly.sort_values(['Date', 'Sector'])

    # Save Date, Sector, Sector_Weight, and all 88 factor columns
    cols_to_save = ['Date', 'Sector', 'Sector_Weight'] + factor_cols
    sector_weekly[cols_to_save].to_csv(OUTPUT_PATH, index=False)

    print(f"\u2713 Saved sector weighted averages for all 88 betas to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

