"""
Generalized weighted betas calculation script that works with any index.
This script replaces the hardcoded SP500 weighted betas calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import argparse

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import get_config, set_index

def calculate_weighted_betas(index_name: str):
    """
    Calculate weekly weighted average factor betas for any index
    by combining holdings weights with individual stock betas
    
    Args:
        index_name: Name of the index (e.g., 'SP500', 'NASDAQ')
    """
    # Get configuration for this index
    config = get_config()
    set_index(index_name)
    
    index_config = config.get_index_config(index_name)
    paths = config.get_paths(index_name)
    
    print(f"=== Calculating {index_config.display_name} Weighted Factor Betas ===")
    print(f"Index: {index_name}")
    print(f"Holdings file: {paths['holdings_consolidated']}")
    print(f"Betas file: {paths['betas_consolidated']}")
    print(f"Output file: {paths['weighted_betas']}")
    print()
    
    # Ensure output directory exists
    paths['weighted_betas'].parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {index_config.display_name} holdings data...")
    if not paths['holdings_consolidated'].exists():
        print(f"Error: Holdings file not found at {paths['holdings_consolidated']}")
        print(f"Please run consolidate_holdings.py {index_name} first")
        return None
    
    holdings = pd.read_parquet(paths['holdings_consolidated'])
    print(f"Loaded {len(holdings):,} holdings records")
    
    print("Loading factor betas data...")
    if not paths['betas_consolidated'].exists():
        print(f"Error: Betas file not found at {paths['betas_consolidated']}")
        print("Please run consolidate_betas.py first")
        return None
    
    betas = pd.read_parquet(paths['betas_consolidated'])
    print(f"Loaded {len(betas):,} beta records")
    
    # Identify factor columns (numeric columns that aren't Date)
    factor_cols = [col for col in betas.columns if col.isdigit()]
    print(f"Found {len(factor_cols)} factor columns: {factor_cols[:5]}...")
    
    # Show data ranges
    print(f"\nData ranges:")
    print(f"Holdings: {holdings['Date'].min()} to {holdings['Date'].max()}")
    print(f"Betas: {betas['Date'].min()} to {betas['Date'].max()}")
    
    # Join holdings with betas on Date and Symbol
    print("\nJoining holdings with betas...")
    merged = holdings.merge(
        betas, 
        on=['Date', 'Symbol'], 
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged):,} records")
    print(f"Coverage: {len(merged) / len(holdings) * 100:.1f}% of holdings have beta data")
    
    # Check for any missing critical data
    missing_betas = holdings[~holdings.set_index(['Date', 'Symbol']).index.isin(
        merged.set_index(['Date', 'Symbol']).index
    )]
    
    if len(missing_betas) > 0:
        print(f"\nWarning: {len(missing_betas)} holdings records missing beta data")
        print("Top missing symbols:")
        missing_symbols = missing_betas['Symbol'].value_counts().head(10)
        for symbol, count in missing_symbols.items():
            print(f"  {symbol}: {count} dates")
    
    # Calculate weighted betas for each date
    print(f"\nCalculating weighted average betas...")
    
    # Group by date and calculate weighted averages
    def calculate_weighted_avg(group):
        # Normalize weights to sum to 1 for each date
        total_weight = group['Weight'].sum()
        if total_weight == 0:
            return pd.Series({col: np.nan for col in factor_cols})
        
        normalized_weights = group['Weight'] / total_weight
        
        # Calculate weighted average for each factor
        result = {}
        for factor in factor_cols:
            if factor in group.columns:
                weighted_avg = (group[factor] * normalized_weights).sum()
                result[factor] = weighted_avg
            else:
                result[factor] = np.nan
        
        return pd.Series(result)
    
    # Apply weighted average calculation
    weighted_betas = merged.groupby('Date').apply(calculate_weighted_avg).reset_index()
    
    # Sort by date
    weighted_betas = weighted_betas.sort_values('Date')
    
    # Show results summary
    print(f"\nResults Summary:")
    print(f"Date range: {weighted_betas['Date'].min().strftime('%Y-%m-%d')} to {weighted_betas['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Number of weeks: {len(weighted_betas)}")
    print(f"Factors calculated: {len(factor_cols)}")
    
    # Show some sample statistics
    print(f"\nSample factor statistics (first 5 factors):")
    for factor in factor_cols[:5]:
        if factor in weighted_betas.columns:
            values = weighted_betas[factor].dropna()
            if len(values) > 0:
                print(f"  Factor {factor}: mean={values.mean():.4f}, std={values.std():.4f}, range=[{values.min():.4f}, {values.max():.4f}]")
    
    # Check data quality
    print(f"\nData Quality Check:")
    total_values = len(weighted_betas) * len(factor_cols)
    nan_count = weighted_betas[factor_cols].isna().sum().sum()
    print(f"Total values: {total_values:,}")
    print(f"NaN values: {nan_count:,} ({nan_count/total_values*100:.1f}%)")
    
    # Save results
    print(f"\nSaving to: {paths['weighted_betas']}")
    weighted_betas.to_parquet(paths['weighted_betas'], index=False)
    
    print(f"\n✓ {index_config.display_name} weighted factor betas calculation completed successfully!")
    
    # Show next steps
    print(f"\nNext steps:")
    print(f"1. Use the visualizer to plot {index_config.display_name} factor evolution")
    print(f"2. Export results to CSV for further analysis")
    
    return weighted_betas

def validate_data_availability(index_name: str):
    """
    Validate that required data files exist for the specified index
    
    Args:
        index_name: Name of the index to validate
    """
    config = get_config()
    validation = config.validate_index_data(index_name)
    
    print(f"Data validation for {index_name}:")
    print(f"  Holdings folder exists: {'✓' if validation['holdings_folder_exists'] else '✗'}")
    print(f"  Holdings files exist: {'✓' if validation['holdings_files_exist'] else '✗'}")
    print(f"  Betas consolidated exists: {'✓' if validation['betas_consolidated_exists'] else '✗'}")
    print(f"  Holdings consolidated exists: {'✓' if validation['holdings_consolidated_exists'] else '✗'}")
    
    return all([
        validation['holdings_folder_exists'],
        validation['holdings_files_exist'],
        validation['betas_consolidated_exists'],
        validation['holdings_consolidated_exists']
    ])

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Calculate weighted factor betas for any index')
    parser.add_argument('index', help='Index name (e.g., SP500, NASDAQ, DOW)')
    parser.add_argument('--list-indices', action='store_true', help='List available indices')
    parser.add_argument('--validate', action='store_true', help='Validate data availability only')
    
    args = parser.parse_args()
    
    if args.list_indices:
        config = get_config()
        indices = config.get_available_indices()
        print("Available indices:")
        for idx in indices:
            print(f"  - {idx}")
        return
    
    # Validate index
    config = get_config()
    if args.index not in config.get_available_indices():
        print(f"Error: Unknown index '{args.index}'")
        print(f"Available indices: {config.get_available_indices()}")
        return
    
    # Validate data availability
    if not validate_data_availability(args.index):
        print(f"\nError: Required data files are missing for {args.index}")
        print("Please ensure:")
        print(f"1. Holdings files exist in raw_data/{args.index}_holdings/")
        print(f"2. Run consolidate_holdings.py {args.index} first")
        print("3. Run consolidate_betas.py to create consolidated factor betas")
        return
    
    if args.validate:
        print("✓ All required data files are available")
        return
    
    # Run calculation
    result = calculate_weighted_betas(args.index)
    
    if result is not None:
        print(f"\n✓ Weighted betas calculation completed for {args.index}")

if __name__ == "__main__":
    main()
