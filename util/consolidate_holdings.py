"""
Generalized holdings consolidation script that works with any index.
This script replaces the hardcoded SP500 consolidation script.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import re
import sys
import argparse

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import get_config, set_index

def parse_date_from_filename(filename, index_name):
    """
    Parse date from filename based on index pattern
    Returns datetime object or None if parsing fails
    """
    try:
        # Remove extension and prefix
        prefix = f"{index_name}_Holdings_"
        if not filename.startswith(prefix):
            print(f"Warning: Filename {filename} doesn't match expected pattern for {index_name}")
            return None
            
        date_part = filename.replace(prefix, '').replace('.csv', '')
        
        # Handle different date formats
        if date_part.count('_') == 2:
            # Format: MM_DD_YYYY
            month, day, year = date_part.split('_')
            return datetime(int(year), int(month), int(day))
        else:
            print(f"Warning: Could not parse date from filename: {filename}")
            return None
            
    except Exception as e:
        print(f"Error parsing date from {filename}: {str(e)}")
        return None

def consolidate_holdings(index_name: str):
    """
    Consolidate all holdings files for a given index into a single parquet file
    
    Args:
        index_name: Name of the index (e.g., 'SP500', 'NASDAQ')
    """
    # Get configuration for this index
    config = get_config()
    set_index(index_name)
    
    index_config = config.get_index_config(index_name)
    paths = config.get_paths(index_name)
    
    print(f"=== Consolidating {index_config.display_name} Holdings ===")
    print(f"Index: {index_name}")
    print(f"Holdings folder: {paths['holdings_folder']}")
    print(f"Output file: {paths['holdings_consolidated']}")
    print()
    
    # Check if holdings folder exists
    if not paths['holdings_folder'].exists():
        print(f"Error: Holdings folder not found at {paths['holdings_folder']}")
        print(f"Please create the folder and add holdings files matching pattern: {index_config.holdings_file_pattern}")
        return None
    
    # Ensure output directory exists
    paths['holdings_consolidated'].parent.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(paths['holdings_folder'].glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {index_config.display_name} holdings folder")
    
    if len(csv_files) == 0:
        print(f"No CSV files found. Expected pattern: {index_config.holdings_file_pattern}")
        return None
    
    print()
    
    # Process each file and collect data
    consolidated_data = []
    processed_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        try:
            # Parse date from filename
            date = parse_date_from_filename(csv_file.name, index_name)
            
            if date is None:
                print(f"Skipping file with unparseable date: {csv_file.name}")
                error_count += 1
                continue
            
            # Read the file
            df = pd.read_csv(csv_file)
            
            # Standardize column names (handle different possible column names)
            # Look for common variations of weight and symbol columns
            weight_cols = [col for col in df.columns if 'weight' in col.lower() or 'pct' in col.lower()]
            symbol_cols = [col for col in df.columns if 'symbol' in col.lower() or 'ticker' in col.lower()]
            
            if not weight_cols:
                print(f"Warning: No weight column found in {csv_file.name}. Columns: {list(df.columns)}")
                error_count += 1
                continue
                
            if not symbol_cols:
                print(f"Warning: No symbol column found in {csv_file.name}. Columns: {list(df.columns)}")
                error_count += 1
                continue
            
            # Use the first matching column
            weight_col = weight_cols[0]
            symbol_col = symbol_cols[0]
            
            # Rename columns to standard names
            df = df.rename(columns={weight_col: 'Weight', symbol_col: 'Symbol'})
            
            # Keep only required columns
            df = df[['Symbol', 'Weight']]
            
            # Add date column
            df['Date'] = date
            
            # Clean and validate data
            df = df.dropna(subset=['Symbol', 'Weight'])
            
            # Convert weight to numeric (handle percentage strings)
            if df['Weight'].dtype == 'object':
                # Remove % sign if present
                df['Weight'] = df['Weight'].astype(str).str.replace('%', '')
                df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
            
            # Drop rows with invalid weights
            df = df.dropna(subset=['Weight'])
            
            # If weights are in percentage form (>1), convert to decimal
            if df['Weight'].max() > 1:
                df['Weight'] = df['Weight'] / 100
            
            # Add to consolidated data
            consolidated_data.append(df)
            processed_count += 1
            
            print(f"Processed: {csv_file.name} - {len(df)} holdings on {date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            error_count += 1
            continue
    
    if not consolidated_data:
        print("No data to consolidate!")
        return None
    
    # Combine all data
    print(f"\nCombining data from {len(consolidated_data)} files...")
    final_df = pd.concat(consolidated_data, ignore_index=True)
    
    # Sort by date and symbol
    final_df = final_df.sort_values(['Date', 'Symbol'])
    
    # Summary statistics
    print(f"\nConsolidation Summary:")
    print(f"Total records: {len(final_df):,}")
    print(f"Date range: {final_df['Date'].min().strftime('%Y-%m-%d')} to {final_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Unique symbols: {final_df['Symbol'].nunique():,}")
    print(f"Unique dates: {final_df['Date'].nunique():,}")
    print(f"Files processed: {processed_count}")
    print(f"Files with errors: {error_count}")
    
    # Show top holdings by average weight
    print(f"\nTop 10 holdings by average weight:")
    top_holdings = final_df.groupby('Symbol')['Weight'].mean().sort_values(ascending=False).head(10)
    for symbol, avg_weight in top_holdings.items():
        print(f"  {symbol}: {avg_weight:.4f} ({avg_weight*100:.2f}%)")
    
    # Save to parquet
    print(f"\nSaving to: {paths['holdings_consolidated']}")
    final_df.to_parquet(paths['holdings_consolidated'], index=False)
    
    print(f"\n✓ {index_config.display_name} holdings consolidation completed successfully!")
    return final_df

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Consolidate holdings files for any index')
    parser.add_argument('index', help='Index name (e.g., SP500, NASDAQ, DOW)')
    parser.add_argument('--list-indices', action='store_true', help='List available indices')
    
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
    
    # Run consolidation
    result = consolidate_holdings(args.index)
    
    if result is not None:
        print("\nNext steps:")
        print(f"1. Run calculate_weighted_betas.py {args.index} to calculate weighted factor betas")
        print(f"2. Use the visualizer to analyze the {args.index} factor evolution")

if __name__ == "__main__":
    main()
