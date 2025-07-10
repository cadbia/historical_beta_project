import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import re

def parse_date_from_sp500_filename(filename):
    """
    Parse date from filename like 'SP500_Holdings_01_01_2024.csv'
    Returns datetime object or None if parsing fails
    """
    try:
        # Remove extension and prefix
        date_part = filename.replace('SP500_Holdings_', '').replace('.csv', '')
        
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

def consolidate_sp500_holdings():
    """
    Consolidate all SP500 holdings files into a single parquet file
    """
    # Define paths
    project_root = Path(__file__).parent.parent
    sp500_folder = project_root / "raw_data" / "SP500_holdings"
    output_file = project_root / "data_warehouse" / "SP500_holdings_consolidated.parquet"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(sp500_folder.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in SP500 holdings folder")
    print()
    
    # Process each file and collect data
    consolidated_data = []
    processed_count = 0
    error_count = 0
    
    for file_path in sorted(csv_files):
        try:
            # Parse date from filename
            date = parse_date_from_sp500_filename(file_path.name)
            if date is None:
                print(f"Skipping {file_path.name} - could not parse date")
                error_count += 1
                continue
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Validate expected columns
            expected_cols = ['Symbol', 'Company Name', 'Weight']
            if not all(col in df.columns for col in expected_cols):
                print(f"Warning: {file_path.name} missing expected columns. Found: {df.columns.tolist()}")
                error_count += 1
                continue
            
            # Add date column at the beginning
            df['Date'] = date
            
            # Reorder columns: Date, Symbol, Company Name, Weight
            df = df[['Date', 'Symbol', 'Company Name', 'Weight']]
            
            consolidated_data.append(df)
            processed_count += 1
            
            if processed_count % 20 == 0:
                print(f"Processed {processed_count} files...")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            error_count += 1
    
    print(f"\nProcessed {processed_count} files successfully")
    print(f"Errors: {error_count}")
    
    if not consolidated_data:
        print("No data to consolidate!")
        return
    
    # Concatenate all dataframes
    print("\nConsolidating data...")
    final_df = pd.concat(consolidated_data, ignore_index=True)
    
    # Sort by Date first, then by Symbol (for time series analysis)
    print("Sorting data...")
    final_df = final_df.sort_values(['Date', 'Symbol']).reset_index(drop=True)
    
    # Data type optimization
    print("Optimizing data types...")
    
    # Convert Date to datetime
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    
    # Convert Symbol and Company Name to category (saves memory and efficient for joins)
    final_df['Symbol'] = final_df['Symbol'].astype('category')
    final_df['Company Name'] = final_df['Company Name'].astype('category')
    
    # Convert Weight to float32 (sufficient precision for weights)
    final_df['Weight'] = final_df['Weight'].astype('float32')
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SP500 HOLDINGS CONSOLIDATION SUMMARY")
    print("="*60)
    print(f"Total records: {len(final_df):,}")
    print(f"Unique symbols: {final_df['Symbol'].nunique():,}")
    print(f"Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"Total weeks: {final_df['Date'].nunique()}")
    print(f"Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for CASH_USD entries
    cash_entries = final_df[final_df['Symbol'] == 'CASH_USD']
    print(f"CASH_USD entries: {len(cash_entries):,}")
    
    # Show weight statistics
    print(f"\nWeight statistics:")
    print(f"  Min weight: {final_df['Weight'].min():.6f}")
    print(f"  Max weight: {final_df['Weight'].max():.6f}")
    print(f"  Mean weight: {final_df['Weight'].mean():.6f}")
    
    # Show sample of data
    print("\nSample data:")
    print(final_df.head(10))
    
    # Save to parquet
    print(f"\nSaving to: {output_file}")
    final_df.to_parquet(output_file, compression='snappy', index=False)
    
    # Verify the saved file
    file_size = output_file.stat().st_size / 1024**2
    print(f"File saved successfully! Size: {file_size:.2f} MB")
    
    # Quick validation
    print("\nValidating saved file...")
    test_df = pd.read_parquet(output_file)
    print(f"Verification: {len(test_df):,} records loaded from parquet file")
    
    print("\n" + "="*60)
    print("SP500 HOLDINGS CONSOLIDATION COMPLETE!")
    print("="*60)
    
    return final_df

def analyze_sp500_holdings():
    """
    Analyze the consolidated SP500 holdings data for insights
    """
    project_root = Path(__file__).parent.parent
    parquet_file = project_root / "data_warehouse" / "SP500_holdings_consolidated.parquet"
    
    if not parquet_file.exists():
        print("SP500 holdings parquet file not found. Run consolidation first.")
        return
    
    print("Loading SP500 holdings data for analysis...")
    df = pd.read_parquet(parquet_file)
    
    print("\n" + "="*60)
    print("SP500 HOLDINGS DATA ANALYSIS")
    print("="*60)
    
    # Time series analysis
    print("\nTime Series Coverage:")
    date_counts = df['Date'].value_counts().sort_index()
    print(f"Date range: {date_counts.index.min()} to {date_counts.index.max()}")
    print(f"Average holdings per week: {date_counts.mean():.0f}")
    print(f"Min holdings per week: {date_counts.min()}")
    print(f"Max holdings per week: {date_counts.max()}")
    
    # Symbol analysis
    print("\nSymbol Analysis:")
    symbol_counts = df['Symbol'].value_counts()
    print(f"Total unique symbols: {len(symbol_counts)}")
    print(f"Average weeks per symbol: {symbol_counts.mean():.1f}")
    
    # Top holdings by frequency
    print(f"\nMost frequently appearing symbols:")
    print(symbol_counts.head(10))
    
    # Weight distribution analysis
    print(f"\nWeight Analysis:")
    print(f"Total weight range per date:")
    weight_sums = df.groupby('Date')['Weight'].sum()
    print(f"  Min total weight: {weight_sums.min():.2f}%")
    print(f"  Max total weight: {weight_sums.max():.2f}%")
    print(f"  Mean total weight: {weight_sums.mean():.2f}%")
    
    # Top weighted stocks
    print(f"\nTop weighted stocks (average across all dates):")
    avg_weights = df.groupby('Symbol')['Weight'].mean().sort_values(ascending=False)
    print(avg_weights.head(10))
    
    # CASH_USD analysis
    cash_data = df[df['Symbol'] == 'CASH_USD']
    if len(cash_data) > 0:
        print(f"\nCASH_USD Analysis:")
        print(f"  Dates with cash: {len(cash_data)}")
        print(f"  Cash weight range: {cash_data['Weight'].min():.6f} to {cash_data['Weight'].max():.6f}")
        print(f"  Average cash weight: {cash_data['Weight'].mean():.6f}")

if __name__ == "__main__":
    # Run consolidation
    df = consolidate_sp500_holdings()
    
    # Run analysis
    if df is not None:
        print("\n" + "="*60)
        analyze_sp500_holdings()
