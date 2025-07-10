import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import re

def parse_date_from_filename(filename):
    """
    Parse date from filename like 'master_factor_betas_01_01_2024.csv'
    Returns datetime object or None if parsing fails
    """
    try:
        # Remove extension and prefix
        date_part = filename.replace('master_factor_betas_', '').replace('.csv', '')
        
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

def consolidate_transformed_betas():
    """
    Consolidate all transformed beta files into a single parquet file
    """
    # Define paths
    project_root = Path(__file__).parent.parent
    transformed_folder = project_root / "raw_data" / "master_factor_betas_transformed"
    new_transformed_folder = project_root / "raw_data" / "master_factor_betas_new_transformed"
    raw_folder = project_root / "raw_data" / "Master_Factor_Betas_Raw"
    output_file = project_root / "data_warehouse" / "consolidated_factor_betas.parquet"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all CSV files from both folders
    all_files = []
    
    # Add files from existing transformed folder
    if transformed_folder.exists():
        transformed_files = list(transformed_folder.glob("*.csv"))
        all_files.extend([(f, "existing") for f in transformed_files])
        print(f"Found {len(transformed_files)} files in existing transformed folder")
    
    # Add files from new transformed folder
    if new_transformed_folder.exists():
        new_files = list(new_transformed_folder.glob("*.csv"))
        all_files.extend([(f, "new") for f in new_files])
        print(f"Found {len(new_files)} files in new transformed folder")
    
    print(f"Total files to process: {len(all_files)}")
    print()
    
    # Process each file and collect data
    consolidated_data = []
    processed_count = 0
    error_count = 0
    
    for file_path, source in all_files:
        try:
            # Parse date from filename
            date = parse_date_from_filename(file_path.name)
            if date is None:
                print(f"Skipping {file_path.name} - could not parse date")
                error_count += 1
                continue
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add date column
            df['Date'] = date
            
            # Reorder columns to have Date after Company Name
            cols = df.columns.tolist()
            # Move Date to position 2 (after Symbol and Company Name)
            cols.remove('Date')
            cols.insert(2, 'Date')
            df = df[cols]
            
            consolidated_data.append(df)
            processed_count += 1
            
            if processed_count % 10 == 0:
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
    
    # Sort by Symbol, then by Date
    print("Sorting data...")
    final_df = final_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # Data type optimization
    print("Optimizing data types...")
    
    # Convert Date to datetime
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    
    # Convert Symbol and Company Name to category (saves memory)
    final_df['Symbol'] = final_df['Symbol'].astype('category')
    final_df['Company Name'] = final_df['Company Name'].astype('category')
    
    # Convert numeric columns to float32 (saves memory while maintaining precision)
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        final_df[col] = final_df[col].astype('float32')
    
    # Display summary statistics
    print("\n" + "="*60)
    print("CONSOLIDATION SUMMARY")
    print("="*60)
    print(f"Total records: {len(final_df):,}")
    print(f"Unique symbols: {final_df['Symbol'].nunique():,}")
    print(f"Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"Total weeks: {final_df['Date'].nunique()}")
    print(f"Factor columns: {len(numeric_cols)}")
    print(f"Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Show sample of data
    print("\nSample data:")
    print(final_df.head())
    
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
    print("CONSOLIDATION COMPLETE!")
    print("="*60)
    
    return final_df

def analyze_consolidated_data():
    """
    Analyze the consolidated data for insights
    """
    project_root = Path(__file__).parent.parent
    parquet_file = project_root / "data_warehouse" / "consolidated_factor_betas.parquet"
    
    if not parquet_file.exists():
        print("Consolidated parquet file not found. Run consolidation first.")
        return
    
    print("Loading consolidated data for analysis...")
    df = pd.read_parquet(parquet_file)
    
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    # Time series analysis
    print("\nTime Series Coverage:")
    date_counts = df['Date'].value_counts().sort_index()
    print(f"Date range: {date_counts.index.min()} to {date_counts.index.max()}")
    print(f"Average stocks per week: {date_counts.mean():.0f}")
    print(f"Min stocks per week: {date_counts.min()}")
    print(f"Max stocks per week: {date_counts.max()}")
    
    # Symbol coverage
    print("\nSymbol Coverage:")
    symbol_counts = df['Symbol'].value_counts()
    print(f"Total unique symbols: {len(symbol_counts)}")
    print(f"Average weeks per symbol: {symbol_counts.mean():.1f}")
    print(f"Symbols with most data: {symbol_counts.head()}")
    
    # Factor statistics
    print("\nFactor Statistics:")
    factor_cols = [col for col in df.columns if col.isdigit()]
    if factor_cols:
        factor_stats = df[factor_cols].describe()
        print(f"Number of factors: {len(factor_cols)}")
        print(f"Overall value range: {df[factor_cols].min().min():.6f} to {df[factor_cols].max().max():.6f}")
        print(f"Mean absolute values: {df[factor_cols].abs().mean().mean():.6f}")

if __name__ == "__main__":
    # Run consolidation
    df = consolidate_transformed_betas()
    
    # Run analysis
    if df is not None:
        print("\n" + "="*60)
        analyze_consolidated_data()
