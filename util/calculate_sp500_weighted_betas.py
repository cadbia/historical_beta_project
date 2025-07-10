import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def calculate_sp500_weighted_betas():
    """
    Calculate weekly weighted average factor betas for the S&P 500
    by combining holdings weights with individual stock betas
    """
    # Define paths
    project_root = Path(__file__).parent.parent
    holdings_file = project_root / "data_warehouse" / "SP500_holdings_consolidated.parquet"
    betas_file = project_root / "data_warehouse" / "consolidated_factor_betas.parquet"
    output_file = project_root / "data_marts" / "SP500_weekly_factor_betas.parquet"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading SP500 holdings data...")
    if not holdings_file.exists():
        print(f"Error: Holdings file not found at {holdings_file}")
        print("Please run consolidate_sp500_holdings.py first")
        return None
    
    holdings = pd.read_parquet(holdings_file)
    print(f"Loaded {len(holdings):,} holdings records")
    
    print("Loading factor betas data...")
    if not betas_file.exists():
        print(f"Error: Betas file not found at {betas_file}")
        print("Please run consolidate_betas.py first")
        return None
    
    betas = pd.read_parquet(betas_file)
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
        how='inner',
        suffixes=('_holdings', '_betas')
    )
    
    print(f"Merged dataset: {len(merged):,} records")
    print(f"Date range after merge: {merged['Date'].min()} to {merged['Date'].max()}")
    print(f"Unique dates: {merged['Date'].nunique()}")
    print(f"Unique symbols: {merged['Symbol'].nunique()}")
    
    # Exclude CASH_USD from beta calculations (it has no factor exposures)
    pre_cash_filter = len(merged)
    merged = merged[merged['Symbol'] != 'CASH_USD']
    print(f"Excluded CASH_USD: {pre_cash_filter - len(merged):,} records removed")
    
    # Calculate weighted average betas for each date
    print("\nCalculating weighted average betas...")
    
    def calculate_weighted_averages(group):
        """Calculate weighted average for each factor for a given date"""
        weights = group['Weight'].values
        total_weight = weights.sum()
        
        if total_weight == 0:
            # Return NaN if no weights
            return pd.Series({col: np.nan for col in factor_cols})
        
        # Calculate weighted average for each factor
        weighted_averages = {}
        for factor in factor_cols:
            factor_values = group[factor].values
            # Handle NaN values - exclude them from calculation
            valid_mask = ~np.isnan(factor_values)
            if valid_mask.sum() == 0:
                weighted_averages[factor] = np.nan
            else:
                valid_weights = weights[valid_mask]
                valid_values = factor_values[valid_mask]
                valid_weight_sum = valid_weights.sum()
                
                if valid_weight_sum > 0:
                    weighted_averages[factor] = (valid_values * valid_weights).sum() / valid_weight_sum
                else:
                    weighted_averages[factor] = np.nan
        
        return pd.Series(weighted_averages)
    
    # Group by date and calculate weighted averages
    sp500_betas = merged.groupby('Date').apply(calculate_weighted_averages, include_groups=False).reset_index()
    
    # Add metadata columns
    sp500_betas['Symbol'] = 'SPX'  # S&P 500 index symbol
    sp500_betas['Company Name'] = 'S&P 500 Index'
    
    # Reorder columns
    cols = ['Date', 'Symbol', 'Company Name'] + factor_cols
    sp500_betas = sp500_betas[cols]
    
    # Sort by date
    sp500_betas = sp500_betas.sort_values('Date').reset_index(drop=True)
    
    # Data type optimization
    print("Optimizing data types...")
    sp500_betas['Date'] = pd.to_datetime(sp500_betas['Date'])
    sp500_betas['Symbol'] = sp500_betas['Symbol'].astype('category')
    sp500_betas['Company Name'] = sp500_betas['Company Name'].astype('category')
    
    # Convert factor columns to float32
    for col in factor_cols:
        sp500_betas[col] = sp500_betas[col].astype('float32')
    
    # Display summary
    print("\n" + "="*70)
    print("S&P 500 WEIGHTED FACTOR BETAS SUMMARY")
    print("="*70)
    print(f"Total weeks: {len(sp500_betas):,}")
    print(f"Date range: {sp500_betas['Date'].min()} to {sp500_betas['Date'].max()}")
    print(f"Factor columns: {len(factor_cols)}")
    print(f"Memory usage: {sp500_betas.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Show sample data
    print("\nSample S&P 500 weighted betas:")
    sample_cols = ['Date', 'Symbol', 'Company Name'] + factor_cols[:5]
    print(sp500_betas[sample_cols].head())
    
    # Calculate summary statistics
    print("\nFactor beta statistics:")
    factor_stats = sp500_betas[factor_cols].describe()
    print(f"Value range: {sp500_betas[factor_cols].min().min():.6f} to {sp500_betas[factor_cols].max().max():.6f}")
    print(f"Mean absolute values: {sp500_betas[factor_cols].abs().mean().mean():.6f}")
    
    # Check for missing values
    missing_values = sp500_betas[factor_cols].isna().sum().sum()
    print(f"Missing values: {missing_values}")
    
    # Save to parquet
    print(f"\nSaving to: {output_file}")
    sp500_betas.to_parquet(output_file, compression='snappy', index=False)
    
    # Verify the saved file
    file_size = output_file.stat().st_size / 1024**2
    print(f"File saved successfully! Size: {file_size:.2f} MB")
    
    # Quick validation
    print("\nValidating saved file...")
    test_df = pd.read_parquet(output_file)
    print(f"Verification: {len(test_df):,} records loaded from parquet file")
    
    print("\n" + "="*70)
    print("S&P 500 WEIGHTED BETAS CALCULATION COMPLETE!")
    print("="*70)
    
    return sp500_betas

def analyze_sp500_betas():
    """
    Analyze the S&P 500 weighted beta time series
    """
    project_root = Path(__file__).parent.parent
    parquet_file = project_root / "data_marts" / "SP500_weekly_factor_betas.parquet"
    
    if not parquet_file.exists():
        print("S&P 500 betas file not found. Run calculation first.")
        return
    
    print("Loading S&P 500 weighted betas for analysis...")
    df = pd.read_parquet(parquet_file)
    
    print("\n" + "="*70)
    print("S&P 500 WEIGHTED BETAS ANALYSIS")
    print("="*70)
    
    # Time series properties
    print(f"\nTime Series Properties:")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total weeks: {len(df)}")
    print(f"Frequency: Weekly")
    
    # Factor analysis
    factor_cols = [col for col in df.columns if col.isdigit()]
    print(f"\nFactor Analysis:")
    print(f"Number of factors: {len(factor_cols)}")
    
    # Most volatile factors (highest standard deviation)
    factor_volatility = df[factor_cols].std().sort_values(ascending=False)
    print(f"\nMost volatile factors (top 10):")
    print(factor_volatility.head(10))
    
    # Most stable factors (lowest standard deviation)
    print(f"\nMost stable factors (top 10):")
    print(factor_volatility.tail(10))
    
    # Factor correlations (sample)
    print(f"\nFactor correlations (sample of first 10 factors):")
    sample_factors = factor_cols[:10]
    corr_matrix = df[sample_factors].corr()
    print(f"Average absolute correlation: {corr_matrix.abs().mean().mean():.4f}")
    
    # Time evolution of key factors
    print(f"\nTime evolution (latest vs earliest values):")
    latest = df.iloc[-1]
    earliest = df.iloc[0]
    
    print(f"Latest date: {latest['Date']}")
    print(f"Earliest date: {earliest['Date']}")
    
    # Show change in some factors
    for factor in factor_cols[:5]:
        change = latest[factor] - earliest[factor]
        print(f"Factor {factor}: {earliest[factor]:.6f} → {latest[factor]:.6f} (Δ{change:+.6f})")

if __name__ == "__main__":
    # Calculate S&P 500 weighted betas
    result = calculate_sp500_weighted_betas()
    
    # Run analysis
    if result is not None:
        print("\n" + "="*70)
        analyze_sp500_betas()
