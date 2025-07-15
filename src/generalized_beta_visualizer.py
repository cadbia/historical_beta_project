"""
Generalized beta visualizer that works with any index.
This script replaces the hardcoded SP500 beta visualizer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import warnings
import sys
import argparse

warnings.filterwarnings('ignore')

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import get_config, set_index

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

FACTOR_GROUPS = {
    '1-7 value': list(map(str, range(1, 8))),
    '8 growth': ['8'],
    '9-23 volatility': list(map(str, range(9, 24))),
    '24-28 growth': list(map(str, range(24, 29))),
    '29 value': ['29'],
    '30-31 commodities': ['30', '31'],
    '32-33 fixed income': ['32', '33'],
    '34-41 fixed income': list(map(str, range(34, 42))),
    '42 volatility': ['42'],
    '43-53 index': list(map(str, range(43, 54))),
    '54-56 fixed income': ['54', '55', '56'],
    '57-63 macro': list(map(str, range(57, 64))),
    '64-68 value': list(map(str, range(64, 69))),
    '70-72 volatility': ['70', '71', '72'],
    '73-78 commodities': list(map(str, range(73, 79))),
    '79-88 sector': list(map(str, range(79, 89)))
}

class GeneralizedBetaVisualizer:
    """
    A generalized class for visualizing factor beta trends for any index and individual stocks
    """
    
    def __init__(self, index_name: str = "SP500"):
        """
        Initialize the visualizer with data paths for a specific index
        
        Args:
            index_name: Name of the index (e.g., 'SP500', 'NASDAQ')
        """
        # Get configuration for this index
        self.config = get_config()
        set_index(index_name)
        
        self.index_name = index_name
        self.index_config = self.config.get_index_config(index_name)
        self.paths = self.config.get_paths(index_name)
        
        # Data containers
        self.index_data = None
        self.individual_data = None
        self.factor_cols = None
        
        self._load_data()
    
    def _load_data(self):
        """Load the data files"""
        try:
            print(f"Loading {self.index_config.display_name} factor betas...")
            if not self.paths['weighted_betas'].exists():
                print(f"Error: Weighted betas file not found at {self.paths['weighted_betas']}")
                print(f"Please run calculate_weighted_betas.py {self.index_name} first")
                return
            
            self.index_data = pd.read_parquet(self.paths['weighted_betas'])
            self.index_data['Date'] = pd.to_datetime(self.index_data['Date'])
            
            print("Loading individual stock factor betas...")
            if not self.paths['betas_consolidated'].exists():
                print(f"Error: Individual betas file not found at {self.paths['betas_consolidated']}")
                print("Please run consolidate_betas.py first")
                return
            
            self.individual_data = pd.read_parquet(self.paths['betas_consolidated'])
            self.individual_data['Date'] = pd.to_datetime(self.individual_data['Date'])
            
            # Get factor columns
            self.factor_cols = [col for col in self.index_data.columns if col.isdigit()]
            print(f"Loaded data with {len(self.factor_cols)} factors")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure all required data files are generated first")
        except Exception as e:
            print(f"Unexpected error loading data: {e}")
    
    def plot_factor_evolution(self, start_date=None, end_date=None, figsize=(15, 10), interactive=True):
        """
        Plot the evolution of grouped factor betas over time for the index
        Each line is the average of a factor group.
        """
        if self.index_data is None or self.factor_cols is None:
            print("No index data loaded")
            return
        
        # Filter data by date range
        data = self.index_data.copy()
        if start_date:
            data = data[data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['Date'] <= pd.to_datetime(end_date)]
        
        # Calculate group averages
        group_averages = {}
        for group_name, factors in FACTOR_GROUPS.items():
            available_factors = [f for f in factors if f in self.factor_cols]
            if available_factors:
                group_averages[group_name] = data[available_factors].mean(axis=1)
        
        # Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        for group_name, values in group_averages.items():
            plt.plot(data['Date'], values, label=group_name, linewidth=2)
        plt.title(f'{self.index_config.display_name} Factor Group Evolution')
        plt.xlabel('Date')
        plt.ylabel('Average Beta Value')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def search_ticker_betas(self, ticker, factors=None, start_date=None, end_date=None):
        """
        Search for and plot factor betas for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            factors: List of factor numbers to plot
            start_date: Start date for plotting
            end_date: End date for plotting
        """
        if self.individual_data is None or self.factor_cols is None:
            print("No individual stock data loaded")
            return
        
        # Filter data for the ticker
        ticker_data = self.individual_data[self.individual_data['Symbol'] == ticker.upper()].copy()
        
        if len(ticker_data) == 0:
            print(f"No data found for ticker: {ticker}")
            print("Available tickers (sample):")
            available_tickers = self.individual_data['Symbol'].unique()[:20]
            for t in available_tickers:
                print(f"  {t}")
            return
        
        # Filter by date range
        if start_date:
            ticker_data = ticker_data[ticker_data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            ticker_data = ticker_data[ticker_data['Date'] <= pd.to_datetime(end_date)]
        
        # Select factors
        if factors is None:
            factors = self.factor_cols[:6]  # First 6 factors for individual stocks
        else:
            factors = [str(f) for f in factors]
        
        factors = [f for f in factors if f in self.factor_cols and f in ticker_data.columns]
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Factor {f}' for f in factors[:6]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, factor in enumerate(factors[:6]):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Individual stock data
            fig.add_trace(
                go.Scatter(
                    x=ticker_data['Date'],
                    y=ticker_data[factor],
                    mode='lines',
                    name=f'{ticker} - Factor {factor}',
                    line=dict(color=colors[0], width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add index average if available
            if self.index_data is not None and factor in self.index_data.columns:
                # Filter index data to match ticker data date range
                index_filtered = self.index_data[
                    (self.index_data['Date'] >= ticker_data['Date'].min()) &
                    (self.index_data['Date'] <= ticker_data['Date'].max())
                ]
                
                fig.add_trace(
                    go.Scatter(
                        x=index_filtered['Date'],
                        y=index_filtered[factor],
                        mode='lines',
                        name=f'{self.index_config.display_name} - Factor {factor}',
                        line=dict(color=colors[1], width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f'{ticker} Factor Betas vs {self.index_config.display_name}',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Beta Value")
        
        fig.show()
        
        # Show summary statistics
        print(f"\nSummary for {ticker}:")
        print(f"Date range: {ticker_data['Date'].min().strftime('%Y-%m-%d')} to {ticker_data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Number of observations: {len(ticker_data)}")
        
        print(f"\nFactor statistics:")
        for factor in factors[:6]:
            if factor in ticker_data.columns:
                values = ticker_data[factor].dropna()
                if len(values) > 0:
                    print(f"  Factor {factor}: mean={values.mean():.4f}, std={values.std():.4f}, range=[{values.min():.4f}, {values.max():.4f}]")
    
    def export_to_csv(self, filename=None):
        """
        Export index weighted betas to CSV
        
        Args:
            filename: Custom filename (optional)
        """
        if self.index_data is None:
            print("No index data loaded")
            return
        
        if filename is None:
            filename = self.paths['csv_export']
        else:
            filename = self.paths['data_marts'] / filename
        
        # Ensure directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Export data
        self.index_data.to_csv(filename, index=False)
        
        print(f"\n✓ {self.index_config.display_name} factor betas exported to: {filename}")
        print(f"Data shape: {self.index_data.shape}")
        print(f"Date range: {self.index_data['Date'].min().strftime('%Y-%m-%d')} to {self.index_data['Date'].max().strftime('%Y-%m-%d')}")
    
    def get_summary_statistics(self):
        """Get summary statistics for the index factor betas"""
        if self.index_data is None or self.factor_cols is None:
            print("No index data loaded")
            return
        
        print(f"\n=== {self.index_config.display_name} Factor Beta Summary ===")
        print(f"Date range: {self.index_data['Date'].min().strftime('%Y-%m-%d')} to {self.index_data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Number of observations: {len(self.index_data)}")
        print(f"Number of factors: {len(self.factor_cols)}")
        
        # Factor statistics
        print(f"\nFactor statistics:")
        for factor in self.factor_cols[:10]:  # Show first 10 factors
            if factor in self.index_data.columns:
                values = self.index_data[factor].dropna()
                if len(values) > 0:
                    print(f"  Factor {factor}: mean={values.mean():.4f}, std={values.std():.4f}, range=[{values.min():.4f}, {values.max():.4f}]")
        
        if len(self.factor_cols) > 10:
            print(f"  ... and {len(self.factor_cols) - 10} more factors")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Visualize factor betas for any index')
    parser.add_argument('index', help='Index name (e.g., SP500, NASDAQ, DOW)')
    parser.add_argument('--factors', nargs='*', type=int, help='Factor numbers to plot')
    parser.add_argument('--ticker', help='Specific ticker to analyze')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    parser.add_argument('--list-indices', action='store_true', help='List available indices')
    parser.add_argument('--no-interactive', action='store_true', help='Use matplotlib instead of plotly')
    
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
    
    # Create visualizer
    visualizer = GeneralizedBetaVisualizer(args.index)
    
    if args.summary:
        visualizer.get_summary_statistics()
        return
    
    if args.export:
        visualizer.export_to_csv()
        return
    
    if args.ticker:
        visualizer.search_ticker_betas(
            args.ticker,
            factors=args.factors,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        visualizer.plot_factor_evolution(
            start_date=args.start_date,
            end_date=args.end_date,
            interactive=not args.no_interactive
        )

if __name__ == "__main__":
    main()
