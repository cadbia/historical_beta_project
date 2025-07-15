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
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BetaVisualizer:
    """
    A class for visualizing factor beta trends for S&P 500 and individual stocks
    """
    
    def __init__(self):
        """Initialize the visualizer with data paths"""
        self.project_root = Path(__file__).parent.parent
        self.sp500_file = self.project_root / "data_marts" / "SP500_weekly_factor_betas.parquet"
        self.individual_file = self.project_root / "data_warehouse" / "consolidated_factor_betas.parquet"
        
        # Load data
        self.sp500_data = None
        self.individual_data = None
        self.factor_cols = None
        
        self._load_data()
    
    def _load_data(self):
        """Load the data files"""
        try:
            print("Loading S&P 500 factor betas...")
            self.sp500_data = pd.read_parquet(self.sp500_file)
            self.sp500_data['Date'] = pd.to_datetime(self.sp500_data['Date'])
            
            print("Loading individual stock factor betas...")
            self.individual_data = pd.read_parquet(self.individual_file)
            self.individual_data['Date'] = pd.to_datetime(self.individual_data['Date'])
            
            # Get factor columns
            self.factor_cols = [col for col in self.sp500_data.columns if col.isdigit()]
            print(f"Loaded data with {len(self.factor_cols)} factors")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find data files. Please run the consolidation scripts first.")
            print(f"Missing file: {e}")
            
    def plot_sp500_factor_evolution(self, factors=None, save_path=None):
        """
        Create a beautiful plot showing S&P 500 factor evolution over time
        
        Parameters:
        factors: list of factor numbers to plot (default: top 10 most volatile)
        save_path: path to save the plot (optional)
        """
        if self.sp500_data is None:
            print("No S&P 500 data available")
            return
        
        # Select factors to plot
        if factors is None:
            # Select top 10 most volatile factors
            factor_volatility = self.sp500_data[self.factor_cols].std().sort_values(ascending=False)
            factors = factor_volatility.head(10).index.tolist()
        
        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Time series of selected factors
        ax1 = axes[0]
        for i, factor in enumerate(factors):
            ax1.plot(self.sp500_data['Date'], self.sp500_data[factor], 
                    label=f'Factor {factor}', linewidth=2, alpha=0.8)
        
        ax1.set_title('S&P 500 Factor Beta Evolution Over Time\n(Top 10 Most Volatile Factors)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Factor Beta Value', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Heatmap of factor correlations
        ax2 = axes[1]
        factor_data = self.sp500_data[factors]
        corr_matrix = factor_data.corr()
        
        im = ax2.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(factors)))
        ax2.set_yticks(range(len(factors)))
        ax2.set_xticklabels([f'Factor {f}' for f in factors], rotation=45)
        ax2.set_yticklabels([f'Factor {f}' for f in factors])
        ax2.set_title('Factor Correlation Matrix (S&P 500)', fontsize=14, fontweight='bold', pad=15)
        
        # Add correlation values to heatmap
        for i in range(len(factors)):
            for j in range(len(factors)):
                text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Correlation', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print some statistics
        print("\nS&P 500 Factor Statistics:")
        print("="*50)
        for factor in factors:
            data = self.sp500_data[factor]
            print(f"Factor {factor:2}: Mean={data.mean():6.3f}, Std={data.std():6.3f}, "
                  f"Range=({data.min():6.3f}, {data.max():6.3f})")
    
    def plot_individual_stock_betas(self, ticker, factors=None, save_path=None):
        """
        Plot individual stock beta evolution over time
        
        Parameters:
        ticker: Stock symbol to analyze
        factors: list of factor numbers to plot (default: top 5 most volatile for this stock)
        save_path: path to save the plot (optional)
        """
        if self.individual_data is None:
            print("No individual stock data available")
            return
        
        # Filter data for the specific ticker
        stock_data = self.individual_data[self.individual_data['Symbol'] == ticker.upper()].copy()
        
        if stock_data.empty:
            print(f"No data found for ticker: {ticker}")
            available_tickers = self.individual_data['Symbol'].unique()[:20]
            print(f"Available tickers (sample): {list(available_tickers)}")
            return
        
        stock_data = stock_data.sort_values('Date')
        
        # Select factors to plot
        if factors is None:
            # Select top 5 most volatile factors for this stock
            factor_volatility = stock_data[self.factor_cols].std().sort_values(ascending=False)
            factors = factor_volatility.head(5).index.tolist()
        
        # Create the plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot 1: Time series of selected factors
        ax1 = axes[0]
        for factor in factors:
            ax1.plot(stock_data['Date'], stock_data[factor], 
                    label=f'Factor {factor}', linewidth=2, alpha=0.8, marker='o', markersize=3)
        
        company_name = stock_data['Company Name'].iloc[0] if not stock_data.empty else ticker
        ax1.set_title(f'{ticker} - {company_name}\nFactor Beta Evolution Over Time', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Factor Beta Value', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Distribution of factor values
        ax2 = axes[1]
        factor_data = stock_data[factors].melt(var_name='Factor', value_name='Beta_Value')
        
        # Create violin plot
        parts = ax2.violinplot([stock_data[factor].dropna() for factor in factors], 
                              positions=range(len(factors)), showmeans=True)
        
        ax2.set_xticks(range(len(factors)))
        ax2.set_xticklabels([f'Factor {f}' for f in factors])
        ax2.set_title(f'{ticker} - Factor Beta Distributions', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Beta Value', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison with S&P 500 (for selected factors)
        ax3 = axes[2]
        
        # Get S&P 500 data for comparison
        sp500_factor = factors[0]  # Use first factor for comparison
        
        # Merge stock and S&P 500 data on date
        comparison_data = stock_data[['Date', sp500_factor]].merge(
            self.sp500_data[['Date', sp500_factor]], 
            on='Date', suffixes=('_stock', '_sp500'), how='inner'
        )
        
        if not comparison_data.empty:
            ax3.plot(comparison_data['Date'], comparison_data[f'{sp500_factor}_stock'], 
                    label=f'{ticker}', linewidth=2, alpha=0.8)
            ax3.plot(comparison_data['Date'], comparison_data[f'{sp500_factor}_sp500'], 
                    label='S&P 500', linewidth=2, alpha=0.8, linestyle='--')
            
            ax3.set_title(f'{ticker} vs S&P 500 - Factor {sp500_factor} Comparison', 
                         fontsize=14, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=12)
            ax3.set_ylabel(f'Factor {sp500_factor} Beta Value', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print statistics
        print(f"\n{ticker} Factor Statistics:")
        print("="*50)
        print(f"Data range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
        print(f"Total observations: {len(stock_data)}")
        
        for factor in factors:
            data = stock_data[factor].dropna()
            if len(data) > 0:
                print(f"Factor {factor:2}: Mean={data.mean():6.3f}, Std={data.std():6.3f}, "
                      f"Range=({data.min():6.3f}, {data.max():6.3f})")
    
    def create_interactive_dashboard(self, save_path=None):
        """
        Create an interactive Plotly dashboard for exploring factor betas
        """
        if self.sp500_data is None:
            print("No data available for dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('S&P 500 Factor Evolution', 'Factor Volatility Ranking',
                          'Recent Factor Values', 'Factor Correlation Network'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Time series of top 5 factors
        factor_volatility = self.sp500_data[self.factor_cols].std().sort_values(ascending=False)
        top_factors = factor_volatility.head(5).index.tolist()
        
        colors = px.colors.qualitative.Set1
        for i, factor in enumerate(top_factors):
            fig.add_trace(
                go.Scatter(x=self.sp500_data['Date'], y=self.sp500_data[factor],
                          mode='lines', name=f'Factor {factor}',
                          line=dict(color=colors[i % len(colors)], width=2)),
                row=1, col=1
            )
        
        # Plot 2: Factor volatility ranking
        fig.add_trace(
            go.Bar(x=[f'F{f}' for f in factor_volatility.head(10).index],
                   y=factor_volatility.head(10).values,
                   name='Volatility', showlegend=False,
                   marker_color='lightblue'),
            row=1, col=2
        )
        
        # Plot 3: Recent factor values (latest date)
        latest_data = self.sp500_data.iloc[-1]
        recent_factors = [f for f in self.factor_cols if abs(latest_data[f]) > 0.5][:10]
        
        fig.add_trace(
            go.Bar(x=[f'F{f}' for f in recent_factors],
                   y=[latest_data[f] for f in recent_factors],
                   name='Recent Values', showlegend=False,
                   marker_color=['red' if v < 0 else 'green' for v in [latest_data[f] for f in recent_factors]]),
            row=2, col=1
        )
        
        # Plot 4: Simplified correlation heatmap
        sample_factors = factor_volatility.head(8).index.tolist()
        corr_data = self.sp500_data[sample_factors].corr()
        
        fig.add_trace(
            go.Heatmap(z=corr_data.values,
                      x=[f'F{f}' for f in sample_factors],
                      y=[f'F{f}' for f in sample_factors],
                      colorscale='RdBu_r',
                      zmid=0,
                      showscale=True),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="S&P 500 Factor Beta Analysis Dashboard",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Beta Value", row=1, col=1)
        fig.update_xaxes(title_text="Factor", row=1, col=2)
        fig.update_yaxes(title_text="Volatility (Std Dev)", row=1, col=2)
        fig.update_xaxes(title_text="Factor", row=2, col=1)
        fig.update_yaxes(title_text="Beta Value", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to: {save_path}")
        
        fig.show()
    
    def export_sp500_betas_to_csv(self, save_path=None):
        """
        Export S&P 500 weekly weighted average factor betas to CSV file
        
        Parameters:
        save_path: path to save the CSV file (optional, defaults to data_marts folder)
        """
        if self.sp500_data is None:
            print("No S&P 500 data available to export")
            return
        
        # Set default save path if not provided
        if save_path is None:
            save_path = self.project_root / "data_marts" / "SP500_weekly_factor_betas.csv"
        
        try:
            # Create a copy of the data for export
            export_data = self.sp500_data.copy()
            
            # Sort by date for better readability
            export_data = export_data.sort_values('Date')
            
            # Format date as string for better CSV compatibility
            export_data['Date'] = export_data['Date'].dt.strftime('%Y-%m-%d')
            
            # Save to CSV
            export_data.to_csv(save_path, index=False)
            
            print(f"✅ S&P 500 weekly factor betas exported to: {save_path}")
            print(f"📊 Data summary:")
            print(f"   - Date range: {export_data['Date'].min()} to {export_data['Date'].max()}")
            print(f"   - Total weeks: {len(export_data)}")
            print(f"   - Total factors: {len(self.factor_cols)}")
            
            # Show sample of the data
            print(f"\n📋 Sample of exported data:")
            print(export_data.head(3).to_string(index=False))
            
            return save_path
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return None

    def export_sp500_weekly_betas_csv(self, save_path=None):
        """
        Export S&P 500 weekly weighted average factor betas to CSV
        
        Parameters:
        save_path: path to save the CSV file (optional, defaults to data_marts folder)
        
        Returns:
        str: path where the file was saved
        """
        if self.sp500_data is None:
            print("No S&P 500 data available for export")
            return None
        
        # Set default save path if not provided
        if save_path is None:
            save_path = self.project_root / "data_marts" / "SP500_weekly_factor_betas.csv"
        
        try:
            # Export the data
            self.sp500_data.to_csv(save_path, index=False)
            
            print(f"✅ S&P 500 weekly weighted average factor betas exported successfully!")
            print(f"📁 File saved to: {save_path}")
            print(f"📊 Data shape: {self.sp500_data.shape[0]} weeks × {self.sp500_data.shape[1]} columns")
            print(f"📅 Date range: {self.sp500_data['Date'].min()} to {self.sp500_data['Date'].max()}")
            print(f"🏭 Factors included: {len(self.factor_cols) if self.factor_cols else 0} factor columns")
            
            return str(save_path)
            
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
            return None

    def search_ticker_info(self, ticker=None):
        """
        Search for available ticker information
        """
        if self.individual_data is None:
            print("No individual stock data available")
            return
        
        if ticker is None:
            # Show summary of available data
            print("Available Ticker Information:")
            print("="*50)
            
            ticker_counts = self.individual_data['Symbol'].value_counts()
            print(f"Total unique tickers: {len(ticker_counts)}")
            print(f"Date range: {self.individual_data['Date'].min()} to {self.individual_data['Date'].max()}")
            
            print(f"\nTop 20 tickers by data availability:")
            for ticker, count in ticker_counts.head(20).items():
                sample_data = self.individual_data[self.individual_data['Symbol'] == ticker].iloc[0]
                company_name = sample_data['Company Name']
                print(f"{ticker:6} - {company_name:40} ({count:3} weeks)")
            
            print(f"\nTo analyze a specific ticker, use: plot_individual_stock_betas('TICKER')")
            
        else:
            # Show specific ticker info
            ticker = ticker.upper()
            ticker_data = self.individual_data[self.individual_data['Symbol'] == ticker]
            
            if ticker_data.empty:
                print(f"Ticker '{ticker}' not found in database")
                # Show similar tickers
                all_tickers = self.individual_data['Symbol'].unique()
                similar = [t for t in all_tickers if ticker in t or t in ticker]
                if similar:
                    print(f"Similar tickers found: {similar[:10]}")
            else:
                print(f"Ticker Information: {ticker}")
                print("="*50)
                company_name = ticker_data['Company Name'].iloc[0]
                print(f"Company: {company_name}")
                print(f"Data points: {len(ticker_data)}")
                print(f"Date range: {ticker_data['Date'].min()} to {ticker_data['Date'].max()}")
                
                # Show factor summary
                factor_data = ticker_data[self.factor_cols].mean()
                top_factors = factor_data.abs().sort_values(ascending=False).head(5)
                print(f"\nTop 5 Factor Exposures (by absolute value):")
                for factor, value in top_factors.items():
                    print(f"  Factor {factor}: {value:6.3f}")

def main():
    """Main function to demonstrate the visualizer"""
    print("🎨 Factor Beta Visualization Tool")
    print("="*50)
    
    # Initialize visualizer
    viz = BetaVisualizer()
    
    # Create S&P 500 evolution plot
    print("\n📊 Creating S&P 500 factor evolution plot...")
    viz.plot_sp500_factor_evolution()
    
    # Show ticker search functionality
    print("\n🔍 Ticker search functionality:")
    viz.search_ticker_info()
    
    # Export S&P 500 weekly betas to CSV
    print("\n📁 Exporting S&P 500 weekly betas to CSV...")
    csv_path = viz.export_sp500_weekly_betas_csv()
    if csv_path:
        print(f"✅ CSV exported successfully to: {csv_path}")
    
    # Example: Plot individual stock (if AAPL is available)
    print("\n📈 Example: Analyzing AAPL...")
    viz.plot_individual_stock_betas('AAPL')
    
    # Create interactive dashboard
    print("\n🚀 Creating interactive dashboard...")
    dashboard_path = viz.project_root / "src" / "factor_beta_dashboard.html"
    viz.create_interactive_dashboard(save_path=dashboard_path)

if __name__ == "__main__":
    main()
