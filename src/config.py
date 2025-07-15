"""
Configuration management for the historical beta project.
This module provides a centralized way to configure different indices and their data sources.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class IndexConfig:
    """Configuration for a specific index"""
    name: str
    display_name: str
    holdings_folder: str
    holdings_file_pattern: str
    holdings_consolidated_file: str
    weighted_betas_file: str
    csv_export_file: str
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.name:
            raise ValueError("Index name cannot be empty")
        if not self.holdings_folder:
            raise ValueError("Holdings folder cannot be empty")

class ProjectConfig:
    """Main configuration manager for the historical beta project"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize project configuration
        
        Args:
            config_file: Path to custom configuration file (optional)
        """
        self.project_root = Path(__file__).parent.parent
        self.config_file = Path(config_file) if config_file else self.project_root / "config.json"
        
        # Default configurations for different indices
        self.default_configs = {
            "SP500": IndexConfig(
                name="SP500",
                display_name="S&P 500",
                holdings_folder="SP500_holdings",
                holdings_file_pattern="SP500_Holdings_*.csv",
                holdings_consolidated_file="SP500_holdings_consolidated.parquet",
                weighted_betas_file="SP500_weekly_factor_betas.parquet",
                csv_export_file="SP500_weekly_factor_betas.csv"
            ),
            "NASDAQ": IndexConfig(
                name="NASDAQ",
                display_name="NASDAQ",
                holdings_folder="NASDAQ_holdings",
                holdings_file_pattern="NASDAQ_Holdings_*.csv",
                holdings_consolidated_file="NASDAQ_holdings_consolidated.parquet",
                weighted_betas_file="NASDAQ_weekly_factor_betas.parquet",
                csv_export_file="NASDAQ_weekly_factor_betas.csv"
            ),
            "DOW": IndexConfig(
                name="DOW",
                display_name="Dow Jones",
                holdings_folder="DOW_holdings",
                holdings_file_pattern="DOW_Holdings_*.csv",
                holdings_consolidated_file="DOW_holdings_consolidated.parquet",
                weighted_betas_file="DOW_weekly_factor_betas.parquet",
                csv_export_file="DOW_weekly_factor_betas.csv"
            )
        }
        
        # Load custom configurations if they exist
        self.custom_configs = self._load_custom_configs()
        
        # Current active index
        self.current_index = "SP500"
    
    def _load_custom_configs(self) -> Dict[str, IndexConfig]:
        """Load custom configurations from JSON file"""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            custom_configs = {}
            for name, config_data in data.get('indices', {}).items():
                custom_configs[name] = IndexConfig(**config_data)
            
            return custom_configs
            
        except Exception as e:
            print(f"Warning: Could not load custom config file: {e}")
            return {}
    
    def get_index_config(self, index_name: str) -> IndexConfig:
        """
        Get configuration for a specific index
        
        Args:
            index_name: Name of the index (e.g., 'SP500', 'NASDAQ')
            
        Returns:
            IndexConfig object for the specified index
        """
        # Check custom configs first
        if index_name in self.custom_configs:
            return self.custom_configs[index_name]
        
        # Check default configs
        if index_name in self.default_configs:
            return self.default_configs[index_name]
        
        raise ValueError(f"Unknown index: {index_name}. Available indices: {self.get_available_indices()}")
    
    def get_available_indices(self) -> list:
        """Get list of all available indices"""
        return list(self.default_configs.keys()) + list(self.custom_configs.keys())
    
    def set_current_index(self, index_name: str):
        """Set the current active index"""
        if index_name not in self.get_available_indices():
            raise ValueError(f"Unknown index: {index_name}. Available indices: {self.get_available_indices()}")
        self.current_index = index_name
    
    def get_current_config(self) -> IndexConfig:
        """Get configuration for the current active index"""
        return self.get_index_config(self.current_index)
    
    def get_paths(self, index_name: Optional[str] = None) -> Dict[str, Path]:
        """
        Get all relevant paths for an index
        
        Args:
            index_name: Name of the index (uses current if not specified)
            
        Returns:
            Dictionary of paths for the index
        """
        config = self.get_index_config(index_name or self.current_index)
        
        return {
            'project_root': self.project_root,
            'raw_data': self.project_root / "raw_data",
            'data_warehouse': self.project_root / "data_warehouse",
            'data_marts': self.project_root / "data_marts",
            'holdings_folder': self.project_root / "raw_data" / config.holdings_folder,
            'holdings_consolidated': self.project_root / "data_warehouse" / config.holdings_consolidated_file,
            'betas_consolidated': self.project_root / "data_warehouse" / "consolidated_factor_betas.parquet",
            'weighted_betas': self.project_root / "data_marts" / config.weighted_betas_file,
            'csv_export': self.project_root / "data_marts" / config.csv_export_file,
            'transformed_betas': self.project_root / "raw_data" / "master_factor_betas_transformed"
        }
    
    def create_sample_config_file(self):
        """Create a sample configuration file"""
        sample_config = {
            "indices": {
                "CUSTOM_INDEX": {
                    "name": "CUSTOM_INDEX",
                    "display_name": "Custom Index",
                    "holdings_folder": "CUSTOM_INDEX_holdings",
                    "holdings_file_pattern": "CUSTOM_INDEX_Holdings_*.csv",
                    "holdings_consolidated_file": "CUSTOM_INDEX_holdings_consolidated.parquet",
                    "weighted_betas_file": "CUSTOM_INDEX_weekly_factor_betas.parquet",
                    "csv_export_file": "CUSTOM_INDEX_weekly_factor_betas.csv"
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"Sample configuration file created at: {self.config_file}")
    
    def validate_index_data(self, index_name: str) -> Dict[str, bool]:
        """
        Validate that required data files exist for an index
        
        Args:
            index_name: Name of the index to validate
            
        Returns:
            Dictionary with validation results
        """
        paths = self.get_paths(index_name)
        
        return {
            'holdings_folder_exists': paths['holdings_folder'].exists(),
            'holdings_files_exist': len(list(paths['holdings_folder'].glob("*.csv"))) > 0 if paths['holdings_folder'].exists() else False,
            'betas_consolidated_exists': paths['betas_consolidated'].exists(),
            'holdings_consolidated_exists': paths['holdings_consolidated'].exists(),
            'weighted_betas_exists': paths['weighted_betas'].exists()
        }

# Global configuration instance
config = ProjectConfig()

def get_config() -> ProjectConfig:
    """Get the global configuration instance"""
    return config

def set_index(index_name: str):
    """Set the global current index"""
    config.set_current_index(index_name)

def get_current_index() -> str:
    """Get the current active index name"""
    return config.current_index

def get_paths(index_name: Optional[str] = None) -> Dict[str, Path]:
    """Get paths for current or specified index"""
    return config.get_paths(index_name)
