import pandas as pd
import numpy as np
import os
from pathlib import Path

def validate_new_transformed_files():
    """
    Validates that all beta values in the new transformed files 
    are between -1.5 and +1.5.
    """
    
    # Define paths
    project_root = Path(__file__).parent.parent
    new_transformed_folder = project_root / "raw_data" / "master_factor_betas_new_transformed"
    
    # Track files with issues
    files_with_issues = []
    
    def check_file_ranges(file_path):
        """Check a single file for values outside the range [-1.5, 1.5]"""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Skip first two columns (Symbol, Company Name) and get numeric data
            numeric_data = df.iloc[:, 2:].select_dtypes(include=[np.number])
            
            # Check for values outside range [-1.5, 1.5]
            out_of_range = (numeric_data < -1.5) | (numeric_data > 1.5)
            
            # Count violations
            violations = out_of_range.sum().sum()
            
            # Find min and max values for reporting
            min_val = numeric_data.min().min()
            max_val = numeric_data.max().max()
            
            if violations > 0:
                files_with_issues.append((file_path.name, violations))
                print(f"❌ {file_path.name}")
                print(f"   Violations: {violations}")
                print(f"   Value range: {min_val:.6f} to {max_val:.6f}")
            else:
                print(f"✅ {file_path.name} - All values within range [{min_val:.6f}, {max_val:.6f}]")
                
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {str(e)}")
            files_with_issues.append((file_path.name, f"Error: {str(e)}"))
    
    print("=" * 80)
    print("VALIDATING NEW TRANSFORMED BETA FILES [-1.5, +1.5]")
    print("=" * 80)
    print()
    
    # Check New Transformed Data Files
    print("🔍 Checking New Transformed files...")
    print("-" * 50)
    
    if new_transformed_folder.exists():
        csv_files = list(new_transformed_folder.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in new transformed folder")
        print()
        
        for csv_file in sorted(csv_files):
            check_file_ranges(csv_file)
    else:
        print(f"❌ New transformed folder not found: {new_transformed_folder}")
    
    # Summary Report
    print()
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    if files_with_issues:
        print(f"❌ Found {len(files_with_issues)} files with issues:")
        print()
        
        for filename, issue in files_with_issues:
            print(f"  • {filename}: {issue}")
        
    else:
        print("✅ All files passed validation - all values are within [-1.5, +1.5] range!")
    
    print("=" * 80)

if __name__ == "__main__":
    validate_new_transformed_files()
