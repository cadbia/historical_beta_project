import pandas as pd
import numpy as np
import os
from pathlib import Path

def validate_beta_ranges():
    """
    Validates that all beta values (excluding first two metadata columns) 
    in both raw and transformed datasets are between -1.5 and +1.5.
    Prints the names of files that contain values outside these bounds.
    """
    
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_folder = project_root / "raw_data" / "master_factor_betas_raw"
    transformed_folder = project_root / "raw_data" / "master_factor_betas_transformed"
    
    # Track files with issues
    files_with_issues = []
    
    def check_file_ranges(file_path, dataset_name):
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
            
            if violations > 0:
                files_with_issues.append((file_path.name, dataset_name, violations))
                
                # Find min and max values for reporting
                min_val = numeric_data.min().min()
                max_val = numeric_data.max().max()
                
                print(f"❌ {dataset_name}: {file_path.name}")
                print(f"   Violations: {violations}")
                print(f"   Value range: {min_val:.6f} to {max_val:.6f}")
                print()
            else:
                print(f"✅ {dataset_name}: {file_path.name} - All values within range")
                
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {str(e)}")
            files_with_issues.append((file_path.name, dataset_name, f"Error: {str(e)}"))
    
    print("=" * 80)
    print("VALIDATING BETA VALUE RANGES [-1.5, +1.5]")
    print("=" * 80)
    print()
    
    # Check Raw Data Files
    print("🔍 Checking Master Factor Betas Raw files...")
    print("-" * 50)
    
    if raw_folder.exists():
        csv_files = list(raw_folder.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in raw data folder")
        print()
        
        for csv_file in sorted(csv_files):
            check_file_ranges(csv_file, "RAW")
    else:
        print(f"❌ Raw data folder not found: {raw_folder}")
    
    print()
    
    # Check Transformed Data Files
    print("🔍 Checking Master Factor Betas Transformed files...")
    print("-" * 50)
    
    if transformed_folder.exists():
        csv_files = list(transformed_folder.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in transformed data folder")
        print()
        
        for csv_file in sorted(csv_files):
            check_file_ranges(csv_file, "TRANSFORMED")
    else:
        print(f"❌ Transformed data folder not found: {transformed_folder}")
    
    # Summary Report
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    if files_with_issues:
        print(f"❌ Found {len(files_with_issues)} files with values outside [-1.5, +1.5]:")
        print()
        
        for filename, dataset, violations in files_with_issues:
            print(f"  • {dataset}: {filename}")
            if isinstance(violations, int):
                print(f"    Violations: {violations}")
            else:
                print(f"    Issue: {violations}")
        
        print()
        print("Files with range violations:")
        for filename, dataset, violations in files_with_issues:
            if isinstance(violations, int):
                print(f"  {filename}")
        
    else:
        print("✅ All files passed validation - all values are within [-1.5, +1.5] range!")
    
    print("=" * 80)
    print("Validation complete.")
    print("=" * 80)

if __name__ == "__main__":
    validate_beta_ranges()
