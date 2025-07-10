import pandas as pd
import numpy as np
import os
from pathlib import Path

def excel_percentrank_exc(sorted_arr, x):
    """
    Replicates Excel's PERCENTRANK.EXC behavior:
      - returns #N/A if x <= min or x >= max
      - otherwise finds k such that sorted_arr[k] <= x <= sorted_arr[k+1]
        and interpolates:
          P = (k + (x - sorted_arr[k]) / (sorted_arr[k+1] - sorted_arr[k])) 
              / (n + 1)
    """
    n = len(sorted_arr)
    if n < 2:
        return np.nan
    # Exclusive: values strictly beyond endpoints are invalid
    if x < sorted_arr[0] or x > sorted_arr[-1]:
        return np.nan
    
    # Handle exact boundary cases
    if x == sorted_arr[0]:
        return 1 / (n + 1)
    if x == sorted_arr[-1]:
        return n / (n + 1)

    # find insertion index so that sorted_arr[idx-1] < x <= sorted_arr[idx]
    idx = np.searchsorted(sorted_arr, x, side="right")
    k = idx - 1
    xk, xk1 = sorted_arr[k], sorted_arr[k+1]
    # interpolate between positions k and k+1
    fraction = (x - xk) / (xk1 - xk)
    # divide by (n+1) for exclusive rank
    return (k + fraction) / (n + 1)

def transform_single_file(input_file_path, output_file_path):
    """
    Transform a single CSV file using the same logic as transform_betas.py
    """
    print(f"Processing: {input_file_path.name}")
    
    try:
        # 1) Load data from CSV
        df = pd.read_csv(input_file_path)
        meta = df.iloc[:, :2]            # ticker & name (first two columns)
        betas = df.iloc[:, 2:].astype(float)  # all remaining columns as numeric

        # 2) Standardize each column (ddof=1)
        z = betas.copy()
        for col in z.columns:
            vals = betas[col].dropna()
            if len(vals) > 0:
                mu = vals.mean()
                sigma = vals.std(ddof=1)
                if sigma > 0:  # Avoid division by zero
                    z[col] = (betas[col] - mu) / sigma
                else:
                    z[col] = 0  # If no variation, set to 0
            else:
                z[col] = np.nan

        # 3) Flatten & sort all z-scores
        flat = z.values.flatten()
        flat = flat[~np.isnan(flat)]
        sorted_flat = np.sort(flat)

        # 4) Compute EXCLUSIVE percentiles and transform
        def transform_exc(val):
            if np.isnan(val):
                return np.nan
            p_exc = excel_percentrank_exc(sorted_flat, val) * 100.0   # 0–100 scale
            return (p_exc - 50.5) / 34.0

        # Apply transformation
        transformed = z.map(transform_exc)

        # 5) Combine metadata with transformed data and save to CSV
        output_df = pd.concat([meta, transformed], axis=1)
        output_df.to_csv(output_file_path, index=False)

        # 6) Validation
        transformed_values = transformed.values.flatten()
        transformed_values = transformed_values[~np.isnan(transformed_values)]
        
        min_val = np.min(transformed_values)
        max_val = np.max(transformed_values)
        out_of_range = np.sum((transformed_values < -1.5) | (transformed_values > 1.5))
        
        print(f"  ✅ Transformed successfully")
        print(f"  📊 Value range: {min_val:.6f} to {max_val:.6f}")
        print(f"  🔍 Out of range values: {out_of_range}")
        
        if out_of_range > 0:
            print(f"  ⚠️  WARNING: {out_of_range} values still outside [-1.5, +1.5]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {input_file_path.name}: {str(e)}")
        return False

def batch_transform_raw_betas():
    """
    Transform all raw beta files that failed validation
    """
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_folder = project_root / "raw_data" / "master_factor_betas_raw"
    output_folder = project_root / "raw_data" / "master_factor_betas_new_transformed"
    
    # List of files that failed validation (all raw files based on previous validation)
    failed_files = [
       
    ]
    
    print("=" * 80)
    print("BATCH TRANSFORMATION OF RAW BETA FILES")
    print("=" * 80)
    print(f"Input folder: {raw_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Files to process: {len(failed_files)}")
    print()
    
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    successful = 0
    failed = 0
    
    for filename in failed_files:
        input_file = raw_folder / filename
        output_file = output_folder / filename
        
        if input_file.exists():
            if transform_single_file(input_file, output_file):
                successful += 1
            else:
                failed += 1
        else:
            print(f"⚠️  File not found: {filename}")
            failed += 1
        
        print()  # Empty line for readability
    
    # Summary
    print("=" * 80)
    print("BATCH TRANSFORMATION SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Failed to process: {failed}")
    print(f"📁 Output files saved to: {output_folder}")
    
    if successful > 0:
        print("\n🎉 Transformation completed! Run validation script to verify results.")
    
    print("=" * 80)

if __name__ == "__main__":
    batch_transform_raw_betas()
