"""
Unified workflow runner for historical beta analysis.
This script orchestrates the entire workflow for any index.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import os

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import get_config, set_index

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def validate_requirements():
    """Validate that all required files and directories exist"""
    project_root = Path(__file__).parent.parent
    
    # Check for required scripts
    required_scripts = [
        "util/validate_beta_ranges.py",
        "util/batch_transform_raw_betas.py",
        "util/consolidate_betas.py",
        "util/consolidate_holdings.py",
        "util/calculate_weighted_betas.py",
        "src/generalized_beta_visualizer.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not (project_root / script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print("Error: Missing required scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        return False
    
    # Check for required directories
    required_dirs = [
        "raw_data",
        "data_warehouse",
        "data_marts"
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    return True

def validate_raw_data():
    """Step 1: Validate raw data ranges"""
    print("\n" + "="*60)
    print("STEP 1: VALIDATING RAW DATA")
    print("="*60)
    
    python_exe = sys.executable
    script_path = Path(__file__).parent / "validate_beta_ranges.py"
    
    return run_command([python_exe, str(script_path)], "Validate raw data ranges")

def transform_raw_data():
    """Step 2: Transform raw data if needed"""
    print("\n" + "="*60)
    print("STEP 2: TRANSFORMING RAW DATA")
    print("="*60)
    
    python_exe = sys.executable
    script_path = Path(__file__).parent / "batch_transform_raw_betas.py"
    
    return run_command([python_exe, str(script_path)], "Transform raw data")

def consolidate_betas():
    """Step 3: Consolidate factor betas"""
    print("\n" + "="*60)
    print("STEP 3: CONSOLIDATING FACTOR BETAS")
    print("="*60)
    
    python_exe = sys.executable
    script_path = Path(__file__).parent / "consolidate_betas.py"
    
    return run_command([python_exe, str(script_path)], "Consolidate factor betas")

def consolidate_holdings(index_name):
    """Step 4: Consolidate holdings for specific index"""
    print("\n" + "="*60)
    print(f"STEP 4: CONSOLIDATING {index_name} HOLDINGS")
    print("="*60)
    
    python_exe = sys.executable
    script_path = Path(__file__).parent / "consolidate_holdings.py"
    
    return run_command([python_exe, str(script_path), index_name], f"Consolidate {index_name} holdings")

def calculate_weighted_betas(index_name):
    """Step 5: Calculate weighted betas for specific index"""
    print("\n" + "="*60)
    print(f"STEP 5: CALCULATING {index_name} WEIGHTED BETAS")
    print("="*60)
    
    python_exe = sys.executable
    script_path = Path(__file__).parent / "calculate_weighted_betas.py"
    
    return run_command([python_exe, str(script_path), index_name], f"Calculate {index_name} weighted betas")

def run_visualizer(index_name, export_csv=False):
    """Step 6: Run visualizer for specific index"""
    print("\n" + "="*60)
    print(f"STEP 6: RUNNING {index_name} VISUALIZER")
    print("="*60)
    
    python_exe = sys.executable
    script_path = Path(__file__).parent.parent / "src" / "generalized_beta_visualizer.py"
    
    commands = [[python_exe, str(script_path), index_name, "--summary"]]
    
    if export_csv:
        commands.append([python_exe, str(script_path), index_name, "--export"])
    
    success = True
    for cmd in commands:
        if not run_command(cmd, f"Run {index_name} visualizer"):
            success = False
    
    return success

def run_full_workflow(index_name, skip_validation=False, skip_transformation=False, export_csv=False):
    """Run the complete workflow for a specific index"""
    print(f"\n{'='*80}")
    print(f"RUNNING FULL WORKFLOW FOR {index_name}")
    print(f"{'='*80}")
    
    # Validate requirements
    if not validate_requirements():
        return False
    
    # Check index configuration
    config = get_config()
    if index_name not in config.get_available_indices():
        print(f"Error: Unknown index '{index_name}'")
        print(f"Available indices: {config.get_available_indices()}")
        return False
    
    # Validate index data availability
    validation = config.validate_index_data(index_name)
    if not validation['holdings_folder_exists']:
        print(f"Error: Holdings folder for {index_name} does not exist")
        print(f"Please create: raw_data/{index_name}_holdings/")
        return False
    
    if not validation['holdings_files_exist']:
        print(f"Error: No holdings files found for {index_name}")
        print(f"Please add CSV files to: raw_data/{index_name}_holdings/")
        return False
    
    # Step 1: Validate raw data (only if not skipping)
    if not skip_validation:
        if not validate_raw_data():
            print("Warning: Raw data validation failed, but continuing...")
    
    # Step 2: Transform raw data (only if not skipping)
    if not skip_transformation:
        if not transform_raw_data():
            print("Warning: Raw data transformation failed, but continuing...")
    
    # Step 3: Consolidate factor betas (common for all indices)
    if not consolidate_betas():
        print("Error: Factor beta consolidation failed")
        return False
    
    # Step 4: Consolidate holdings for specific index
    if not consolidate_holdings(index_name):
        print(f"Error: Holdings consolidation failed for {index_name}")
        return False
    
    # Step 5: Calculate weighted betas for specific index
    if not calculate_weighted_betas(index_name):
        print(f"Error: Weighted beta calculation failed for {index_name}")
        return False
    
    # Step 6: Run visualizer
    if not run_visualizer(index_name, export_csv):
        print(f"Error: Visualizer failed for {index_name}")
        return False
    
    print(f"\n{'='*80}")
    print(f"✓ WORKFLOW COMPLETED SUCCESSFULLY FOR {index_name}")
    print(f"{'='*80}")
    
    # Show next steps
    config = get_config()
    paths = config.get_paths(index_name)
    index_config = config.get_index_config(index_name)
    
    print(f"\nGenerated files:")
    print(f"  - Holdings: {paths['holdings_consolidated']}")
    print(f"  - Weighted betas: {paths['weighted_betas']}")
    if export_csv:
        print(f"  - CSV export: {paths['csv_export']}")
    
    print(f"\nNext steps:")
    print(f"  - Use the visualizer to explore {index_config.display_name} factor evolution")
    print(f"  - Analyze individual stock betas vs {index_config.display_name}")
    print(f"  - Export data for further analysis")
    
    return True

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Run complete historical beta analysis workflow')
    parser.add_argument('index', nargs='?', help='Index name (e.g., SP500, NASDAQ, DOW)')
    parser.add_argument('--skip-validation', action='store_true', help='Skip raw data validation step')
    parser.add_argument('--skip-transformation', action='store_true', help='Skip raw data transformation step')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--list-indices', action='store_true', help='List available indices')
    parser.add_argument('--validate-only', action='store_true', help='Only validate requirements and data')
    
    args = parser.parse_args()
    
    if args.list_indices:
        config = get_config()
        indices = config.get_available_indices()
        print("Available indices:")
        for idx in indices:
            print(f"  - {idx}")
        return
    
    if not args.index:
        print("Error: index argument is required unless using --list-indices")
        parser.print_help()
        return
    
    if args.validate_only:
        print("Validating requirements...")
        if validate_requirements():
            print("✓ All requirements satisfied")
            
            # Also validate data for the specified index
            config = get_config()
            if args.index in config.get_available_indices():
                validation = config.validate_index_data(args.index)
                print(f"\nData validation for {args.index}:")
                print(f"  Holdings folder exists: {'✓' if validation['holdings_folder_exists'] else '✗'}")
                print(f"  Holdings files exist: {'✓' if validation['holdings_files_exist'] else '✗'}")
                print(f"  Betas consolidated exists: {'✓' if validation['betas_consolidated_exists'] else '✗'}")
                print(f"  Holdings consolidated exists: {'✓' if validation['holdings_consolidated_exists'] else '✗'}")
                print(f"  Weighted betas exists: {'✓' if validation['weighted_betas_exists'] else '✗'}")
        else:
            print("✗ Requirements validation failed")
        return
    
    # Run full workflow
    success = run_full_workflow(
        args.index,
        skip_validation=args.skip_validation,
        skip_transformation=args.skip_transformation,
        export_csv=args.export_csv
    )
    
    if not success:
        print("\n✗ Workflow failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
