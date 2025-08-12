import os
import glob
import pandas as pd
import json
from pathlib import Path

# Load config.json
with open('config.json', 'r') as f:
    config = json.load(f)

indices = config['indices']
index_keys = list(indices.keys())
print("Available indices:")
for i, key in enumerate(index_keys):
    print(f"  {i+1}. {indices[key]['display_name']} ({key})")

# Prompt user for index selection
while True:
    sel = input(f"Select index by number (1-{len(index_keys)}): ").strip()
    if sel.isdigit() and 1 <= int(sel) <= len(index_keys):
        index_key = index_keys[int(sel)-1]
        break
    print("Invalid selection. Try again.")

index_cfg = indices[index_key]
print(f"\nSelected index: {index_cfg['display_name']} ({index_key})\n")



# Paths from config
RAW_BETAS_DIR = Path('raw_data/master_factor_betas_transformed')
RAW_HOLDINGS_DIR = Path(index_cfg['holdings_folder'])
CONSOL_BETAS_PATH = Path('data_warehouse/consolidated_factor_betas.parquet')
CONSOL_HOLDINGS_PATH = Path(index_cfg['holdings_consolidated_file'])
SECTOR_WEEKLY_SCRIPT = Path('util/calculate_sector_weekly_avg_betas.py')
SYMBOL_SECTOR_PATH = Path(index_cfg['symbol_sector_file'])

# Check required folders/files
missing = []
if not RAW_BETAS_DIR.exists():
    missing.append(f"Missing folder: {RAW_BETAS_DIR}")
if not RAW_HOLDINGS_DIR.exists():
    missing.append(f"Missing folder: {RAW_HOLDINGS_DIR}")
if not CONSOL_BETAS_PATH.parent.exists():
    missing.append(f"Missing folder: {CONSOL_BETAS_PATH.parent}")
if not CONSOL_HOLDINGS_PATH.parent.exists():
    missing.append(f"Missing folder: {CONSOL_HOLDINGS_PATH.parent}")
if not SECTOR_WEEKLY_SCRIPT.exists():
    missing.append(f"Missing script: {SECTOR_WEEKLY_SCRIPT}")
if not SYMBOL_SECTOR_PATH.exists():
    missing.append(f"Missing symbol/sector file: {SYMBOL_SECTOR_PATH}")
if missing:
    print("\nERROR: The following required folders/files are missing:")
    for m in missing:
        print("  - " + str(m))
    print("\nPlease create these folders/files before running the workflow.")
    exit(1)


## 1. Append only new factor betas to consolidated_factor_betas.parquet
print('Checking for new factor betas to append...')
beta_files = sorted(glob.glob(str(RAW_BETAS_DIR / '*.csv')))
if CONSOL_BETAS_PATH.exists():
    consol_betas = pd.read_parquet(CONSOL_BETAS_PATH)
    # Always use 'Date' column, convert to string for comparison
    existing_dates = set(pd.to_datetime(consol_betas['Date']).dt.strftime('%m_%d_%Y'))
else:
    consol_betas = pd.DataFrame()
    existing_dates = set()
new_beta_dfs = []
for f in beta_files:
    df = pd.read_csv(f)
    # Ensure date column exists, else extract from filename
    if 'Date' not in df.columns:
        import re
        m = re.search(r'(\d{1,2}_\d{1,2}_\d{4})', os.path.basename(f))
        if m:
            date_str = m.group(1)
            df['Date'] = pd.to_datetime(date_str, format='%m_%d_%Y')
        else:
            raise ValueError(f"Could not extract date from filename: {f}")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    # Only keep rows with new weeks (compare as string)
    new_rows = df[~df['Date'].dt.strftime('%m_%d_%Y').isin(existing_dates)]
    if not new_rows.empty:
        new_weeks = set(new_rows['Date'].dt.strftime('%m_%d_%Y'))
        print(f"  Appending from {os.path.basename(f)}: weeks {sorted(new_weeks)}")
        new_beta_dfs.append(new_rows)
if new_beta_dfs or not CONSOL_BETAS_PATH.exists():
    # If output does not exist, create it from all available input
    if not CONSOL_BETAS_PATH.exists():
        print(f"Output file {CONSOL_BETAS_PATH} does not exist. Creating from all available input...")
        all_betas = pd.concat([df for df in [consol_betas] + new_beta_dfs if not df.empty], ignore_index=True)
        all_betas.to_parquet(CONSOL_BETAS_PATH, index=False)
        print(f'✓ Created {CONSOL_BETAS_PATH} with {len(all_betas)} rows.')
    else:
        updated_betas = pd.concat([consol_betas] + new_beta_dfs, ignore_index=True)
        updated_betas.to_parquet(CONSOL_BETAS_PATH, index=False)
        print(f'✓ Appended {sum(len(df) for df in new_beta_dfs)} new beta rows. Total: {len(updated_betas)} rows.')
else:
    print('No new factor beta data to append.')

## 2. Append only new holdings to consolidated holdings parquet
print(f'Checking for new {index_cfg["display_name"]} holdings to append...')
holdings_files = sorted(glob.glob(str(RAW_HOLDINGS_DIR / '*.csv')))
if CONSOL_HOLDINGS_PATH.exists():
    consol_holdings = pd.read_parquet(CONSOL_HOLDINGS_PATH)
    existing_dates = set(pd.to_datetime(consol_holdings['Date']).dt.strftime('%m_%d_%Y'))
else:
    consol_holdings = pd.DataFrame()
    existing_dates = set()
new_holdings_dfs = []
for f in holdings_files:
    df = pd.read_csv(f)
    # Ensure date column exists, else extract from filename
    if 'Date' not in df.columns:
        import re
        m = re.search(r'(\d{1,2}_\d{1,2}_\d{4})', os.path.basename(f))
        if m:
            date_str = m.group(1)
            df['Date'] = pd.to_datetime(date_str, format='%m_%d_%Y')
        else:
            raise ValueError(f"Could not extract date from filename: {f}")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    new_rows = df[~df['Date'].dt.strftime('%m_%d_%Y').isin(existing_dates)]
    if not new_rows.empty:
        new_weeks = set(new_rows['Date'].dt.strftime('%m_%d_%Y'))
        print(f"  Appending from {os.path.basename(f)}: weeks {sorted(new_weeks)}")
        new_holdings_dfs.append(new_rows)
if new_holdings_dfs or not CONSOL_HOLDINGS_PATH.exists():
    # If output does not exist, create it from all available input
    if not CONSOL_HOLDINGS_PATH.exists():
        print(f"Output file {CONSOL_HOLDINGS_PATH} does not exist. Creating from all available input...")
        all_holdings = pd.concat([df for df in [consol_holdings] + new_holdings_dfs if not df.empty], ignore_index=True)
        all_holdings.to_parquet(CONSOL_HOLDINGS_PATH, index=False)
        print(f'✓ Created {CONSOL_HOLDINGS_PATH} with {len(all_holdings)} rows.')
    else:
        updated_holdings = pd.concat([consol_holdings] + new_holdings_dfs, ignore_index=True)
        updated_holdings.to_parquet(CONSOL_HOLDINGS_PATH, index=False)
        print(f'✓ Appended {sum(len(df) for df in new_holdings_dfs)} new holdings rows. Total: {len(updated_holdings)} rows.')
else:
    print(f'No new {index_cfg["display_name"]} holdings data to append.')

## 3. Run sector weekly aggregation script
print('Running sector weekly aggregation...')
os.system(f'python {SECTOR_WEEKLY_SCRIPT} {index_key}')

print('\nAll steps complete. Check outputs above.')
