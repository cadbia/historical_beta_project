def is_non_numeric_and_short(symbol):
    return symbol.isalpha() and len(symbol) < 5

import os
import glob
import pandas as pd

# GICS sector mapping
GICS_MAP = {
    "Industrials": "Industrials",
    "Financials": "Financials",
    "Finance": "Financials",
    "Health Care": "Health Care",
    "Healthcare": "Health Care",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Cyclicals": "Consumer Discretionary",
    "Consumer Non-Cyclicals": "Consumer Staples",
    "Consumer Staples": "Consumer Staples",
    "Consumer Services": "Consumer Discretionary",
    # "Consumer Services": "Consumer Services",
    "Information Technology": "Information Technology",
    "Technology": "Information Technology",
    "Materials": "Materials",
    "Non-Energy Materials": "Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
    "Telecommunications": "Communication Services",
    "Energy": "Energy",
    "Utilities": "Utilities",
     # "Business Services": "Business Services",
    "Business Services": "Industrials", # catch‑all; split further if needed
}

def standardize_sector(sector):
    # Normalize all null/empty/NA/0 values to '@NA'
    if pd.isna(sector):
        return "@NA"
    s = str(sector).strip()
    if s in {"", "@NA", "NA", "N/A", "nan", "NaN", "None", "0", "Not Classified"}:
        return "@NA"
    return GICS_MAP.get(s, s)

def main():
    import json
    folder = os.path.join(os.path.dirname(__file__), '../raw_data/master_sector_files')
    data_warehouse = os.path.join(os.path.dirname(__file__), '../data_warehouse')
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    # Prompt for index
    with open(config_path, 'r') as f:
        config = json.load(f)
    indices = config.get('indices', {})
    print("Available indices:")
    for i, idx in enumerate(indices.keys()):
        print(f"{i+1}. {idx}")
    idx_choice = input("Select index by number: ").strip()
    try:
        idx_choice = int(idx_choice) - 1
        index_key = list(indices.keys())[idx_choice]
    except Exception:
        print("Invalid selection.")
        return
    index_cfg = indices[index_key]
    holdings_consolidated_file = index_cfg['holdings_consolidated_file']
    symbol_sector_file = index_cfg['symbol_sector_file']
    symbol_sector_file_conflicts = symbol_sector_file.replace('.csv', '_conflicts.csv')
    symbol_sector_file_no_conflicts = symbol_sector_file.replace('.csv', '_no_conflicts.csv')
    # Load tickers for selected index
    if not os.path.exists(holdings_consolidated_file):
        print(f"Holdings consolidated file not found: {holdings_consolidated_file}")
        return
    holdings_df = pd.read_parquet(holdings_consolidated_file)
    index_symbols = set(holdings_df['Symbol'].unique())
    # Read all files once and store DataFrames in a dict
    import re
    def extract_date(file):
        m = re.search(r'master_sector_(\d{2})_(\d{2})_(\d{4})', os.path.basename(file))
        if m:
            month, day, year = map(int, m.groups())
            return (year, month, day)
        return (0, 0, 0)
    pattern = os.path.join(folder, 'master_sector_*.csv')
    files = [f for f in glob.glob(pattern) if not f.endswith('consolidated.csv')]
    files = sorted(files, key=extract_date)
    file_dfs = {}
    all_rows = []
    for file in files:
        df = pd.read_csv(file, dtype=str)
        if not set(['Symbol', 'Company Name', 'Sector']).issubset(df.columns):
            continue
        df = df[['Symbol', 'Company Name', 'Sector']]
        df['Sector'] = df['Sector'].apply(standardize_sector)
        all_rows.append(df)
        file_dfs[file] = df
    if not all_rows:
        print("No valid sector files found.")
        return
    combined = pd.concat(all_rows, ignore_index=True)
    # Filter to only tickers in selected index
    combined_index = combined[combined['Symbol'].isin(index_symbols)]
    # Group by Symbol, collect all unique sectors after standardization
    sector_map = combined_index.groupby('Symbol')['Sector'].agg(lambda x: set(x.dropna())).to_dict()
    # Collect and save enhanced conflicts
    conflict_rows = []
    conflict_symbols = set()
    for symbol, sectors in sector_map.items():
        filtered = sectors - set(['@NA', 'nan', 'NaN', 'None', ''])
        if len(filtered) > 1:
            # Only save to conflicts file if symbol is non-numeric and <5 chars, or starts with * and is <=5 chars
            def is_special_conflict_symbol(symbol):
                if symbol.isalpha() and len(symbol) < 5:
                    return True
                if symbol.startswith('*') and len(symbol) <= 5:
                    return True
                return False
            if is_special_conflict_symbol(symbol):
                print(f"Conflict for {symbol}: {sectors}")
                conflict_symbols.add(symbol)
                # Get all rows for this symbol
                rows = combined_index[combined_index['Symbol'] == symbol]
                company_name = rows.iloc[-1]['Company Name'] if not rows.empty else ''
                # Find which files this symbol appears in
                files_with_symbol = set()
                for file, df in file_dfs.items():
                    if 'Symbol' in df.columns and symbol in df['Symbol'].values:
                        files_with_symbol.add(file)
                num_files_symbol = len(files_with_symbol)
                # For each sector, count in how many files it appears for this symbol
                sector_file_counts = {}
                for sector in filtered:
                    count = 0
                    for file in files_with_symbol:
                        df = file_dfs[file]
                        if 'Symbol' in df.columns and 'Sector' in df.columns:
                            sub = df[df['Symbol'] == symbol]
                            if not sub.empty and 'Sector' in sub.columns:
                                sub_sector = sub['Sector']
                                if sector in set(sub_sector):
                                    count += 1
                    sector_file_counts[sector] = count
                most_recent_sector = rows.iloc[-1]['Sector'] if not rows.empty else ''
                max_count = max(sector_file_counts.values()) if sector_file_counts else 0
                sectors_with_max = [s for s, c in sector_file_counts.items() if c == max_count]
                sector_with_highest = sorted(sectors_with_max)[0] if sectors_with_max else ''
                flag = (most_recent_sector == sector_with_highest)
                row = [symbol, company_name, most_recent_sector, flag]
                for sector in sorted(filtered):
                    count = sector_file_counts.get(sector, 0)
                    percent = round(100 * count / num_files_symbol, 2) if num_files_symbol > 0 else 0.0
                    fraction = f"{count}/{num_files_symbol}"
                    percent_str = f"{percent} ({fraction})"
                    row.extend([sector, percent_str])
                conflict_rows.append(row)
    # Save index-specific symbol-sector files
    if conflict_rows:
        max_sectors = max((len(row)-3)//2 for row in conflict_rows)
        col_names = ['Symbol', 'Company Name', 'Most_Recent_Sector', 'Most_Recent_Matches_Highest']
        for i in range(max_sectors):
            col_names.extend([f'Sector_{i+1}', f'Percent_{i+1}'])
        padded_rows = [row + ['']*(2*max_sectors+3-len(row)) for row in conflict_rows]
        conflicts_df = pd.DataFrame(padded_rows, columns=col_names)
        conflicts_df.to_csv(symbol_sector_file_conflicts, index=False)
        print(f"Saved sector conflicts for index {index_key} to {symbol_sector_file_conflicts}")
    # Exclude conflicted symbols from deduplicated output
    deduped = combined_index.drop_duplicates(subset=['Symbol'], keep='first')
    deduped_no_conflicts = deduped[~deduped['Symbol'].isin(conflict_symbols)]
    deduped_no_conflicts = deduped_no_conflicts[['Symbol', 'Company Name', 'Sector']]
    deduped_no_conflicts.to_csv(symbol_sector_file_no_conflicts, index=False)
    print(f"Wrote symbol-sector file for index {index_key} (excluding conflicts) to {symbol_sector_file_no_conflicts}")
    if conflict_rows:
        print(f"Note: {symbol_sector_file_no_conflicts} omits symbols with sector conflicts (see {symbol_sector_file_conflicts})")
    else:
        print(f"No sector conflicts found for index {index_key}. All symbols included in {symbol_sector_file_no_conflicts}.")

if __name__ == "__main__":
    main()
