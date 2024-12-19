import pandas as pd
import numpy as np
from datetime import datetime

def load_weekly_data(file_path):
    df = pd.read_csv(file_path)
    # Extract date from filename
    date_str = file_path.split('_')[1].split('.')[0]
    df['ScrapeDate'] = pd.to_datetime(date_str, format='%Y%m%d')
    return df

def load_import_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    # Add source file info
    df['DataSource'] = file_path
    return df

# Load weekly container status data
weekly_files = [
    'Results_20241220.csv',
    'Results_20241227.csv',
    'Results_20250103.csv'
]

# Load and combine weekly data
weekly_dfs = []
for file in weekly_files:
    df = load_weekly_data(file)
    weekly_dfs.append(df)

# Combine all weekly data
combined_weekly = pd.concat(weekly_dfs, ignore_index=True)

# Sort by ScrapeDate and ContainerId, then keep the latest record for each container
latest_weekly = (combined_weekly.sort_values(['ContainerId', 'ScrapeDate'])
                .groupby('ContainerId')
                .last()
                .reset_index())

# Load import data files
import_files = [
    'Results_ig20_2712.csv',
    'Results_ig27_2712.csv',
    'Results_ig2025_31.xlsx'
]

# Load and combine import data
import_dfs = []
for file in import_files:
    df = load_import_data(file)
    import_dfs.append(df)

# Combine all import data
combined_import = pd.concat(import_dfs, ignore_index=True)

# Remove duplicate containers keeping the latest record based on the file source
# Assuming files are named in chronological order
latest_import = combined_import.drop_duplicates(subset='containerNumber', keep='last')

# Merge weekly status with import data
merged_data = pd.merge(
    latest_weekly,
    latest_import,
    left_on='ContainerId',
    right_on='containerNumber',
    how='outer',
    suffixes=('_weekly', '_import')
)

# Save the merged data
output_file = 'merged_december_container_data.csv'
merged_data.to_csv(output_file, index=False)

# Print summary statistics
print(f"\nMerged Data Summary:")
print(f"Total unique containers: {len(merged_data)}")
print(f"Containers with weekly status: {merged_data['ContainerId'].notna().sum()}")
print(f"Containers with import data: {merged_data['containerNumber'].notna().sum()}")
print(f"Containers with both weekly and import data: {merged_data['ContainerId'].notna() & merged_data['containerNumber'].notna()}.sum()")
print(f"\nData saved to: {output_file}")
