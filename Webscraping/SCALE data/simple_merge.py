import pandas as pd

# 1. Combine all import (ig) data first
print("Loading import data...")
ig_dfs = []
for file in ['Results_ig20_2712.csv', 'Results_ig27_2712.csv', 'Results_ig2025_31.xlsx']:
    if file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    ig_dfs.append(df)

# Combine all import data
all_imports = pd.concat(ig_dfs, ignore_index=True)
# Keep latest record for each container
latest_imports = all_imports.drop_duplicates(subset='containerNumber', keep='last')
print(f"Total unique containers in import data: {len(latest_imports)}")

# 2. Load tracking data (using most recent snapshot)
print("\nLoading tracking data...")
tracking = pd.read_csv('Results_20250103.csv')
print(f"Total containers in tracking data: {len(tracking)}")

# 3. Simple merge on container number
print("\nMerging data...")
merged = pd.merge(
    latest_imports,
    tracking,
    left_on='containerNumber',
    right_on='CONTAINER NUMBER',
    how='left'
)

print(f"\nFinal merged dataset size: {len(merged)}")
print(f"Number of containers with tracking data: {merged['CONTAINER NUMBER'].notna().sum()}")

# Save merged data
merged.to_csv('december_merged_data.csv', index=False)
print("\nSaved to: december_merged_data.csv")
