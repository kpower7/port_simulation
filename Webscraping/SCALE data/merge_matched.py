import pandas as pd

# 1. Load tracking data
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].str.upper()
print(f"Tracking data rows: {len(tracking)}")

# 2. Load import data
print("\nLoading import data...")
combined_file = r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\combined_results.csv"
print(f"Processing {combined_file}")
combined = pd.read_csv(combined_file, low_memory=False)

# Process container numbers from combined data
print("Processing container numbers...")
import_data = []
for _, row in combined.iterrows():
    if pd.notna(row['containerNumber']):
        containers = str(row['containerNumber']).strip().split('\n')
        for container in containers:
            container = container.strip().upper()
            if container:
                new_row = row.copy()
                new_row['containerNumber'] = container
                import_data.append(new_row)

all_imports = pd.DataFrame(import_data)
print(f"Total import rows after splitting: {len(all_imports)}")
print(f"Unique containers in import data: {all_imports['containerNumber'].nunique()}")

# 3. Merge data
print("\nMerging data...")
merged = pd.merge(
    tracking,
    all_imports,
    left_on='CONTAINER NUMBER',
    right_on='containerNumber',
    how='left'
)

print("\nMerge Statistics:")
print(f"Total rows: {len(merged)}")
print(f"Rows with tracking data: {merged['CONTAINER NUMBER'].notna().sum()}")
print(f"Rows with import data: {merged['containerNumber'].notna().sum()}")
print(f"Unique containers with matches: {merged[merged['containerNumber'].notna()]['CONTAINER NUMBER'].nunique()}")

# Save only matched data
matched = merged[merged['containerNumber'].notna()]
output_file = 'december_final_merged_clean.csv'
matched.to_csv(output_file, index=False)
print(f"\nSaved {len(matched)} matched rows to: {output_file}")

# Print sample of matched data
print("\nSample of matched data (first 5 rows):")
sample_cols = ['CONTAINER NUMBER', 'VesselName', 'YardLocation']
print(matched[sample_cols].head())
