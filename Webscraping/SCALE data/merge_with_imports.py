import pandas as pd
import numpy as np

def clean_container_numbers(df):
    # Split container numbers on newlines and create new rows
    container_rows = []
    for _, row in df.iterrows():
        containers = str(row['containerNumber']).strip().split('\n')
        for container in containers:
            container = container.strip()
            if container:  # Only add non-empty containers
                new_row = row.copy()
                new_row['containerNumber'] = container
                container_rows.append(new_row)
    return pd.DataFrame(container_rows)

# 1. Load the merged tracking data and clean it
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
print(f"Initial tracking data rows: {len(tracking)}")
print(f"Unique containers in tracking: {tracking['CONTAINER NUMBER'].nunique()}")

# 2. Load and combine import data
print("\nLoading import data...")
ig_files = ['Results_ig20_2712.csv', 'Results_ig27_2712.csv', 'Results_ig2025_31.xlsx']
all_imports = None

for file in ig_files:
    print(f"Loading {file}")
    if file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file, low_memory=False)
    
    # Clean and split container numbers
    df = clean_container_numbers(df)
    
    if all_imports is None:
        all_imports = df
    else:
        # Only add containers we haven't seen before
        existing_containers = set(all_imports['containerNumber'])
        df = df[~df['containerNumber'].isin(existing_containers)]
        all_imports = pd.concat([all_imports, df], ignore_index=True)

print(f"Total unique containers in import data: {len(all_imports)}")

# 3. Merge tracking with import data
print("\nMerging data...")
# Convert container numbers to uppercase for consistent matching
tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].str.upper()
all_imports['containerNumber'] = all_imports['containerNumber'].str.upper()

# Create a temporary DataFrame with just container numbers for matching
import_containers = pd.DataFrame({'containerNumber': all_imports['containerNumber'].unique()})

# First merge to get all tracking data containers
merged = pd.merge(
    tracking,
    import_containers,
    left_on='CONTAINER NUMBER',
    right_on='containerNumber',
    how='left'
)

print(f"\nFinal Statistics:")
print(f"Total containers from tracking: {len(merged)}")
print(f"Containers with matches in import data: {merged['containerNumber'].notna().sum()}")

# Get the matched containers and merge with full import data
matched_containers = merged[merged['containerNumber'].notna()]['containerNumber'].unique()
print(f"Unique matched containers: {len(matched_containers)}")

# Get the full data for matched containers
matched_imports = all_imports[all_imports['containerNumber'].isin(matched_containers)]
final_merged = pd.merge(
    merged[merged['containerNumber'].notna()].drop(columns=['containerNumber']),  # Only keep matched containers
    matched_imports,
    left_on='CONTAINER NUMBER',
    right_on='containerNumber',
    how='left'
)

# Save merged data
output_file = 'december_final_merged_clean.csv'
final_merged.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Print some sample data to verify merge
print("\nSample of merged data (first 5 rows):")
sample_cols = ['CONTAINER NUMBER', 'containerNumber', 'VesselName', 'YardLocation']
print(final_merged[sample_cols].head())

# Print containers that didn't match
print("\nSample of unmatched containers from tracking data:")
unmatched = merged[merged['containerNumber'].isna()]
print(f"Total unmatched: {len(unmatched)}")
if len(unmatched) > 0:
    print("Sample unmatched containers:")
    print(unmatched['CONTAINER NUMBER'].head())
