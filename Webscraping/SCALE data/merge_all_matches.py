import pandas as pd

# 1. Load tracking data
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].str.upper()
tracking_containers = set(tracking['CONTAINER NUMBER'].dropna())
print(f"Total tracking containers: {len(tracking_containers)}")

# Function to extract containers from a DataFrame
def get_containers_from_df(df):
    containers = {}  # container -> row mapping
    for idx, row in df.iterrows():
        if pd.notna(row['containerNumber']):
            for container in str(row['containerNumber']).strip().split('\n'):
                container = container.strip().upper()
                if container and container not in containers:  # Keep first occurrence
                    containers[container] = idx
    return containers

# 2. Load and process all import files
print("\nProcessing import files...")
import_files = [
    r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\combined_results.csv",
    'Results_ig20_2712.csv',
    'Results_ig27_2712.csv',
    'Results_ig2025_31.xlsx'
]

# Keep track of which containers we've found and their data
found_containers = {}  # container -> (file, row_idx) mapping
container_counts = {file: 0 for file in import_files}

for file in import_files:
    print(f"\nProcessing {file}...")
    try:
        if file.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, low_memory=False)
        
        # Get containers from this file
        file_containers = get_containers_from_df(df)
        
        # Find matches in this file
        for container in file_containers:
            if container in tracking_containers and container not in found_containers:
                found_containers[container] = (file, file_containers[container])
                container_counts[file] += 1
        
        print(f"Containers in file: {len(file_containers)}")
        print(f"New matches found: {container_counts[file]}")
        print(f"Total matches so far: {len(found_containers)}")
        
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

print("\nFinal Summary:")
print("--------------")
print(f"Total tracking containers: {len(tracking_containers)}")
print(f"Total matched containers: {len(found_containers)}")
print(f"Remaining unmatched: {len(tracking_containers - set(found_containers.keys()))}")

print("\nMatches by file:")
for file, count in container_counts.items():
    print(f"{file}: {count} matches")

# Now merge the data
print("\nMerging matched data...")
merged_rows = []

for container in tracking_containers:
    if container in found_containers:
        file, row_idx = found_containers[container]
        # Load the file if needed
        if file.endswith('.xlsx'):
            import_df = pd.read_excel(file)
        else:
            import_df = pd.read_csv(file, low_memory=False)
        
        # Get the tracking data
        tracking_row = tracking[tracking['CONTAINER NUMBER'] == container].iloc[0]
        
        # Get the import data
        import_row = import_df.iloc[row_idx]
        
        # Combine the data
        merged_row = pd.concat([tracking_row, import_row])
        merged_rows.append(merged_row)

# Create final DataFrame
final_df = pd.DataFrame(merged_rows)
print(f"\nFinal merged data has {len(final_df)} rows")

# Save the results
output_file = 'december_final_merged_clean.csv'
final_df.to_csv(output_file, index=False)
print(f"Saved merged data to: {output_file}")

# Print sample
print("\nSample of merged data (first 5 rows):")
print(final_df[['CONTAINER NUMBER', 'containerNumber', 'VesselName']].head())
