import pandas as pd
import numpy as np

def process_file(file_path):
    """Process a single import file and return container matches"""
    print(f"Processing {file_path}...")
    
    # Load file
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Extract containers
    containers_data = {}
    for _, row in df.iterrows():
        if pd.notna(row['containerNumber']):
            containers = str(row['containerNumber']).strip().split('\n')
            for container in containers:
                container = container.strip().upper()
                if container and container not in containers_data:
                    # Only keep essential columns
                    data = {
                        'import_containerNumber': container,
                        'import_VesselName': row.get('VesselName', ''),
                        'import_VoyageNumber': row.get('VoyageNumber', ''),
                        'import_PortOfLoading': row.get('PortOfLoading', ''),
                        'import_PortOfDischarge': row.get('PortOfDischarge', ''),
                        'import_CarrierName': row.get('CarrierName', ''),
                        'import_source_file': file_path
                    }
                    containers_data[container] = data
    
    return containers_data

# 1. Load tracking data
print("Loading tracking data...")
tracking = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Port of NYNJ - Container Data.csv", low_memory=False)
tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].str.upper()
print(f"Tracking rows: {len(tracking)}")
print("\nTracking data columns:")
print(tracking.columns.tolist())

# 2. Process import files
import_files = [
    r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\combined_results.csv",
    r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_ig20_2712.csv",
    r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_ig27_2712.csv",
    r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_ig2025_31.xlsx"
]

# Process each file and collect matches
all_matches = {}
matches_by_file = {}

for file in import_files:
    file_data = process_file(file)
    new_matches = 0
    
    # Only keep matches we haven't seen before
    for container, data in file_data.items():
        if container not in all_matches:
            all_matches[container] = data
            new_matches += 1
    
    matches_by_file[file] = new_matches
    print(f"Found {new_matches} new matches")
    print(f"Total matches so far: {len(all_matches)}")

# 3. Create final merged dataset
print("\nMerging data...")
merged_rows = []

for _, tracking_row in tracking.iterrows():
    container = tracking_row['CONTAINER NUMBER']
    if container in all_matches:
        # Get import data for this container
        import_data = all_matches[container]
        
        # Combine tracking and import data
        row_data = {
            'CONTAINER NUMBER': container,
            'VESSEL NAME': tracking_row['VESSEL NAME'],
            **import_data
        }
        merged_rows.append(row_data)

# Convert to DataFrame
final_df = pd.DataFrame(merged_rows)
print(f"\nFinal dataset has {len(final_df)} rows")

# Save results
output_file = r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\december_final_merged_clean.csv"
final_df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")

# Print some stats
print("\nStats:")
print(f"Total tracking containers: {tracking['CONTAINER NUMBER'].nunique()}")
print(f"Total matched containers: {len(final_df)}")

print("\nMatches by file:")
for file, count in matches_by_file.items():
    print(f"{file}: {count} matches")

# Show sample
print("\nSample of merged data (first 3 rows):")
sample_cols = ['CONTAINER NUMBER', 'VESSEL NAME', 'import_VesselName', 'import_source_file']
print(final_df[sample_cols].head(3))
