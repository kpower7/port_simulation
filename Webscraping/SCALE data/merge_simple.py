import pandas as pd

# Load tracking data
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].str.upper()
tracking_containers = set(tracking['CONTAINER NUMBER'].dropna())
print(f"Total tracking containers: {len(tracking_containers)}")

# Load all import files
import_files = [
    r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\combined_results.csv",
    'Results_ig20_2712.csv',
    'Results_ig27_2712.csv',
    'Results_ig2025_31.xlsx'
]

# Process each file
all_matches = {}  # container -> import row mapping
matches_by_file = {}

for file in import_files:
    print(f"\nProcessing {file}...")
    # Load file
    if file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    
    # Find matches in this file
    file_matches = 0
    for idx, row in df.iterrows():
        if pd.notna(row['containerNumber']):
            containers = str(row['containerNumber']).strip().split('\n')
            for container in containers:
                container = container.strip().upper()
                if container in tracking_containers and container not in all_matches:
                    all_matches[container] = row
                    file_matches += 1
    
    matches_by_file[file] = file_matches
    print(f"Found {file_matches} new matches")
    print(f"Total matches so far: {len(all_matches)}")

print("\nSummary:")
print(f"Total tracking containers: {len(tracking_containers)}")
print(f"Total matched containers: {len(all_matches)}")
print("\nMatches by file:")
for file, count in matches_by_file.items():
    print(f"{file}: {count}")

# Create merged DataFrame
print("\nCreating final merged dataset...")
merged_data = []

# For each tracking container that has a match
for _, tracking_row in tracking.iterrows():
    container = tracking_row['CONTAINER NUMBER']
    if container in all_matches:
        import_row = all_matches[container]
        # Combine the data
        combined = {}
        # Add tracking data with prefix
        for col in tracking_row.index:
            combined[f"tracking_{col}"] = tracking_row[col]
        # Add import data with prefix
        for col in import_row.index:
            combined[f"import_{col}"] = import_row[col]
        merged_data.append(combined)

# Convert to DataFrame
final_df = pd.DataFrame(merged_data)
print(f"\nFinal dataset has {len(final_df)} rows")

# Print column names
print("\nColumns in final dataset:")
print(final_df.columns.tolist())

# Save results
output_file = 'december_final_merged_clean.csv'
final_df.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")

# Show sample
print("\nSample of merged data (first 5 rows):")
sample_cols = ['tracking_CONTAINER NUMBER', 'import_containerNumber']
print(final_df[sample_cols].head())
