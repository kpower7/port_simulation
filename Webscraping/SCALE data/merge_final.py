import pandas as pd
import shutil
import os

def clean_container_number(number):
    if pd.isna(number):
        return None
    # Split on newline and take first valid container number
    parts = str(number).strip().split('\n')
    return parts[0].strip()

# First, copy the files from Downloads to our working directory
downloads_path = os.path.expanduser("~/Downloads")
working_dir = r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\SCALE data"

# Copy the additional files
for file in ['Results.csv', 'Results_20241217.csv']:
    src = os.path.join(downloads_path, file)
    dst = os.path.join(working_dir, file)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {file} to working directory")

# 1. Load and combine all import (ig) data, keeping only unique containers
print("\nProcessing import data...")
ig_files = ['Results_ig20_2712.csv', 'Results_ig27_2712.csv', 'Results_ig2025_31.xlsx']
all_imports = None

for file in ig_files:
    print(f"Loading {file}")
    if file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    
    # Clean container numbers
    df['containerNumber'] = df['containerNumber'].apply(clean_container_number)
    df = df.dropna(subset=['containerNumber'])
    
    if all_imports is None:
        all_imports = df
    else:
        # Only add containers we haven't seen before
        existing_containers = set(all_imports['containerNumber'])
        df = df[~df['containerNumber'].isin(existing_containers)]
        all_imports = pd.concat([all_imports, df], ignore_index=True)

print(f"Total unique containers in import data: {len(all_imports)}")

# 2. Load tracking data in reverse chronological order and merge
tracking_files = [
    'Results_20250103.csv',  # Newest first
    'Results_20241227.csv',
    'Results_20241220.csv',
    'Results_20241217.csv',
    'Results.csv'  # From Dec 11
]

merged_data = all_imports.copy()
containers_with_tracking = set()  # Keep track of which containers we've already got tracking data for

for file in tracking_files:
    print(f"\nProcessing {file}")
    if not os.path.exists(file):
        print(f"Warning: {file} not found, skipping...")
        continue
        
    tracking = pd.read_csv(file)
    tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].apply(clean_container_number)
    tracking = tracking.dropna(subset=['CONTAINER NUMBER'])
    
    # Only process containers we haven't got tracking data for yet
    containers_to_process = set(merged_data['containerNumber']) - containers_with_tracking
    if not containers_to_process:
        print("All containers have tracking data, stopping...")
        break
        
    print(f"Looking for tracking data for {len(containers_to_process)} containers")
    
    # Filter tracking data to only new containers
    relevant_tracking = tracking[tracking['CONTAINER NUMBER'].isin(containers_to_process)]
    
    # Drop any duplicate columns before merging
    tracking_cols = [col for col in relevant_tracking.columns if col != 'CONTAINER NUMBER']
    
    # Merge new tracking data with unique suffixes
    merged_data = pd.merge(
        merged_data,
        relevant_tracking[['CONTAINER NUMBER'] + tracking_cols],
        left_on='containerNumber',
        right_on='CONTAINER NUMBER',
        how='left',
        suffixes=('', f'_{file.split(".")[0]}')
    )
    
    # Update our set of containers that have tracking data
    containers_with_tracking.update(relevant_tracking['CONTAINER NUMBER'])
    print(f"Found tracking data for {len(relevant_tracking)} containers")

print(f"\nFinal Statistics:")
print(f"Total containers: {len(merged_data)}")
print(f"Containers with tracking data: {len(containers_with_tracking)}")

# Save merged data
output_file = 'december_final_merged.csv'
merged_data.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Print some diagnostics about unmatched containers
print("\nSample of containers without tracking data:")
unmatched = merged_data[~merged_data['containerNumber'].isin(containers_with_tracking)]
print(unmatched['containerNumber'].head())
