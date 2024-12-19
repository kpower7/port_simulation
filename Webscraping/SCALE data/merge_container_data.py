import pandas as pd
import numpy as np
from datetime import datetime

def clean_container_number(number):
    if pd.isna(number):
        return number
    return str(number).strip().upper()

# Load tracking tool data (most recent snapshot)
tracking_df = pd.read_csv('Results_20250103.csv')  # Using the most recent snapshot
tracking_df['CONTAINER NUMBER'] = tracking_df['CONTAINER NUMBER'].apply(clean_container_number)

# Load all import data
import_dfs = []
for file in ['Results_ig20_2712.csv', 'Results_ig27_2712.csv', 'Results_ig2025_31.xlsx']:
    if file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df['containerNumber'] = df['containerNumber'].apply(clean_container_number)
    import_dfs.append(df)

# Combine import data, keeping the latest record for each container
combined_import = pd.concat(import_dfs, ignore_index=True)
latest_import = combined_import.drop_duplicates(subset='containerNumber', keep='last')

# Merge tracking and import data
merged_data = pd.merge(
    tracking_df,
    latest_import,
    left_on='CONTAINER NUMBER',
    right_on='containerNumber',
    how='outer',
    indicator=True
)

# Print matching statistics
print("\nMerging Statistics:")
print(f"Total containers in tracking data: {len(tracking_df)}")
print(f"Total containers in import data: {len(latest_import)}")
print("\nMerge results:")
print(merged_data['_merge'].value_counts())

# Save only containers that appear in both datasets
matched_data = merged_data[merged_data['_merge'] == 'both'].copy()
matched_data.drop('_merge', axis=1, inplace=True)

print(f"\nTotal matched containers: {len(matched_data)}")

# Save the matched data
output_file = 'december_container_data_matched.csv'
matched_data.to_csv(output_file, index=False)
print(f"\nMatched data saved to: {output_file}")

# Investigate some unmatched containers
print("\nSample of unmatched containers:")
print("\nIn tracking but not import:")
tracking_only = merged_data[merged_data['_merge'] == 'left_only']['CONTAINER NUMBER'].head()
print(tracking_only)
print("\nIn import but not tracking:")
import_only = merged_data[merged_data['_merge'] == 'right_only']['containerNumber'].head()
print(import_only)
