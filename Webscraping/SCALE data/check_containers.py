import pandas as pd

# Load a sample from Results.csv
tracking = pd.read_csv('Results.csv')
print("\nResults.csv info:")
print(f"Total rows: {len(tracking)}")
print(f"Unique containers: {tracking['CONTAINER NUMBER'].nunique()}")
print("\nSample container numbers:")
print(tracking['CONTAINER NUMBER'].head())

# Load a sample from import data
import_df = pd.read_csv('Results_ig27_2712.csv')
print("\nImport data sample:")
print(f"Total rows: {len(import_df)}")
print(f"Unique containers: {import_df['containerNumber'].nunique()}")
print("\nSample container numbers:")
print(import_df['containerNumber'].head())

# Check for exact matches between these two files
tracking_containers = set(tracking['CONTAINER NUMBER'].dropna())
import_containers = set(import_df['containerNumber'].dropna())
matches = tracking_containers.intersection(import_containers)
print(f"\nMatches between these two files: {len(matches)}")

# Check for case differences
print("\nChecking case differences...")
tracking_upper = set(x.upper() if isinstance(x, str) else x for x in tracking_containers)
import_upper = set(x.upper() if isinstance(x, str) else x for x in import_containers)
case_matches = tracking_upper.intersection(import_upper)
print(f"Matches after converting to uppercase: {len(case_matches)}")
