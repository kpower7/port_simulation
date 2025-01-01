import pandas as pd

# Load tracking data
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)

print(f"\nTracking Data Stats:")
print(f"Total rows: {len(tracking)}")
print(f"Unique containers: {tracking['CONTAINER NUMBER'].nunique()}")

print("\nSample container numbers:")
print(tracking['CONTAINER NUMBER'].head())

# Check for any empty or invalid container numbers
print("\nChecking for invalid containers:")
print(f"Empty container numbers: {tracking['CONTAINER NUMBER'].isna().sum()}")
print(f"Blank container numbers: {(tracking['CONTAINER NUMBER'] == '').sum()}")

# Load one import file to compare format
print("\nLoading one import file to compare...")
ig = pd.read_csv('Results_ig20_2712.csv', low_memory=False)
print("\nImport Data Sample:")
print(ig['containerNumber'].head())

# Try different case matching
tracking_upper = tracking['CONTAINER NUMBER'].str.upper()
sample_ig_upper = ig['containerNumber'].str.upper()

print("\nSample container number formats:")
print("Tracking data format:", tracking['CONTAINER NUMBER'].iloc[0])
print("Import data format:", ig['containerNumber'].iloc[0])
