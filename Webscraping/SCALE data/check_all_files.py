import pandas as pd

# Check all tracking files
tracking_files = [
    'Results_20250103.csv',
    'Results_20241227.csv',
    'Results_20241220.csv',
    'Results_20241217.csv',
    'Results.csv'
]

print("Tracking Files Analysis:")
print("-----------------------")
for file in tracking_files:
    try:
        df = pd.read_csv(file)
        print(f"\n{file}:")
        print(f"Total rows: {len(df)}")
        print(f"Unique containers: {df['CONTAINER NUMBER'].nunique()}")
        print("Sample containers:")
        print(df['CONTAINER NUMBER'].head(3))
    except Exception as e:
        print(f"\nError reading {file}: {str(e)}")

print("\n\nImport Files Analysis:")
print("---------------------")
import_files = [
    'Results_ig20_2712.csv',
    'Results_ig27_2712.csv',
    'Results_ig2025_31.xlsx'
]

for file in import_files:
    try:
        if file.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        print(f"\n{file}:")
        print(f"Total rows: {len(df)}")
        print(f"Unique containers: {df['containerNumber'].nunique()}")
        print("Sample containers:")
        print(df['containerNumber'].head(3))
    except Exception as e:
        print(f"\nError reading {file}: {str(e)}")
