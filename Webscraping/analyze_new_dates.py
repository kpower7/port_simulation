import pandas as pd
import os

def analyze_dates(file_path, source):
    print(f"\nAnalyzing {os.path.basename(file_path)} ({source}):")
    try:
        df = pd.read_csv(file_path)
        
        # Different column names based on source
        date_col = 'arrivalDate' if source == 'IG' else 'DischargedDate'
        
        if date_col in df.columns:
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Add date-only column for grouping
            df['date_only'] = df[date_col].dt.date
            
            # Get counts by date
            date_counts = df['date_only'].value_counts().sort_index()
            
            # Show overall stats
            print(f"\nTotal records: {len(df)}")
            print(f"Records with valid dates: {df[date_col].notna().sum()}")
            
            # Show counts for December 2024
            print("\nDecember 2024 counts by date:")
            dec_2024_counts = date_counts[date_counts.index.astype(str).str.startswith('2024-12')]
            print(dec_2024_counts)
            
            # Show counts for other months (if any)
            other_counts = date_counts[~date_counts.index.astype(str).str.startswith('2024-12')]
            if len(other_counts) > 0:
                print("\nNon-December 2024 counts (potential outliers):")
                print(other_counts)
                print(f"\nTotal non-December records: {other_counts.sum()}")
        else:
            print(f"Warning: {date_col} column not found in file")
            print("Available columns:", df.columns.tolist())
    except Exception as e:
        print(f"Error processing file: {str(e)}")

print("=== Analysis for December 20 files ===")
# Analyze Dec 20 files
analyze_dates(os.path.join('SCALE data', 'Results_20241220.csv'), 'APM')
analyze_dates(os.path.join('SCALE data', 'Results_ig20_2712.csv'), 'IG')

print("\n=== Analysis for December 27 files ===")
# Analyze Dec 27 files
analyze_dates(os.path.join('SCALE data', 'Results_20241227.csv'), 'APM')
analyze_dates(os.path.join('SCALE data', 'Results_ig27_2712.csv'), 'IG')
