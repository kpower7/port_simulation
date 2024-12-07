import pandas as pd
import os
from datetime import datetime

# Read the merged results
file_path = os.path.join('SCALE data', 'merged_results.csv')
df = pd.read_csv(file_path)

# Find all date-related columns
date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
print('Date columns found:', date_columns)

# Function to safely convert to datetime
def safe_to_datetime(x):
    if pd.isna(x) or x == 'No':
        return pd.NaT
    try:
        return pd.to_datetime(x)
    except:
        return pd.NaT

# Analyze each date column
for col in date_columns:
    print(f'\nDate range for {col}:')
    # Convert column to datetime
    date_series = df[col].apply(safe_to_datetime)
    # Get min and max dates
    min_date = date_series.min()
    max_date = date_series.max()
    print(f'Earliest: {min_date if not pd.isna(min_date) else "No valid date"}')
    print(f'Latest: {max_date if not pd.isna(max_date) else "No valid date"}')
