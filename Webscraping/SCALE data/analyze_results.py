import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
weekly_files = [
    'Results_20241220.csv',
    'Results_20241227.csv',
    'Results_20250103.csv'
]

ig_files = [
    'Results_ig20_2712.csv',
    'Results_ig27_2712.csv'
]

excel_file = 'Results_ig2025_31.xlsx'

def load_and_analyze_file(filepath):
    print(f"\nAnalyzing {filepath}")
    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nSample of first few rows:")
        print(df.head(2))
        print("\nData Types:")
        print(df.dtypes)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

# Analyze weekly files
print("\n=== Analyzing Weekly Result Files ===")
for file in weekly_files:
    df = load_and_analyze_file(file)

print("\n=== Analyzing IG Files ===")
for file in ig_files:
    df = load_and_analyze_file(file)

print("\n=== Analyzing Excel File ===")
df_excel = load_and_analyze_file(excel_file)
