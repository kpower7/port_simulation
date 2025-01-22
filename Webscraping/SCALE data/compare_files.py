import pandas as pd
import numpy as np

def read_csv_safe(filepath):
    """Try to read CSV with different encodings"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            print(f"Trying {encoding} encoding...")
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"Success with {encoding} encoding!")
            return df
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding")
            continue
        except Exception as e:
            print(f"Other error with {encoding}: {str(e)}")
            continue
    raise Exception("Could not read file with any encoding")

# Read the CSV files
print("Reading files...")

# Read clean file
print("\nReading clean file...")
clean_path = r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\SCALE data\december_final_merged_clean.csv"
clean_df = read_csv_safe(clean_path)
print("Clean file shape:", clean_df.shape)
print("\nClean file first few rows:")
print(clean_df.head().to_string())
print("\nClean file columns:")
print(clean_df.columns.tolist())

print("\n" + "="*80 + "\n")

# Read original file
print("Reading original file...")
original_path = r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\SCALE data\december_final_merged.csv"
original_df = read_csv_safe(original_path)
print("Original file shape:", original_df.shape)
print("\nOriginal file first few rows:")
print(original_df.head().to_string())
print("\nOriginal file columns:")
print(original_df.columns.tolist())
