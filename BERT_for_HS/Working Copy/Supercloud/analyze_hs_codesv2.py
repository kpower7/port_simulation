import pandas as pd
import numpy as np

# File paths
valid_file = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_valid_IND.csv.gz'
test_file = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_test_IND.csv.gz'

# Read the files
print("Reading files...")
df_valid = pd.read_csv(valid_file)
df_test = pd.read_csv(test_file)

# Basic comparison
print("\nBasic information:")
print(f"Valid file shape: {df_valid.shape}")
print(f"Test file shape: {df_test.shape}")

# Check if dataframes are exactly equal
are_identical = df_valid.equals(df_test)
print(f"\nAre the files identical? {are_identical}")

if not are_identical and df_valid.shape == df_test.shape:
    # If same shape but different content, find differences
    print("\nChecking differences in each column...")
    for column in df_valid.columns:
        column_identical = df_valid[column].equals(df_test[column])
        if not column_identical:
            print(f"\nDifferences found in column: {column}")
            # Show first few differences
            mask = df_valid[column] != df_test[column]
            if mask.any():
                print("\nFirst few differences:")
                print("Index | Valid | Test")
                print("-" * 40)
                for idx in mask[mask].index[:5]:  # Show first 5 differences
                    print(f"{idx} | {df_valid.loc[idx, column]} | {df_test.loc[idx, column]}") 