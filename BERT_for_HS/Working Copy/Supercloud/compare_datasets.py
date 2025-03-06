import pandas as pd
import hashlib
import numpy as np

# File paths
test_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_test_IND.csv.gz'
valid_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_valid_IND.csv.gz'

print("Loading datasets...")
dtypes = {'PRODUCT DESCRIPTION': 'str', 'PRODUCT DESCRIPTION_ASCII': 'str'}

# Load datasets
test_df = pd.read_csv(test_path, dtype=dtypes)
valid_df = pd.read_csv(valid_path, dtype=dtypes)

print(f"Test dataset shape: {test_df.shape}")
print(f"Validation dataset shape: {valid_df.shape}")

# Check if the shapes are identical
if test_df.shape == valid_df.shape:
    print("\nBoth datasets have the same number of rows and columns.")
else:
    print("\nDatasets have different shapes.")

# Check column names
test_cols = set(test_df.columns)
valid_cols = set(valid_df.columns)

if test_cols == valid_cols:
    print("Both datasets have the same column names.")
else:
    print("Datasets have different column names:")
    print(f"Columns only in test: {test_cols - valid_cols}")
    print(f"Columns only in validation: {valid_cols - test_cols}")

# Function to create a hash of a dataframe for comparison
def hash_dataframe(df):
    # Sort dataframe to ensure consistent comparison
    df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    # Convert to string and hash
    return hashlib.md5(pd.util.hash_pandas_object(df_sorted).values).hexdigest()

# Check if the content is identical
test_hash = hash_dataframe(test_df)
valid_hash = hash_dataframe(valid_df)

print(f"\nTest dataset hash: {test_hash}")
print(f"Validation dataset hash: {valid_hash}")

if test_hash == valid_hash:
    print("\nCONCLUSION: The datasets are IDENTICAL in content.")
else:
    print("\nCONCLUSION: The datasets are DIFFERENT in content.")
    
    # If different, let's check how many rows are shared
    print("\nPerforming detailed comparison...")
    
    # Create a unique identifier for each row (combining all columns)
    def create_row_id(row):
        return hash(tuple(row.values))
    
    test_df['row_id'] = test_df.apply(create_row_id, axis=1)
    valid_df['row_id'] = valid_df.apply(create_row_id, axis=1)
    
    test_ids = set(test_df['row_id'])
    valid_ids = set(valid_df['row_id'])
    
    common_ids = test_ids.intersection(valid_ids)
    
    print(f"Number of identical rows: {len(common_ids)}")
    print(f"Percentage of test dataset rows found in validation: {len(common_ids)/len(test_df)*100:.2f}%")
    print(f"Percentage of validation dataset rows found in test: {len(common_ids)/len(valid_df)*100:.2f}%")
    
    # Sample a few rows that are different
    test_unique = test_ids - valid_ids
    valid_unique = valid_ids - test_ids
    
    if len(test_unique) > 0:
        print("\nSample row from test dataset not in validation:")
        sample_test_unique = test_df[test_df['row_id'].isin(list(test_unique)[:1])]
        print(sample_test_unique.drop('row_id', axis=1).iloc[0])
    
    if len(valid_unique) > 0:
        print("\nSample row from validation dataset not in test:")
        sample_valid_unique = valid_df[valid_df['row_id'].isin(list(valid_unique)[:1])]
        print(sample_valid_unique.drop('row_id', axis=1).iloc[0])

# Check distribution of HS codes
print("\n===== HS CODE DISTRIBUTION COMPARISON =====")

# 2-digit HS codes
test_hs2 = test_df['HS CODE6'].astype(str).str[:2].value_counts()
valid_hs2 = valid_df['HS CODE6'].astype(str).str[:2].value_counts()

print(f"Number of unique 2-digit HS codes in test: {len(test_hs2)}")
print(f"Number of unique 2-digit HS codes in validation: {len(valid_hs2)}")

# Compare top 5 HS codes
print("\nTop 5 most common 2-digit HS codes in test:")
print(test_hs2.head(5))
print("\nTop 5 most common 2-digit HS codes in validation:")
print(valid_hs2.head(5))

# Check if the distributions are similar using correlation
all_hs2 = sorted(list(set(test_hs2.index).union(set(valid_hs2.index))))
test_counts = [test_hs2.get(code, 0) for code in all_hs2]
valid_counts = [valid_hs2.get(code, 0) for code in all_hs2]

correlation = np.corrcoef(test_counts, valid_counts)[0, 1]
print(f"\nCorrelation between test and validation HS code distributions: {correlation:.4f}")

if correlation > 0.99:
    print("The distributions of HS codes are extremely similar.")
elif correlation > 0.9:
    print("The distributions of HS codes are very similar.")
elif correlation > 0.7:
    print("The distributions of HS codes are somewhat similar.")
else:
    print("The distributions of HS codes are different.")
