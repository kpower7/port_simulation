import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
train_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_train_IND.csv.gz'
test_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_test_IND.csv.gz'
valid_path = 'C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\BERT_for_HS\\Working Copy\\Supercloud\\hc_codes_valid_IND.csv.gz'

# Check if files exist
for path in [train_path, test_path, valid_path]:
    if not os.path.exists(path):
        print(f"Warning: File {path} does not exist!")

# Load datasets
print("Loading datasets...")
dtypes = {'PRODUCT DESCRIPTION': 'str', 'PRODUCT DESCRIPTION_ASCII': 'str'}

try:
    train_df = pd.read_csv(train_path, dtype=dtypes)
    test_df = pd.read_csv(test_path, dtype=dtypes)
    valid_df = pd.read_csv(valid_path, dtype=dtypes)
    print("Datasets loaded successfully!")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Function to extract 2-digit HS code
def extract_2digit_hs(df, hs_column='HS CODE6'):
    # Convert to string first to handle any numeric types
    df['HS_2digit'] = df[hs_column].astype(str).str[:2]
    return df

# Apply to all datasets
train_df = extract_2digit_hs(train_df)
test_df = extract_2digit_hs(test_df)
valid_df = extract_2digit_hs(valid_df)

# Basic dataset info
print("\n===== DATASET OVERVIEW =====")
print(f"Train set: {len(train_df)} records")
print(f"Test set: {len(test_df)} records")
print(f"Validation set: {len(valid_df)} records")
print(f"Total: {len(train_df) + len(test_df) + len(valid_df)} records")

# Analyze 2-digit HS codes
print("\n===== 2-DIGIT HS CODE ANALYSIS =====")
train_hs2_counts = train_df['HS_2digit'].value_counts()
test_hs2_counts = test_df['HS_2digit'].value_counts()
valid_hs2_counts = valid_df['HS_2digit'].value_counts()

print(f"Number of unique 2-digit HS codes in train set: {len(train_hs2_counts)}")
print(f"Number of unique 2-digit HS codes in test set: {len(test_hs2_counts)}")
print(f"Number of unique 2-digit HS codes in validation set: {len(valid_hs2_counts)}")

# Find 2-digit codes in one set but not others
train_unique = set(train_hs2_counts.index)
test_unique = set(test_hs2_counts.index)
valid_unique = set(valid_hs2_counts.index)

print(f"\n2-digit HS codes in train but not in test: {len(train_unique - test_unique)}")
print(f"2-digit HS codes in test but not in train: {len(test_unique - train_unique)}")
print(f"2-digit HS codes in all three sets: {len(train_unique.intersection(test_unique, valid_unique))}")

# Create a comprehensive table of all 2-digit HS codes and their counts across datasets
print("\n===== DETAILED 2-DIGIT HS CODE COUNTS ACROSS DATASETS =====")
# Get all unique 2-digit codes across all datasets
all_hs2_codes = sorted(list(train_unique.union(test_unique, valid_unique)))

# Create a DataFrame to hold the counts
hs2_counts_df = pd.DataFrame(index=all_hs2_codes, columns=['Train', 'Test', 'Validation', 'Total'])

# Fill in the counts for each dataset
for code in all_hs2_codes:
    hs2_counts_df.loc[code, 'Train'] = train_hs2_counts.get(code, 0)
    hs2_counts_df.loc[code, 'Test'] = test_hs2_counts.get(code, 0)
    hs2_counts_df.loc[code, 'Validation'] = valid_hs2_counts.get(code, 0)
    hs2_counts_df.loc[code, 'Total'] = (hs2_counts_df.loc[code, 'Train'] + 
                                       hs2_counts_df.loc[code, 'Test'] + 
                                       hs2_counts_df.loc[code, 'Validation'])

# Sort by total count descending
hs2_counts_df = hs2_counts_df.sort_values('Total', ascending=False)

# Instead of printing the entire DataFrame (which gets truncated), just show the top 20
print("\nTop 20 most common 2-digit HS codes across all datasets:")
print(hs2_counts_df.head(20))

# Save to CSV for easier analysis
hs2_counts_df.to_csv('hs_2digit_counts.csv')
print(f"\nDetailed 2-digit HS code counts for all {len(hs2_counts_df)} codes saved to 'hs_2digit_counts.csv'")
print("Open this CSV file to view the complete list of 2-digit HS codes and their counts.")

# Top 10 most common 2-digit HS codes in each set
print("\n===== TOP 10 MOST COMMON 2-DIGIT HS CODES =====")
print("\nTrain set:")
print(train_hs2_counts.head(10))
print("\nTest set:")
print(test_hs2_counts.head(10))
print("\nValidation set:")
print(valid_hs2_counts.head(10))

# Distribution of product description lengths
print("\n===== PRODUCT DESCRIPTION LENGTH STATISTICS =====")
train_desc_len = train_df['PRODUCT DESCRIPTION'].str.len()
test_desc_len = test_df['PRODUCT DESCRIPTION'].str.len()
valid_desc_len = valid_df['PRODUCT DESCRIPTION'].str.len()

print("\nTrain set:")
print(f"Min length: {train_desc_len.min()}")
print(f"Max length: {train_desc_len.max()}")
print(f"Mean length: {train_desc_len.mean():.2f}")
print(f"Median length: {train_desc_len.median()}")

print("\nTest set:")
print(f"Min length: {test_desc_len.min()}")
print(f"Max length: {test_desc_len.max()}")
print(f"Mean length: {test_desc_len.mean():.2f}")
print(f"Median length: {test_desc_len.median()}")

print("\nValidation set:")
print(f"Min length: {valid_desc_len.min()}")
print(f"Max length: {valid_desc_len.max()}")
print(f"Mean length: {valid_desc_len.mean():.2f}")
print(f"Median length: {valid_desc_len.median()}")

# Full 6-digit HS code analysis
print("\n===== 6-DIGIT HS CODE ANALYSIS =====")
train_hs6_counts = train_df['HS CODE6'].value_counts()
test_hs6_counts = test_df['HS CODE6'].value_counts()
valid_hs6_counts = valid_df['HS CODE6'].value_counts()

print(f"Number of unique 6-digit HS codes in train set: {len(train_hs6_counts)}")
print(f"Number of unique 6-digit HS codes in test set: {len(test_hs6_counts)}")
print(f"Number of unique 6-digit HS codes in validation set: {len(valid_hs6_counts)}")

# Distribution of samples per 6-digit HS code
print("\n===== SAMPLES PER 6-DIGIT HS CODE =====")
print("\nTrain set:")
print(f"Min samples: {train_hs6_counts.min()}")
print(f"Max samples: {train_hs6_counts.max()}")
print(f"Mean samples: {train_hs6_counts.mean():.2f}")
print(f"Median samples: {train_hs6_counts.median()}")

print("\nTest set:")
print(f"Min samples: {test_hs6_counts.min()}")
print(f"Max samples: {test_hs6_counts.max()}")
print(f"Mean samples: {test_hs6_counts.mean():.2f}")
print(f"Median samples: {test_hs6_counts.median()}")

print("\nValidation set:")
print(f"Min samples: {valid_hs6_counts.min()}")
print(f"Max samples: {valid_hs6_counts.max()}")
print(f"Mean samples: {valid_hs6_counts.mean():.2f}")
print(f"Median samples: {valid_hs6_counts.median()}")

# Sample records
print("\n===== SAMPLE RECORDS =====")
print("\nTrain set sample (5 records):")
print(train_df[['PRODUCT DESCRIPTION', 'HS CODE6', 'HS_2digit']].head(5))

print("\nTest set sample (5 records):")
print(test_df[['PRODUCT DESCRIPTION', 'HS CODE6', 'HS_2digit']].head(5))

print("\nValidation set sample (5 records):")
print(valid_df[['PRODUCT DESCRIPTION', 'HS CODE6', 'HS_2digit']].head(5))

# Optional: Create visualizations
try:
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Plot distribution of top 20 2-digit HS codes in train set
    plt.subplot(2, 1, 1)
    train_top20 = train_hs2_counts.head(20)
    sns.barplot(x=train_top20.index, y=train_top20.values)
    plt.title('Top 20 2-Digit HS Codes in Training Set')
    plt.xlabel('2-Digit HS Code')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Plot distribution of product description lengths
    plt.subplot(2, 1, 2)
    plt.hist(train_desc_len, bins=50, alpha=0.7, label='Train')
    plt.hist(test_desc_len, bins=50, alpha=0.5, label='Test')
    plt.hist(valid_desc_len, bins=50, alpha=0.3, label='Valid')
    plt.title('Distribution of Product Description Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hs_code_analysis.png')
    print("\nVisualization saved as 'hs_code_analysis.png'")
except Exception as e:
    print(f"\nCouldn't create visualization: {e}")

print("\nAnalysis complete!")
