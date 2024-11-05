import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Read the combined CSV file
df = pd.read_csv('combined_results.csv')

# Check for duplicates
total_rows = len(df)
duplicates = df.duplicated().sum()
df_clean = df.drop_duplicates()
rows_after_dedup = len(df_clean)

print("\n=== DUPLICATE ANALYSIS ===")
print(f"Total rows: {total_rows}")
print(f"Duplicate rows: {duplicates}")
print(f"Rows after removing duplicates: {rows_after_dedup}")

# Basic statistics
print("\n=== TOP VESSELS ===")
print(df_clean['vesselName'].value_counts().head(10))

print("\n=== TOP CARRIERS ===")
print(df_clean['carrierName'].value_counts().head(10))

print("\n=== TOP FOREIGN PORTS ===")
print(df_clean['foreignPort'].value_counts().head(10))

print("\n=== TOP US PORTS ===")
print(df_clean['usPort'].value_counts().head(10))

print("\n=== TOP CONTAINER TYPES ===")
print(df_clean['containerType'].value_counts().head(10))

print("\n=== TOP COUNTRIES OF ORIGIN ===")
print(df_clean['countryOfOrigin'].value_counts().head(10))

print("\n=== CONTAINER COUNT BY MONTH ===")
# Clean date data by removing invalid dates
df_clean = df_clean[df_clean['arrivalDate'] != '-NOT AVAILABLE-']
df_clean['arrivalDate'] = pd.to_datetime(df_clean['arrivalDate'])
monthly_counts = df_clean.groupby(df_clean['arrivalDate'].dt.strftime('%Y-%m')).size()
print(monthly_counts)

print("\n=== WEIGHT STATISTICS (in kg) ===")
weight_stats = df_clean['grossWeightKg'].describe()
print(weight_stats)

print("\n=== AVERAGE CONTAINERS PER VESSEL ===")
containers_per_vessel = df_clean.groupby('vesselName').size()
print(f"Mean containers per vessel: {containers_per_vessel.mean():.2f}")
print(f"Median containers per vessel: {containers_per_vessel.median():.2f}")
print(f"Max containers per vessel: {containers_per_vessel.max()}")

# Save cleaned data
df_clean.to_csv('cleaned_results.csv', index=False)
print("\nCleaned data saved to 'cleaned_results.csv'")
