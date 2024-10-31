import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_ig4.csv")

def analyze_dataframe(df):
    print("\n=== Basic DataFrame Information ===")
    print("\nDataFrame Shape:", df.shape)
    print(f"\nTotal rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    
    print("\n=== Column Information ===")
    print("\nColumns:", df.columns.tolist())
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_info[missing_info['Missing Values'] > 0])
    
    print("\n=== Duplicate Rows Analysis ===")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    print(f"Percentage of duplicate rows: {(duplicates/len(df))*100:.2f}%")
    
    print("\n=== Numerical Columns Summary ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(df[numerical_cols].describe())
    
    print("\n=== Categorical Columns Summary ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nUnique values in {col}: {df[col].nunique()}")
        print("\nTop 5 most common values:")
        print(df[col].value_counts().head())
        print(df.iloc[61935])

def analyze_shipping_lines(df):
    print("\n=== Top 10 Shipping Lines Analysis ===")
    top_carriers = df['carrierName'].value_counts().head(10)
    print("\nTop 10 Shipping Lines by Number of Shipments:")
    print(top_carriers)
    
    # Analysis for each top carrier
    for carrier in top_carriers.index:
        carrier_df = df[df['carrierName'] == carrier]
        print(f"\nDetailed Analysis for {carrier}:")
        print(f"Total Shipments: {len(carrier_df)}")
        print(f"Most Common US Ports: {carrier_df['usPort'].value_counts().head(1).index[0]}")
        print(f"Most Common Container Types: {carrier_df['containerType'].value_counts().head(1).index[0]}")
        print(f"Average Shipment Weight (kg): {carrier_df['grossWeightKg'].mean():.2f}")

# Run the analysis
analyze_dataframe(df)
analyze_shipping_lines(df)
