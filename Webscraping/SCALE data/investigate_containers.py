import pandas as pd
import numpy as np

# Load a sample from tracking tool data
tracking_df = pd.read_csv('Results_20241227.csv')
print("\nTracking Tool Container Numbers (first 5):")
print(tracking_df['CONTAINER NUMBER'].head())
print("\nTracking Tool Total Containers:", len(tracking_df['CONTAINER NUMBER'].unique()))

# Load a sample from import data
import_df = pd.read_csv('Results_ig27_2712.csv')
print("\nImport Records Container Numbers (first 5):")
print(import_df['containerNumber'].head())
print("\nImport Records Total Containers:", len(import_df['containerNumber'].unique()))

# Check for exact matches
tracking_containers = set(tracking_df['CONTAINER NUMBER'].dropna())
import_containers = set(import_df['containerNumber'].dropna())

print("\nSample from Tracking but not in Import:")
print(list(tracking_containers - import_containers)[:5])
print("\nSample from Import but not in Tracking:")
print(list(import_containers - tracking_containers)[:5])

# Check for case differences or spacing
print("\nTracking Container Number Example Format:")
print(tracking_df['CONTAINER NUMBER'].iloc[0])
print("Length:", len(tracking_df['CONTAINER NUMBER'].iloc[0]))

print("\nImport Container Number Example Format:")
print(import_df['containerNumber'].iloc[0])
print("Length:", len(import_df['containerNumber'].iloc[0]))
