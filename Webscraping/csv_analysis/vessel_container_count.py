import pandas as pd

# Read the CSV file
df = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\csv_analysis\merged_results.csv")

# Print column names
print("Available columns in the CSV:")
print(df.columns.tolist())

# Print first few rows to understand the data structure
print("\nFirst few rows of the data:")
print(df.head())

# Group by vessel name and count containers
vessel_counts = df.groupby('VesselName').size().reset_index(name='container_count')

# Sort by container count in descending order
vessel_counts = vessel_counts.sort_values('container_count', ascending=False)

# Display results
print("\nVessel Container Counts:")
print("-" * 50)
for _, row in vessel_counts.iterrows():
    print(f"{row['VesselName']}: {row['container_count']} containers")

print("\nTotal unique vessels:", len(vessel_counts))
print("Total containers:", vessel_counts['container_count'].sum())
