import pandas as pd

# Read the CSV files
df1 = pd.read_csv('Results_dec1-3.csv')
df2 = pd.read_csv('Results_ig4.csv')

# Combine the dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

# Sort by date column (assuming there's a date column - adjust the column name as needed)
combined_df = combined_df.sort_values(by='arrivalDate')

# Save the combined and sorted dataframe
combined_df.to_csv('combined_results.csv', index=False)

print("Number of rows in first file:", len(df1))
print("Number of rows in second file:", len(df2))
print("Number of rows in combined file:", len(combined_df))
