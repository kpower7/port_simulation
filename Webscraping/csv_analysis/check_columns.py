import pandas as pd

# Read the CSV file
df = pd.read_csv(r'C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_apm_all.csv')

# Display column names
print("Column names in the CSV file:")
for col in df.columns:
    print(col)
