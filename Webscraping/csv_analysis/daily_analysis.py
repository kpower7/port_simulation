import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv(r'C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_apm_all.csv')

# Convert Discharge Date to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce')

# Create date-only column
df['DischargeDay'] = df['DischargedDate'].dt.date

# Get counts by date
daily_counts = df['DischargeDay'].value_counts().sort_index()

# Print statistics
print("\nDaily Discharge Statistics:")
print(f"Earliest discharge date: {df['DischargedDate'].min().date()}")
print(f"Latest discharge date: {df['DischargedDate'].max().date()}")
print(f"\nNumber of containers not yet discharged: {df['DischargedDate'].isna().sum()}")
print(f"Percentage of containers not discharged: {(df['DischargedDate'].isna().sum()/len(df)*100):.2f}%")

print("\nTop 10 busiest dates for discharges:")
print(daily_counts.nlargest(10))

# Create a simple plot
plt.figure(figsize=(12, 6))
daily_counts.plot(kind='bar')
plt.title('Container Discharges by Date')
plt.xlabel('Date')
plt.ylabel('Number of Containers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
