import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

# Read the CSV file
df = pd.read_csv(r'C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_apm_all.csv')

# Convert Discharge Date to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce')

# Create date and week columns
df['DischargeDay'] = df['DischargedDate'].dt.date
df['DischargeWeek'] = df['DischargedDate'].dt.strftime('%Y-%U')

# Get counts
daily_counts = df['DischargeDay'].value_counts().sort_index()
weekly_counts = df['DischargeWeek'].value_counts().sort_index()

# Print statistics
print("\nDischarge Statistics:")
print(f"Earliest discharge date: {df['DischargedDate'].min().date()}")
print(f"Latest discharge date: {df['DischargedDate'].max().date()}")
print(f"Number of containers not yet discharged: {df['DischargedDate'].isna().sum()}")
print(f"Percentage of containers not discharged: {(df['DischargedDate'].isna().sum()/len(df)*100):.2f}%")

print("\nTop 10 busiest dates for discharges:")
print(daily_counts.nlargest(10))

print("\nTop 10 busiest weeks for discharges (Year-Week format):")
print(weekly_counts.nlargest(10))

# Create daily plot
plt.figure(figsize=(12, 6))
daily_counts.plot(kind='bar')
plt.title('Container Discharges by Date')
plt.xlabel('Date')
plt.ylabel('Number of Containers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create weekly plot
plt.figure(figsize=(12, 6))
weekly_counts.plot(kind='bar')
plt.title('Container Discharges by Week')
plt.xlabel('Year-Week')
plt.ylabel('Number of Containers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
