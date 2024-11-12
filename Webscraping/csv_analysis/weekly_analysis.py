import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv(r'C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_apm_all.csv')

# Convert Discharge Date to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce')

# Create week-only column
df['DischargeWeek'] = df['DischargedDate'].dt.strftime('%Y-%U')

# Get counts by week
weekly_counts = df['DischargeWeek'].value_counts().sort_index()

print("\nWeekly Discharge Statistics:")
print("\nTop 10 busiest weeks for discharges (Year-Week format):")
print(weekly_counts.nlargest(10))

# Create a simple plot
plt.figure(figsize=(12, 6))
weekly_counts.plot(kind='bar')
plt.title('Container Discharges by Week')
plt.xlabel('Year-Week')
plt.ylabel('Number of Containers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
