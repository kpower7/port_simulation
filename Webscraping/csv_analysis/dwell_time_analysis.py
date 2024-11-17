import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style for better visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Read the merged results
df = pd.read_csv('merged_results.csv')

# Print unique ISO codes first
print("\nUnique ISO Codes:")
print(df['IsoCode'].unique())

# Print column names
print("\nAvailable columns:")
print(df.columns.tolist())

# Convert dates to datetime, making them timezone naive
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce', utc=True).dt.tz_localize(None)
df['GateOutDate'] = pd.to_datetime(df['GateOutDate'], errors='coerce', utc=True).dt.tz_localize(None)

# Check containers with relevant dates
has_discharge = df['DischargedDate'].notna()
has_gateout = df['GateOutDate'].notna()

# Basic statistics as before
print("\nDwell Time Analysis:")
print(f"Total containers: {len(df)}")
print(f"Containers with discharge date: {has_discharge.sum()}")
print(f"Containers with gate out date: {has_gateout.sum()}")

print("\nStatus Analysis:")
print(f"Containers with both dates (completed cycle): {(has_discharge & has_gateout).sum()}")
print(f"Containers discharged but not yet gated out: {(has_discharge & ~has_gateout).sum()}")
print(f"Containers with gate out but no discharge date (possible data issue): {(~has_discharge & has_gateout).sum()}")
print(f"Containers with neither date: {(~has_discharge & ~has_gateout).sum()}")

# For containers with both dates, calculate dwell time
both_dates = df[has_discharge & has_gateout].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

# Categorize containers as reefer or dry based on ISO code
def categorize_container(iso_code):
    if pd.isna(iso_code):
        return 'Unknown'
    iso_code = str(iso_code).upper()
    if 'R' in iso_code:
        return 'Reefer'
    return 'Dry'

both_dates['container_category'] = both_dates['IsoCode'].apply(categorize_container)

# Calculate current dwell time for containers still in port
still_in_port = df[has_discharge & ~has_gateout].copy()
current_time = pd.Timestamp('2024-12-15 11:37:56', tz='America/New_York').tz_localize(None)
still_in_port['current_dwell_days'] = (current_time - still_in_port['DischargedDate']).dt.total_seconds() / (24*60*60)
still_in_port['container_category'] = still_in_port['IsoCode'].apply(categorize_container)

# Print unique ISO codes and their categories for verification
print("\nISO Code Categories:")
iso_types = pd.DataFrame({
    'ISO Code': df['IsoCode'].unique(),
    'Category': [categorize_container(iso_code) for iso_code in df['IsoCode'].unique()]
})
print(iso_types.sort_values('Category'))

# Detailed Statistics by Container Category
print("\nDwell Time Statistics by Container Category (Completed Containers):")
category_stats = both_dates.groupby('container_category').agg({
    'dwell_time_days': ['count', 'mean', 'std', 'min', 'median', 'max'],
    'CONTAINER NUMBER': 'count'
}).round(2)
print(category_stats)

# Overall Statistics
print("\nOverall Dwell Time Statistics (days):")
stats_description = both_dates['dwell_time_days'].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95])
print(stats_description)

# Calculate skewness and kurtosis
print("\nDistribution Statistics:")
print(f"Skewness: {stats.skew(both_dates['dwell_time_days']):.2f}")
print(f"Kurtosis: {stats.kurtosis(both_dates['dwell_time_days']):.2f}")

# Create visualizations
plt.figure(figsize=(20, 15))

# 1. Box plot comparison
plt.subplot(2, 2, 1)
sns.boxplot(data=both_dates, x='container_category', y='dwell_time_days')
plt.title('Dwell Time Distribution by Container Category')
plt.xlabel('Container Category')
plt.ylabel('Dwell Time (days)')

# 2. Violin plot comparison
plt.subplot(2, 2, 2)
sns.violinplot(data=both_dates, x='container_category', y='dwell_time_days')
plt.title('Dwell Time Distribution (Violin Plot)')
plt.xlabel('Container Category')
plt.ylabel('Dwell Time (days)')

# 3. Histogram comparison
plt.subplot(2, 2, 3)
for category in ['Reefer', 'Dry']:
    subset = both_dates[both_dates['container_category'] == category]['dwell_time_days']
    plt.hist(subset[subset <= subset.quantile(0.95)], 
             bins=50, alpha=0.5, label=category, density=True)
plt.title('Dwell Time Distribution (95th percentile)')
plt.xlabel('Dwell Time (days)')
plt.ylabel('Density')
plt.legend()

# 4. Cumulative distribution
plt.subplot(2, 2, 4)
for category in ['Reefer', 'Dry']:
    subset = both_dates[both_dates['container_category'] == category]['dwell_time_days']
    plt.hist(subset, bins=50, density=True, cumulative=True, 
             histtype='step', label=category, alpha=0.8)
plt.title('Cumulative Distribution of Dwell Times')
plt.xlabel('Dwell Time (days)')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()

# Additional plot for current dwell times
plt.figure(figsize=(12, 6))
sns.boxplot(data=still_in_port, x='container_category', y='current_dwell_days')
plt.title('Current Dwell Times by Container Category (Containers Still in Port)')
plt.xlabel('Container Category')
plt.ylabel('Current Dwell Time (days)')
plt.show()

# Print summary statistics for containers still in port
print("\nCurrent Dwell Time Statistics by Container Category (Containers Still in Port):")
current_category_stats = still_in_port.groupby('container_category').agg({
    'current_dwell_days': ['count', 'mean', 'std', 'min', 'median', 'max'],
    'CONTAINER NUMBER': 'count'
}).round(2)
print(current_category_stats)

# Statistical significance test
reefer_times = both_dates[both_dates['container_category'] == 'Reefer']['dwell_time_days']
dry_times = both_dates[both_dates['container_category'] == 'Dry']['dwell_time_days']
t_stat, p_value = stats.ttest_ind(reefer_times, dry_times)
print("\nStatistical Test (Reefer vs Dry):")
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")
