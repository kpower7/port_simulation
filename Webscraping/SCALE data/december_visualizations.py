import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Set style for better visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]

# Read the data
print("Reading December data...")
df = pd.read_csv('C:\\Users\\k_pow\\OneDrive\\Documents\\Capstone\\Webscraping\\SCALE data\\merged_results_final_december.csv')

# Convert dates to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'])
df['GateOutDate'] = pd.to_datetime(df['GateOutDate'])
df['arrivalDate'] = pd.to_datetime(df['arrivalDate'])

# Create date-only columns
df['discharge_date'] = df['DischargedDate'].dt.date
df['gateout_date'] = df['GateOutDate'].dt.date
df['arrival_date'] = df['arrivalDate'].dt.date

# 1. Daily Container Counts
plt.figure(figsize=(15, 8))
# Get daily counts
discharge_counts = df.groupby('discharge_date').size()
gateout_counts = df.groupby('gateout_date').size()

# Create date range for December
date_range = pd.date_range(start='2024-12-01', end='2024-12-31', freq='D')
discharge_counts = discharge_counts.reindex(date_range.date, fill_value=0)
gateout_counts = gateout_counts.reindex(date_range.date, fill_value=0)

plt.plot(date_range, discharge_counts.values, 'b-', label='Discharge', marker='o')
plt.plot(date_range, gateout_counts.values, 'g-', label='Gate Out', marker='o')

plt.title('Daily Container Volumes - December 2024')
plt.xlabel('Date')
plt.ylabel('Number of Containers')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('december_daily_volumes.png')
print("Saved daily volumes plot")

# 2. Container Type Distribution by Week
plt.figure(figsize=(12, 6))
df['week'] = df['DischargedDate'].dt.isocalendar().week
df['container_type'] = df['IsoCode'].apply(lambda x: 'Reefer' if 'R' in str(x).upper() else 'Dry')

weekly_type_counts = df.groupby(['week', 'container_type']).size().unstack(fill_value=0)
weekly_type_counts.plot(kind='bar', stacked=True)
plt.title('Container Types by Week - December 2024')
plt.xlabel('Week of December')
plt.ylabel('Number of Containers')
plt.legend(title='Container Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('december_weekly_types.png')
print("Saved weekly container types plot")

# 3. Dwell Time Distribution
plt.figure(figsize=(12, 6))
both_dates = df[df['DischargedDate'].notna() & df['GateOutDate'].notna()].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

# Filter out extreme outliers for visualization
dwell_times = both_dates[both_dates['dwell_time_days'] <= both_dates['dwell_time_days'].quantile(0.95)]

plt.hist(dwell_times['dwell_time_days'], bins=50, alpha=0.7)
plt.axvline(dwell_times['dwell_time_days'].mean(), color='r', linestyle='dashed', label=f'Mean: {dwell_times["dwell_time_days"].mean():.1f} days')
plt.axvline(dwell_times['dwell_time_days'].median(), color='g', linestyle='dashed', label=f'Median: {dwell_times["dwell_time_days"].median():.1f} days')

plt.title('Container Dwell Time Distribution - December 2024')
plt.xlabel('Dwell Time (days)')
plt.ylabel('Number of Containers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('december_dwell_times.png')
print("Saved dwell time distribution plot")

# 4. Hourly Pattern Heatmaps
plt.figure(figsize=(15, 12))

# Discharge Heatmap
plt.subplot(2, 1, 1)
df['discharge_hour'] = df['DischargedDate'].dt.hour
df['discharge_day'] = df['DischargedDate'].dt.day

discharge_activity = pd.pivot_table(
    df, 
    values='ContainerId',
    index='discharge_hour',
    columns='discharge_day',
    aggfunc='count',
    fill_value=0
)

sns.heatmap(discharge_activity, cmap='YlOrRd')
plt.title('Container Discharge Activity Heatmap - December 2024')
plt.xlabel('Day of Month')
plt.ylabel('Hour of Day')

# Gate Out Heatmap
plt.subplot(2, 1, 2)
df_gateout = df[df['GateOutDate'].notna()].copy()
df_gateout['gateout_hour'] = df_gateout['GateOutDate'].dt.hour
df_gateout['gateout_day'] = df_gateout['GateOutDate'].dt.day

gateout_activity = pd.pivot_table(
    df_gateout, 
    values='ContainerId',
    index='gateout_hour',
    columns='gateout_day',
    aggfunc='count',
    fill_value=0
)

sns.heatmap(gateout_activity, cmap='YlOrRd')
plt.title('Container Gate Out Activity Heatmap - December 2024')
plt.xlabel('Day of Month')
plt.ylabel('Hour of Day')

plt.tight_layout()
plt.savefig('december_activity_heatmaps.png')
print("Saved activity heatmaps")

# 5. Hourly Averages Plot
plt.figure(figsize=(15, 6))

# Calculate average containers per hour
hourly_discharge = df.groupby('discharge_hour')['ContainerId'].count() / 31  # 31 days in December
hourly_gateout = df_gateout.groupby('gateout_hour')['ContainerId'].count() / 31

plt.plot(hourly_discharge.index, hourly_discharge.values, 'b-', label='Discharge', marker='o')
plt.plot(hourly_gateout.index, hourly_gateout.values, 'g-', label='Gate Out', marker='o')

plt.title('Average Hourly Container Activity - December 2024')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Containers')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(range(24))
plt.tight_layout()
plt.savefig('december_hourly_averages.png')
print("Saved hourly averages plot")

# Print hourly statistics
print("\nHourly Statistics:")
print("\nDischarge Activity:")
peak_discharge_hour = hourly_discharge.idxmax()
print(f"Peak hour: {peak_discharge_hour:02d}:00 ({hourly_discharge[peak_discharge_hour]:.1f} containers/day)")
print(f"Top 5 busiest hours:")
for hour, count in hourly_discharge.nlargest(5).items():
    print(f"  {hour:02d}:00 - {count:.1f} containers/day")

print("\nGate Out Activity:")
peak_gateout_hour = hourly_gateout.idxmax()
print(f"Peak hour: {peak_gateout_hour:02d}:00 ({hourly_gateout[peak_gateout_hour]:.1f} containers/day)")
print(f"Top 5 busiest hours:")
for hour, count in hourly_gateout.nlargest(5).items():
    print(f"  {hour:02d}:00 - {count:.1f} containers/day")

# Calculate percentage of activity by time of day
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning (6-12)'
    elif 12 <= hour < 18:
        return 'Afternoon (12-18)'
    elif 18 <= hour < 22:
        return 'Evening (18-22)'
    else:
        return 'Night (22-6)'

df['discharge_time_of_day'] = df['discharge_hour'].apply(get_time_of_day)
df_gateout['gateout_time_of_day'] = df_gateout['gateout_hour'].apply(get_time_of_day)

print("\nActivity Distribution by Time of Day:")
print("\nDischarge Activity:")
discharge_tod = df.groupby('discharge_time_of_day')['ContainerId'].count()
for tod, count in discharge_tod.items():
    print(f"{tod}: {count:,} containers ({count/len(df)*100:.1f}%)")

print("\nGate Out Activity:")
gateout_tod = df_gateout.groupby('gateout_time_of_day')['ContainerId'].count()
for tod, count in gateout_tod.items():
    print(f"{tod}: {count:,} containers ({count/len(df_gateout)*100:.1f}%)")

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total Containers: {len(df):,}")
print(f"Containers with complete dwell times: {len(both_dates):,}")
print("\nDaily Averages:")
print(f"Average daily discharges: {discharge_counts.mean():.1f}")
print(f"Average daily gate outs: {gateout_counts.mean():.1f}")
print("\nPeak Days:")
print(f"Peak discharge day: December {discharge_counts.idxmax().day} ({discharge_counts.max():,} containers)")
print(f"Peak gate out day: December {gateout_counts.idxmax().day} ({gateout_counts.max():,} containers)")
print("\nDwell Time Statistics:")
print(f"Average dwell time: {both_dates['dwell_time_days'].mean():.1f} days")
print(f"Median dwell time: {both_dates['dwell_time_days'].median():.1f} days")
print(f"95th percentile dwell time: {both_dates['dwell_time_days'].quantile(0.95):.1f} days")

# 6. Continuous Hourly Timeline
plt.figure(figsize=(20, 8))

# Create hourly bins for December
start_date = pd.Timestamp('2024-12-01')
end_date = pd.Timestamp('2025-01-01')
hourly_range = pd.date_range(start=start_date, end=end_date, freq='H')

# Count containers for each hour
discharge_hourly = df.groupby(pd.Timestamp('2024-12-01') + pd.to_timedelta(df['DischargedDate'].dt.strftime('%j').astype(int) - 335, unit='D') + 
                            pd.to_timedelta(df['DischargedDate'].dt.hour, unit='H'))['ContainerId'].count()

gateout_hourly = df_gateout.groupby(pd.Timestamp('2024-12-01') + pd.to_timedelta(df_gateout['GateOutDate'].dt.strftime('%j').astype(int) - 335, unit='D') + 
                                   pd.to_timedelta(df_gateout['GateOutDate'].dt.hour, unit='H'))['ContainerId'].count()

# Reindex to include all hours
discharge_hourly = discharge_hourly.reindex(hourly_range, fill_value=0)
gateout_hourly = gateout_hourly.reindex(hourly_range, fill_value=0)

# Create hour index (0 to 744)
hour_index = range(len(hourly_range))

# Plot
plt.plot(hour_index, discharge_hourly.values, 'b-', label='Discharge', alpha=0.7)
plt.plot(hour_index, gateout_hourly.values, 'g-', label='Gate Out', alpha=0.7)

# Customize plot
plt.title('Hourly Container Activity - December 2024')
plt.xlabel('Hour of Month (0-744)')
plt.ylabel('Number of Containers')
plt.grid(True, alpha=0.3)
plt.legend()

# Add vertical lines for weeks
for week in range(1, 5):
    plt.axvline(x=week * 168, color='gray', linestyle='--', alpha=0.5)
    plt.text(week * 168, plt.ylim()[1], f'Week {week}', rotation=0, ha='right', va='bottom')

plt.tight_layout()
plt.savefig('december_continuous_hourly.png')
print("Saved continuous hourly timeline")

# Print statistics for continuous timeline
print("\nContinuous Hourly Statistics:")
print("\nDischarge Activity:")
print(f"Peak hour of month: Hour {discharge_hourly.values.argmax()} ({discharge_hourly.max()} containers)")
print(f"Average containers per hour: {discharge_hourly.mean():.1f}")
print(f"Hours with zero activity: {(discharge_hourly == 0).sum()}")

print("\nGate Out Activity:")
print(f"Peak hour of month: Hour {gateout_hourly.values.argmax()} ({gateout_hourly.max()} containers)")
print(f"Average containers per hour: {gateout_hourly.mean():.1f}")
print(f"Hours with zero activity: {(gateout_hourly == 0).sum()}")

# Calculate rolling averages
discharge_rolling = discharge_hourly.rolling(window=24).mean()
gateout_rolling = gateout_hourly.rolling(window=24).mean()

# Plot with rolling averages
plt.figure(figsize=(20, 8))
plt.plot(hour_index, discharge_hourly.values, 'b-', label='Discharge', alpha=0.3)
plt.plot(hour_index, discharge_rolling.values, 'b-', label='Discharge (24h avg)', linewidth=2)
plt.plot(hour_index, gateout_hourly.values, 'g-', label='Gate Out', alpha=0.3)
plt.plot(hour_index, gateout_rolling.values, 'g-', label='Gate Out (24h avg)', linewidth=2)

plt.title('Hourly Container Activity with 24-hour Moving Average - December 2024')
plt.xlabel('Hour of Month (0-744)')
plt.ylabel('Number of Containers')
plt.grid(True, alpha=0.3)
plt.legend()

# Add vertical lines for weeks
for week in range(1, 5):
    plt.axvline(x=week * 168, color='gray', linestyle='--', alpha=0.5)
    plt.text(week * 168, plt.ylim()[1], f'Week {week}', rotation=0, ha='right', va='bottom')

plt.tight_layout()
plt.savefig('december_continuous_hourly_smoothed.png')
print("Saved smoothed continuous hourly timeline")

# 7. Train vs Truck Analysis
print("\nTrain vs Truck Analysis:")

# Check unique modality values
print("\nUnique Modality values:")
print(df['Modality'].value_counts(dropna=False))

# Check unique carrier codes
print("\nUnique Carrier Codes:")
print(df['carrierCode'].value_counts().head(10))

# Calculate dwell times for analysis
both_dates = df[df['DischargedDate'].notna() & df['GateOutDate'].notna()].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

# Convert columns to string
both_dates['Modality'] = both_dates['Modality'].astype(str)
both_dates['carrierCode'] = both_dates['carrierCode'].astype(str)

# Identify train vs truck based on carrier code
both_dates['transport_mode'] = 'Truck'  # Default to truck
rail_carriers = ['CN', 'CP', 'CSX', 'NS', 'UP', 'RAIL', 'BNSF']
rail_pattern = '|'.join(rail_carriers)
both_dates.loc[both_dates['carrierCode'].str.contains(rail_pattern, case=False, na=False), 'transport_mode'] = 'Train'

# Print transport mode counts
print("\nTransport mode counts:")
print(both_dates['transport_mode'].value_counts())
print("\nSample of Train carriers:")
print(both_dates[both_dates['transport_mode'] == 'Train']['carrierCode'].value_counts().head())

# Basic statistics by transport mode
stats_columns = {
    'dwell_time_days': ['count', 'mean', 'median', 'std', lambda x: x.quantile(0.95)],
    'ContainerId': ['count']
}

transport_stats = both_dates.groupby('transport_mode').agg(stats_columns).round(2)

# Flatten column names
transport_stats.columns = ['dwell_count', 'dwell_mean', 'dwell_median', 'dwell_std', 'dwell_95th', 'container_count']
print("\nTransport Mode Statistics:")
print(transport_stats)

# Calculate percentages
total_containers = both_dates['ContainerId'].count()
for mode in both_dates['transport_mode'].unique():
    count = both_dates[both_dates['transport_mode'] == mode]['ContainerId'].count()
    print(f"{mode} percentage: {count/total_containers*100:.1f}%")

# Plot dwell time distributions
plt.figure(figsize=(12, 6))
for mode in ['Train', 'Truck']:
    mode_data = both_dates[both_dates['transport_mode'] == mode]['dwell_time_days']
    mode_data = mode_data[mode_data <= mode_data.quantile(0.95)]  # Remove outliers for visualization
    plt.hist(mode_data, bins=50, alpha=0.5, label=mode, density=True)

plt.title('Dwell Time Distribution by Transport Mode - December 2024')
plt.xlabel('Dwell Time (days)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('december_transport_mode_dwell.png')
print("Saved transport mode dwell time distribution")

# Daily patterns by transport mode
daily_transport = both_dates.groupby(['transport_mode', both_dates['GateOutDate'].dt.date])['ContainerId'].count().unstack(0, fill_value=0)
plt.figure(figsize=(15, 6))
daily_transport.plot(marker='o')
plt.title('Daily Container Volumes by Transport Mode - December 2024')
plt.xlabel('Date')
plt.ylabel('Number of Containers')
plt.grid(True, alpha=0.3)
plt.legend(title='Transport Mode')
plt.tight_layout()
plt.savefig('december_daily_transport.png')
print("Saved daily transport mode volumes")

# Container type analysis by transport mode
type_transport = both_dates.groupby(['transport_mode', 'container_type'])['ContainerId'].count().unstack(0, fill_value=0)
print("\nContainer Types by Transport Mode:")
print(type_transport)
print("\nPercentages within each transport mode:")
print((type_transport / type_transport.sum() * 100).round(1))

# Time of day analysis
both_dates['hour'] = both_dates['GateOutDate'].dt.hour
hourly_transport = both_dates.groupby(['transport_mode', 'hour'])['ContainerId'].count().unstack(0, fill_value=0)

plt.figure(figsize=(12, 6))
hourly_transport.plot()
plt.title('Hourly Pattern by Transport Mode - December 2024')
plt.xlabel('Hour of Day')
plt.ylabel('Total Containers')
plt.grid(True, alpha=0.3)
plt.legend(title='Transport Mode')
plt.tight_layout()
plt.savefig('december_hourly_transport.png')
print("Saved hourly transport mode patterns")

# Calculate dwell time statistics by container type and transport mode
dwell_stats = both_dates.groupby(['container_type', 'transport_mode'])['dwell_time_days'].agg([
    'count', 'mean', 'median', 'std', lambda x: x.quantile(0.95)
]).round(2)
dwell_stats.columns = ['count', 'mean_dwell', 'median_dwell', 'std_dwell', '95th_percentile']
print("\nDwell Time Statistics by Container Type and Transport Mode:")
print(dwell_stats)
