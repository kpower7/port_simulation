import pandas as pd

# Read the merged results
df = pd.read_csv('merged_results.csv')

# Convert dates to datetime, making them timezone naive
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce', utc=True).dt.tz_localize(None)
df['arrivalDate'] = pd.to_datetime(df['arrivalDate'], errors='coerce', utc=True).dt.tz_localize(None)

# Check containers with discharge dates
has_discharge = df['DischargedDate'].notna()
has_arrival = df['arrivalDate'].notna()

print("Date Analysis:")
print(f"Total containers: {len(df)}")
print(f"Containers with discharge date: {has_discharge.sum()}")
print(f"Containers with arrival date: {has_arrival.sum()}")

print("\nOverlap Analysis:")
print(f"Containers with both dates: {(has_discharge & has_arrival).sum()}")
print(f"Containers with discharge but no arrival: {(has_discharge & ~has_arrival).sum()}")
print(f"Containers with arrival but no discharge: {(~has_discharge & has_arrival).sum()}")
print(f"Containers with neither date: {(~has_discharge & ~has_arrival).sum()}")

# For containers with both dates, calculate the difference
both_dates = df[has_discharge & has_arrival].copy()
both_dates['days_difference'] = (both_dates['DischargedDate'] - both_dates['arrivalDate']).dt.total_seconds() / (24*60*60)

print("\nFor containers with both dates:")
print("Days between arrival and discharge:")
print(both_dates['days_difference'].describe())

# Sample of containers with both dates
print("\nSample of containers with both dates (sorted by days_difference):")
sample_cols = ['CONTAINER NUMBER', 'DischargedDate', 'arrivalDate', 'days_difference', 'VesselName']
print(both_dates[sample_cols].sort_values('days_difference').head())
print("\nLongest delays:")
print(both_dates[sample_cols].sort_values('days_difference', ascending=False).head())
