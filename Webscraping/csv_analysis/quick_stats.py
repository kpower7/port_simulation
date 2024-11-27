import pandas as pd
import numpy as np

# Read the merged results
df = pd.read_csv('merged_results.csv')

# Convert dates to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce', utc=True).dt.tz_localize(None)
df['GateOutDate'] = pd.to_datetime(df['GateOutDate'], errors='coerce', utc=True).dt.tz_localize(None)

# Calculate dwell times for containers with both dates
both_dates = df[df['DischargedDate'].notna() & df['GateOutDate'].notna()].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

# Categorize containers
def categorize_container(iso_code):
    if pd.isna(iso_code):
        return 'Unknown'
    iso_code = str(iso_code).upper()
    return 'Reefer' if 'R' in iso_code else 'Dry'

both_dates['container_category'] = both_dates['IsoCode'].apply(categorize_container)

# Calculate percentage of containers gated out within 1 day
for category in ['Dry', 'Reefer']:
    category_data = both_dates[both_dates['container_category'] == category]
    total_containers = len(category_data)
    one_day_containers = len(category_data[category_data['dwell_time_days'] <= 1])
    percentage = (one_day_containers / total_containers) * 100
    
    print(f"\n{category} Containers:")
    print(f"Total containers: {total_containers}")
    print(f"Containers gated out within 1 day: {one_day_containers}")
    print(f"Percentage within 1 day: {percentage:.1f}%")
