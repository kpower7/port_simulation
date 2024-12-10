import pandas as pd
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')

# Read the cleaned data
print("Reading cleaned data...")
df = pd.read_csv('merged_results_final_clean.csv')

# Convert dates to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'])
df['arrivalDate'] = pd.to_datetime(df['arrivalDate'])

# Create date-only columns
df['discharge_date'] = df['DischargedDate'].dt.date
df['arrival_date'] = df['arrivalDate'].dt.date

# Get daily counts
discharge_counts = df.groupby('discharge_date').size()
arrival_counts = df.groupby('arrival_date').size()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle('Container Volume Analysis - December 2024', fontsize=16, y=0.95)

# Plot 1: Discharge Date Volumes
discharge_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Daily Container Volumes by Discharge Date')
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of Containers')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Add value labels on top of bars
for i, v in enumerate(discharge_counts):
    ax1.text(i, v, str(v), ha='center', va='bottom')

# Plot 2: Arrival Date Volumes
arrival_counts.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Daily Container Volumes by Arrival Date')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Containers')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Add value labels on top of bars
for i, v in enumerate(arrival_counts):
    ax2.text(i, v, str(v), ha='center', va='bottom')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('container_volumes.png', dpi=300, bbox_inches='tight')
print("Saved visualization to container_volumes.png")

# Create a second figure for a comparison line plot
plt.figure(figsize=(15, 8))
plt.plot(discharge_counts.index, discharge_counts.values, 'b-', label='Discharge Date', linewidth=2, marker='o')
plt.plot(arrival_counts.index, arrival_counts.values, 'g-', label='Arrival Date', linewidth=2, marker='o')
plt.title('Container Volumes Comparison - December 2024', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Number of Containers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Save the comparison plot
plt.tight_layout()
plt.savefig('volume_comparison.png', dpi=300, bbox_inches='tight')
print("Saved comparison visualization to volume_comparison.png")

# Print some statistics
print("\nKey Statistics:")
print(f"Total containers: {len(df):,}")
print(f"\nDischarge Date Stats:")
print(f"Average daily volume: {discharge_counts.mean():,.1f}")
print(f"Maximum daily volume: {discharge_counts.max():,} (on {discharge_counts.idxmax()})")
print(f"Minimum daily volume: {discharge_counts.min():,} (on {discharge_counts.idxmin()})")

print(f"\nArrival Date Stats:")
print(f"Average daily volume: {arrival_counts.mean():,.1f}")
print(f"Maximum daily volume: {arrival_counts.max():,} (on {arrival_counts.idxmax()})")
print(f"Minimum daily volume: {arrival_counts.min():,} (on {arrival_counts.idxmin()})")
