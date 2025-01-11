import pandas as pd

# Load tracking data
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].str.upper()
print(f"Tracking data:")
print(f"Total rows: {len(tracking)}")
print(f"Unique containers: {tracking['CONTAINER NUMBER'].nunique()}")

# Load combined data
print("\nLoading combined data...")
combined = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\combined_results.csv", low_memory=False)

# Process container numbers from combined data
combined_containers = set()
for container_str in combined['containerNumber']:
    if pd.notna(container_str):
        containers = str(container_str).strip().split('\n')
        for container in containers:
            container = container.strip().upper()
            if container:
                combined_containers.add(container)

print(f"\nCombined data unique containers: {len(combined_containers)}")

# Check overlap
tracking_containers = set(tracking['CONTAINER NUMBER'].dropna().str.upper())
matches = tracking_containers.intersection(combined_containers)
print(f"\nOverlap analysis:")
print(f"Tracking containers: {len(tracking_containers)}")
print(f"Combined containers: {len(combined_containers)}")
print(f"Matching containers: {len(matches)}")

# Show sample of unmatched containers
unmatched = tracking_containers - combined_containers
print(f"\nUnmatched tracking containers: {len(unmatched)}")
if unmatched:
    print("Sample unmatched:")
    print(list(unmatched)[:5])

# Check if there are any invalid characters or formatting issues
print("\nChecking for formatting issues...")
print("Sample tracking container formats:")
print(tracking['CONTAINER NUMBER'].head())
print("\nSample combined container formats:")
print(list(combined_containers)[:5])
