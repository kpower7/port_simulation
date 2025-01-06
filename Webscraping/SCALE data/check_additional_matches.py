import pandas as pd

# First load the unmatched containers from tracking data
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
tracking_containers = set(tracking['CONTAINER NUMBER'].str.upper().unique())
print(f"Total tracking containers: {len(tracking_containers)}")

# Load the containers we already matched
print("\nLoading previously matched containers...")
with open('matched_containers.txt', 'r') as f:
    matched_containers = set(line.strip() for line in f)
print(f"Previously matched containers: {len(matched_containers)}")

# Get unmatched containers
unmatched = tracking_containers - matched_containers
print(f"Unmatched containers: {len(unmatched)}")

# Load the new file
print("\nChecking new import file...")
new_file = r"C:\Users\k_pow\Downloads\Results_ig4 (1).csv"
new_imports = pd.read_csv(new_file, low_memory=False)

# Process container numbers
new_containers = set()
for container_str in new_imports['containerNumber']:
    if pd.notna(container_str):
        containers = str(container_str).strip().split('\n')
        for container in containers:
            container = container.strip().upper()
            if container:
                new_containers.add(container)

print(f"Containers in new file: {len(new_containers)}")

# Check for new matches
new_matches = unmatched.intersection(new_containers)
print(f"\nNew matches found: {len(new_matches)}")
print("\nSample of new matches:")
print(list(new_matches)[:5])

# Calculate total coverage
total_matches = len(matched_containers) + len(new_matches)
print(f"\nTotal coverage with new file:")
print(f"Total tracking containers: {len(tracking_containers)}")
print(f"Total matched containers: {total_matches}")
print(f"Coverage percentage: {(total_matches/len(tracking_containers))*100:.1f}%")
