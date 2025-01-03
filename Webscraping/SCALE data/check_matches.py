import pandas as pd

# Load tracking data
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
tracking_containers = tracking['CONTAINER NUMBER'].str.upper().unique()
print(f"Unique containers in tracking data: {len(tracking_containers)}")
print("\nSample tracking containers:")
print(tracking_containers[:5])

# Load and process import data
print("\nLoading import data...")
ig_files = ['Results_ig20_2712.csv', 'Results_ig27_2712.csv', 'Results_ig2025_31.xlsx']
import_containers = set()

for file in ig_files:
    print(f"\nProcessing {file}")
    if file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file, low_memory=False)
    
    # Split container numbers and add to set
    for container_str in df['containerNumber']:
        if pd.notna(container_str):
            containers = str(container_str).strip().split('\n')
            for container in containers:
                container = container.strip().upper()
                if container:
                    import_containers.add(container)
    
    print(f"Total unique containers so far: {len(import_containers)}")

# Convert to list for easier viewing
import_containers = sorted(list(import_containers))
print("\nSample import containers:")
print(import_containers[:5])

# Check matches
tracking_set = set(tracking_containers)
import_set = set(import_containers)

matches = tracking_set.intersection(import_set)
print(f"\nMatching Statistics:")
print(f"Total tracking containers: {len(tracking_set)}")
print(f"Total import containers: {len(import_set)}")
print(f"Number of matches: {len(matches)}")

# Show some containers that didn't match
unmatched = tracking_set - import_set
print("\nSample unmatched tracking containers:")
print(list(unmatched)[:5])

# Save matches to file for verification
with open('matched_containers.txt', 'w') as f:
    f.write('\n'.join(sorted(matches)))

print("\nSaved matched containers to matched_containers.txt")
