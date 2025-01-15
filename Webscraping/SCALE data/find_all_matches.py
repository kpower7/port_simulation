import pandas as pd

# Load tracking data first
print("Loading tracking data...")
tracking = pd.read_csv('Port of NYNJ - Container Data.csv', low_memory=False)
tracking['CONTAINER NUMBER'] = tracking['CONTAINER NUMBER'].str.upper()
tracking_containers = set(tracking['CONTAINER NUMBER'].dropna())
print(f"Total tracking containers: {len(tracking_containers)}")

# Function to extract containers from a DataFrame
def get_containers_from_df(df):
    containers = set()
    for container_str in df['containerNumber']:
        if pd.notna(container_str):
            for container in str(container_str).strip().split('\n'):
                container = container.strip().upper()
                if container:
                    containers.add(container)
    return containers

# Load all possible import files
import_files = [
    r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\combined_results.csv",
    'Results_ig20_2712.csv',
    'Results_ig27_2712.csv',
    'Results_ig2025_31.xlsx',
    r"C:\Users\k_pow\Downloads\Results_ig4 (1).csv"
]

# Process each file and track matches
all_import_containers = set()
matches_by_file = {}
unmatched = tracking_containers.copy()

for file in import_files:
    print(f"\nProcessing {file}...")
    try:
        if file.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, low_memory=False)
        
        # Get containers from this file
        file_containers = get_containers_from_df(df)
        all_import_containers.update(file_containers)
        
        # Find new matches from this file
        new_matches = unmatched.intersection(file_containers)
        matches_by_file[file] = new_matches
        unmatched -= new_matches
        
        print(f"Containers in file: {len(file_containers)}")
        print(f"New matches found: {len(new_matches)}")
        print(f"Remaining unmatched: {len(unmatched)}")
        
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

print("\nFinal Summary:")
print("--------------")
print(f"Total tracking containers: {len(tracking_containers)}")
print(f"Total import containers found: {len(all_import_containers)}")
print(f"Total matched containers: {len(tracking_containers - unmatched)}")
print(f"Remaining unmatched: {len(unmatched)}")

print("\nMatches by file:")
for file, matches in matches_by_file.items():
    print(f"{file}: {len(matches)} matches")

if unmatched:
    print("\nSample of unmatched containers:")
    print(list(unmatched)[:5])

# Save the matches for verification
with open('all_matches.txt', 'w') as f:
    f.write("Matches by file:\n")
    f.write("--------------\n")
    for file, matches in matches_by_file.items():
        f.write(f"\n{file}:\n")
        f.write("\n".join(sorted(matches)))
        f.write("\n")
    
    if unmatched:
        f.write("\nUnmatched containers:\n")
        f.write("-----------------\n")
        f.write("\n".join(sorted(unmatched)))
