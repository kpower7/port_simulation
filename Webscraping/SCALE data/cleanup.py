import os
import shutil

# Create archive directory if it doesn't exist
archive_dir = 'archive'
if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)

# Files to keep
keep_files = {
    'merged_results_final_december.csv',  # New final merged dataset
    'circular_comparison_new.py',
    'cycle_analysis_new.py',
    'visualize_data.py',
    'circular_comparison.png',
    'container_volumes.png',
    'volume_comparison.png',
    'dry_cycle_analysis.png',
    'reefer_cycle_analysis.png',
    'Results_20241220.csv',
    'Results_20241227.csv',
    'Results_ig20_2712.csv',
    'Results_ig27_2712.csv',
    'Results_20250103.csv',      # New APM data
    'Results_ig2025_31.xlsx',    # New IG data
    '2025 SCALE Challenge.zip',
    'cleanup.py'  # Keep this script
}

# Move all other files to archive
for filename in os.listdir('.'):
    if os.path.isfile(filename) and filename not in keep_files:
        try:
            shutil.move(filename, os.path.join(archive_dir, filename))
            print(f"Moved {filename} to archive/")
        except Exception as e:
            print(f"Error moving {filename}: {e}")

print("\nCleanup complete!")
print("\nKept files:")
for filename in sorted(os.listdir('.')):
    if os.path.isfile(filename):
        print(f"- {filename}")

print("\nArchived files:")
for filename in sorted(os.listdir(archive_dir)):
    if os.path.isfile(os.path.join(archive_dir, filename)):
        print(f"- {filename}")
