import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\Capstone\Webscraping\Results_ig4.csv")

# Define terminal-carrier mapping
terminal_carriers = {
    'APM Terminals': ['MAERSK LINE'],
    'Maher Terminals': ['HAPAG LLOYD A G', 'MSC MEDITTERRANEAN SHIPPING COMPANY SA', 
                       'HAMBURG SUD', 'HYUNDAI', 'SAFMARINE', 'SEALAND', 
                       'ZIM ISRAEL NAVIGATION CO LTD'],
    'GCT Bayonne': ['ORATEL NETWORKS', 'ONE'],
    'GCT New York': ['CHINA OCEAN SHIPPING COMPANY', 'ORIENT OVERSEAS CONTAINER LINE LTD', 
                     'EVERGREEN LINE', 'ACL', 'APL', 'COSCO'],
    'Port Newark Container Terminal': ['COMPAGNIE MARITIME D-AFFRETEMENT', 'CMA CGM'],
    'Red Hook Container Terminal': ['CMA CGM']
}

def assign_terminal(carrier):
    if pd.isna(carrier):
        return 'Unknown'
    carrier_str = str(carrier).upper()
    for terminal, carriers in terminal_carriers.items():
        if any(c.upper() in carrier_str for c in carriers):
            return terminal
    return 'Unknown'

# Add terminal column to dataframe
df['Terminal'] = df['carrierName'].apply(assign_terminal)

# Analysis of container distribution by terminal
terminal_analysis = df['Terminal'].value_counts()
print("\n=== Container Distribution by Terminal ===")
print(terminal_analysis)

# Analysis of average shipment weight by terminal
terminal_weights = df.groupby('Terminal')['grossWeightKg'].agg(['mean', 'count', 'sum']).round(2)
print("\n=== Shipment Weight Analysis by Terminal ===")
print("mean = average weight in kg")
print("count = number of containers")
print("sum = total weight in kg")
print(terminal_weights)

# Most common container types by terminal
print("\n=== Most Common Container Types by Terminal ===")
for terminal in df['Terminal'].unique():
    container_types = df[df['Terminal'] == terminal]['containerType'].value_counts().head(3)
    print(f"\n{terminal}:")
    print(container_types)

# Carrier distribution within terminals
print("\n=== Carrier Distribution within Terminals ===")
for terminal in df['Terminal'].unique():
    carriers = df[df['Terminal'] == terminal]['carrierName'].value_counts().head(3)
    print(f"\n{terminal}:")
    print(carriers)
