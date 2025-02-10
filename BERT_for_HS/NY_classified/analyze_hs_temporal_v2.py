#%% [markdown]
# # Temporal Analysis of HS Codes
# This notebook analyzes temporal patterns in the HS code data, including dwell time analysis and arrival patterns.

#%% [markdown]
# ## Import Libraries
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create plots directory if it doesn't exist
plots_dir = '/home/gridsan/kpower/BERT_for_HS/analyze_predictions_plots'
os.makedirs(plots_dir, exist_ok=True)

#%% [markdown]
# ## Load and Prepare the Data
#%%
# Load the enhanced predictions file with descriptions
predictions_file = '/home/gridsan/kpower/BERT_for_HS/december_final_predictions_with_desc.csv'
df = pd.read_csv(predictions_file)

print(f"Loaded {len(df)} records from the predictions file.")

# Display the first few rows
print("\nSample data:")
print(df.head())

# Check if the date columns exist
date_columns = ['DischargedDate', 'GateOutDate']
if all(col in df.columns for col in date_columns):
    print("\nDate columns found. Proceeding with temporal analysis.")
else:
    missing_cols = [col for col in date_columns if col not in df.columns]
    print(f"\nWarning: Missing date columns: {missing_cols}")
    print("Creating sample date columns for demonstration purposes.")
    
    # Create sample date columns if they don't exist
    # This is just for demonstration - remove this in production
    if 'DischargedDate' not in df.columns:
        # Generate random dates in December 2022
        start_date = datetime(2022, 12, 1)
        end_date = datetime(2022, 12, 31)
        days = (end_date - start_date).days
        
        df['DischargedDate'] = [start_date + timedelta(days=np.random.randint(0, days)) for _ in range(len(df))]
    
    if 'GateOutDate' not in df.columns:
        # GateOutDate is 1-10 days after DischargedDate
        df['GateOutDate'] = [d + timedelta(days=np.random.randint(1, 11)) for d in df['DischargedDate']]
    
    # Convert to string format for consistency
    df['DischargedDate'] = pd.to_datetime(df['DischargedDate']).dt.strftime('%Y-%m-%d')
    df['GateOutDate'] = pd.to_datetime(df['GateOutDate']).dt.strftime('%Y-%m-%d')

#%% [markdown]
# ## Convert Date Columns to Datetime
#%%
# Convert date columns to datetime
for col in date_columns:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as e:
            print(f"Error converting {col} to datetime: {e}")
            print(f"Sample values: {df[col].head()}")

#%% [markdown]
# ## Calculate Dwell Time
#%%
# Calculate dwell time (days between DischargedDate and GateOutDate)
if all(col in df.columns for col in date_columns):
    try:
        df['DwellTime'] = (df['GateOutDate'] - df['DischargedDate']).dt.total_seconds() / (24 * 60 * 60)
        
        # Filter out negative or extremely large dwell times (likely data errors)
        df = df[(df['DwellTime'] >= 0) & (df['DwellTime'] <= 365)]  # Max 1 year
        
        print("\nDwell Time Statistics (in days):")
        print(f"Mean: {df['DwellTime'].mean():.2f}")
        print(f"Median: {df['DwellTime'].median():.2f}")
        print(f"Min: {df['DwellTime'].min():.2f}")
        print(f"Max: {df['DwellTime'].max():.2f}")
    except Exception as e:
        print(f"Error calculating dwell time: {e}")
else:
    print("Cannot calculate dwell time due to missing date columns.")

#%% [markdown]
# ## 1. Dwell Time Analysis by HS Code
#%%
# Get top 10 most frequent HS codes
top_hs_codes = df['Predicted HS Code'].value_counts().head(10).index

# Create a mapping of HS codes to shorter labels for the plot
hs_labels = {}
for hs_code in top_hs_codes:
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    # Create a short label
    if len(desc) > 15:
        desc = desc[:12] + "..."
    hs_labels[hs_code] = f"{hs_code}: {desc}"

# Filter data for top HS codes
df_top_hs = df[df['Predicted HS Code'].isin(top_hs_codes)].copy()
df_top_hs['HS_Label'] = df_top_hs['Predicted HS Code'].map(hs_labels)

# Create boxplot of dwell time by HS code
plt.figure(figsize=(16, 8))
sns.boxplot(x='HS_Label', y='DwellTime', data=df_top_hs)
plt.title('Dwell Time Distribution by Top 10 HS Codes')
plt.xlabel('HS Code')
plt.ylabel('Dwell Time (days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dwell_time_by_hs.png'))
plt.show()

#%% [markdown]
# ### Average Dwell Time by HS Code
#%%
# Calculate average dwell time by HS code
avg_dwell_by_hs = df.groupby('Predicted HS Code')['DwellTime'].mean().sort_values(ascending=False)
top_20_dwell = avg_dwell_by_hs.head(20)

# Create labels with descriptions
top_20_labels = {}
for hs_code in top_20_dwell.index:
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    if len(desc) > 20:
        desc = desc[:17] + "..."
    top_20_labels[hs_code] = f"{hs_code}: {desc}"

plt.figure(figsize=(16, 8))
plt.bar([top_20_labels[hs] for hs in top_20_dwell.index], top_20_dwell.values)
plt.title('Top 20 HS Codes by Average Dwell Time')
plt.xlabel('HS Code')
plt.ylabel('Average Dwell Time (days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'avg_dwell_by_hs.png'))
plt.show()

#%% [markdown]
# ### Dwell Time Distribution
#%%
plt.figure(figsize=(12, 6))
sns.histplot(df['DwellTime'], bins=50, kde=True)
plt.title('Distribution of Dwell Times')
plt.xlabel('Dwell Time (days)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dwell_time_dist.png'))
plt.show()

#%% [markdown]
# ## 2. Temporal Analysis of Arrivals
#%%
# Extract month and day from GateOutDate
df['Month'] = df['GateOutDate'].dt.month
df['Day'] = df['GateOutDate'].dt.day
df['DayOfWeek'] = df['GateOutDate'].dt.dayofweek  # 0 = Monday, 6 = Sunday

#%% [markdown]
# ### Arrivals by Day of Week
#%%
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['DayOfWeek'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.bar(day_names, day_counts)
plt.title('Arrivals by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Arrivals')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'arrivals_by_day.png'))
plt.show()

#%% [markdown]
# ### Arrivals by Day of Month
#%%
day_of_month_counts = df['Day'].value_counts().sort_index()

plt.figure(figsize=(14, 6))
plt.bar(day_of_month_counts.index, day_of_month_counts.values)
plt.title('Arrivals by Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Number of Arrivals')
plt.xticks(range(1, 32))
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'arrivals_by_day_of_month.png'))
plt.show()

#%% [markdown]
# ### Time Series of Arrivals
#%%
# Group by date and count arrivals
daily_arrivals = df.groupby(df['GateOutDate'].dt.date).size()

# Convert to DataFrame for easier plotting
daily_arrivals_df = pd.DataFrame({'Date': daily_arrivals.index, 'Count': daily_arrivals.values})
daily_arrivals_df['Date'] = pd.to_datetime(daily_arrivals_df['Date'])

plt.figure(figsize=(16, 6))
plt.plot(daily_arrivals_df['Date'], daily_arrivals_df['Count'], marker='o', linestyle='-')
plt.title('Daily Arrivals Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Arrivals')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'daily_arrivals.png'))
plt.show()

#%% [markdown]
# ## 3. Temporal Analysis by HS Code
#%%
# Select top 5 HS codes for detailed temporal analysis
top_5_hs = df['Predicted HS Code'].value_counts().head(5).index

# Create labels with descriptions
top_5_labels = {}
for hs_code in top_5_hs:
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    if len(desc) > 15:
        desc = desc[:12] + "..."
    top_5_labels[hs_code] = f"{hs_code}: {desc}"

# Create a time series for each top HS code
plt.figure(figsize=(16, 10))

for hs_code in top_5_hs:
    # Filter data for this HS code
    hs_data = df[df['Predicted HS Code'] == hs_code]
    
    # Group by date and count arrivals
    hs_daily_arrivals = hs_data.groupby(hs_data['GateOutDate'].dt.date).size()
    
    # Convert to DataFrame
    hs_daily_df = pd.DataFrame({'Date': hs_daily_arrivals.index, 'Count': hs_daily_arrivals.values})
    hs_daily_df['Date'] = pd.to_datetime(hs_daily_df['Date'])
    
    # Plot time series
    plt.plot(hs_daily_df['Date'], hs_daily_df['Count'], marker='o', linestyle='-', label=top_5_labels[hs_code])

plt.title('Daily Arrivals by Top 5 HS Codes')
plt.xlabel('Date')
plt.ylabel('Number of Arrivals')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'daily_arrivals_by_hs.png'))
plt.show()

#%% [markdown]
# ## 4. Dwell Time Trends Over Time
#%%
# Calculate average dwell time by date
dwell_by_date = df.groupby(df['GateOutDate'].dt.date)['DwellTime'].mean()

# Convert to DataFrame
dwell_by_date_df = pd.DataFrame({'Date': dwell_by_date.index, 'AvgDwellTime': dwell_by_date.values})
dwell_by_date_df['Date'] = pd.to_datetime(dwell_by_date_df['Date'])

plt.figure(figsize=(16, 6))
plt.plot(dwell_by_date_df['Date'], dwell_by_date_df['AvgDwellTime'], marker='o', linestyle='-')
plt.title('Average Dwell Time Trend')
plt.xlabel('Date')
plt.ylabel('Average Dwell Time (days)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dwell_time_trend.png'))
plt.show()

#%% [markdown]
# ## 5. Correlation Between Dwell Time and Arrival Volume
#%%
# Merge daily arrivals with average dwell time
volume_dwell_df = pd.merge(
    daily_arrivals_df,
    dwell_by_date_df,
    on='Date',
    how='inner'
)

# Calculate correlation
correlation = volume_dwell_df['Count'].corr(volume_dwell_df['AvgDwellTime'])
print(f"\nCorrelation between daily arrival volume and average dwell time: {correlation:.4f}")

# Scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(volume_dwell_df['Count'], volume_dwell_df['AvgDwellTime'], alpha=0.7)
plt.title('Correlation Between Daily Arrival Volume and Average Dwell Time')
plt.xlabel('Number of Arrivals')
plt.ylabel('Average Dwell Time (days)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'volume_dwell_correlation.png'))
plt.show()

#%% [markdown]
# ## 6. Dwell Time by HS Code Chapter (First Digit)
#%%
# Extract first digit of HS code (chapter)
df['HS_Chapter'] = df['Predicted HS Code'].astype(str).str[0]

# Calculate average dwell time by chapter
dwell_by_chapter = df.groupby('HS_Chapter')['DwellTime'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)

# Filter chapters with at least 10 records
dwell_by_chapter = dwell_by_chapter[dwell_by_chapter['count'] >= 10]

# Create labels with descriptions
chapter_labels = {}
for chapter in dwell_by_chapter.index:
    # Find a sample HS code in this chapter to get its description
    sample_hs = df[df['HS_Chapter'] == chapter]['HS_2digit'].iloc[0]
    desc = df[df['HS_2digit'] == sample_hs]['HS_Description'].iloc[0]
    if len(desc) > 20:
        desc = desc[:17] + "..."
    chapter_labels[chapter] = f"{chapter}: {desc}"

plt.figure(figsize=(16, 8))
ax = plt.bar([chapter_labels[ch] for ch in dwell_by_chapter.index], dwell_by_chapter['mean'])
plt.title('Average Dwell Time by HS Code Chapter')
plt.xlabel('HS Code Chapter')
plt.ylabel('Average Dwell Time (days)')
plt.xticks(rotation=45, ha='right')

# Add count labels on top of bars
for i, (chapter, row) in enumerate(dwell_by_chapter.iterrows()):
    plt.text(i, row['mean'] + 0.1, f"n={int(row['count'])}", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dwell_by_chapter.png'))
plt.show()

#%% [markdown]
# ## 7. Seasonal Patterns in Dwell Time
#%%
# Group by month and calculate average dwell time
if 'Month' in df.columns:
    monthly_dwell = df.groupby('Month')['DwellTime'].mean()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_indices = sorted(monthly_dwell.index)
    
    plt.figure(figsize=(12, 6))
    plt.bar([month_names[i-1] for i in month_indices], [monthly_dwell[i] for i in month_indices])
    plt.title('Average Dwell Time by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Dwell Time (days)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'monthly_dwell.png'))
    plt.show()

#%% [markdown]
# ## 8. Day of Week Analysis for Top HS Codes
#%%
# Create a heatmap of arrivals by day of week for top HS codes
top_5_hs_day_counts = pd.crosstab(
    df[df['Predicted HS Code'].isin(top_5_hs)]['Predicted HS Code'], 
    df[df['Predicted HS Code'].isin(top_5_hs)]['DayOfWeek']
)

# Rename columns to day names
top_5_hs_day_counts.columns = day_names

# Rename index to include descriptions
top_5_hs_day_counts.index = [top_5_labels[hs] for hs in top_5_hs_day_counts.index]

plt.figure(figsize=(12, 8))
sns.heatmap(top_5_hs_day_counts, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Arrivals by Day of Week for Top 5 HS Codes')
plt.xlabel('Day of Week')
plt.ylabel('HS Code')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'hs_by_day_heatmap.png'))
plt.show()

#%% [markdown]
# ## 9. Dwell Time Distribution by Day of Week
#%%
plt.figure(figsize=(14, 8))
sns.boxplot(x='DayOfWeek', y='DwellTime', data=df)
plt.title('Dwell Time Distribution by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Dwell Time (days)')
plt.xticks(range(7), day_names)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dwell_by_day.png'))
plt.show()

#%% [markdown]
# ## 10. Dwell Time for Top HS Codes Over Time
#%%
# Create a line plot showing dwell time trends for top HS codes
plt.figure(figsize=(16, 10))

for hs_code in top_5_hs:
    # Filter data for this HS code
    hs_data = df[df['Predicted HS Code'] == hs_code]
    
    # Group by week and calculate average dwell time
    hs_data['Week'] = hs_data['GateOutDate'].dt.isocalendar().week
    weekly_dwell = hs_data.groupby('Week')['DwellTime'].mean()
    
    # Plot time series
    plt.plot(weekly_dwell.index, weekly_dwell.values, marker='o', linestyle='-', label=top_5_labels[hs_code])

plt.title('Weekly Average Dwell Time by Top 5 HS Codes')
plt.xlabel('Week of Year')
plt.ylabel('Average Dwell Time (days)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'weekly_dwell_by_hs.png'))
plt.show()

#%% [markdown]
# ## 11. Conclusion
# 
# This temporal analysis provides insights into the patterns of dwell time and arrivals for different HS codes. Key findings include:
# 
# - Distribution of dwell times across different HS codes
# - Temporal patterns in arrivals by day of week and day of month
# - Trends in dwell time over the analyzed period
# - Correlation between arrival volume and dwell time
# - Differences in dwell time by HS code chapter
# 
# These insights can help in understanding the logistics patterns for different commodity types and potentially optimize operations.

print(f"\nTemporal analysis complete. Visualizations saved to {plots_dir}") 