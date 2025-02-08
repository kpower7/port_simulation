#%% [markdown]
# # HS Code Prediction Analysis
# This notebook analyzes the distribution and patterns of predicted HS codes.

#%% [markdown]
# ## Import Libraries
#%% 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
import os

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create plots directory if it doesn't exist
plots_dir = '/home/gridsan/kpower/BERT_for_HS/analyze_predictions_plots'
os.makedirs(plots_dir, exist_ok=True)

#%% [markdown]
# ## Load and Examine the Data
#%%
# Load the enhanced predictions file with descriptions
predictions_file = '/home/gridsan/kpower/BERT_for_HS/december_final_predictions_with_desc.csv'
df = pd.read_csv(predictions_file)

print(f"Loaded {len(df)} records from the predictions file.")

# Display the first few rows
print("\nSample data:")
print(df.head())

#%% [markdown]
# ## 1. Distribution of Predicted HS Codes
#%%
print("\n--- Distribution of Predicted HS Codes ---")
hs_counts = df['Predicted HS Code'].value_counts()
top_hs_codes = hs_counts.head(20)
print(f"Number of unique HS codes predicted: {len(hs_counts)}")
print("\nTop 20 most frequent HS codes:")
for hs_code in top_hs_codes.index:
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    print(f"{hs_code} ({desc}): {hs_counts[hs_code]}")

#%% [markdown]
# ## 2. Visualize Top 20 Most Frequent HS Codes
#%%
# Get descriptions for top HS codes
top_hs_with_desc = {}
for hs_code in top_hs_codes.index:
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    # Truncate description if too long
    if len(desc) > 30:
        desc = desc[:27] + "..."
    top_hs_with_desc[f"{hs_code}: {desc}"] = hs_counts[hs_code]

plt.figure(figsize=(16, 10))
plt.bar(top_hs_with_desc.keys(), top_hs_with_desc.values())
plt.title('Top 20 Most Frequent HS Codes')
plt.xlabel('HS Code and Description')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'top_hs_codes.png'))
plt.show()

#%% [markdown]
# ## 3. Pie Chart of Top 10 HS Codes vs Others
#%%
# Get descriptions for top 10 HS codes
top_10_hs = hs_counts.head(10)
top_10_with_desc = []
for hs_code in top_10_hs.index:
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    # Truncate description if too long
    if len(desc) > 20:
        desc = desc[:17] + "..."
    top_10_with_desc.append(f"{hs_code}: {desc}")

others_count = hs_counts[10:].sum()
sizes = list(top_10_hs) + [others_count]
labels = top_10_with_desc + ['Others']

plt.figure(figsize=(14, 14))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Top 10 HS Codes vs Others')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'hs_codes_pie.png'))
plt.show()

#%% [markdown]
# ## 4. Word Frequency Analysis for Top HS Codes
#%%
def extract_keywords(text):
    if pd.isna(text):
        return []
    # Convert to lowercase and remove punctuation
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words and filter out short words
    words = [word for word in text.split() if len(word) > 3]
    return words

# Get the top 5 HS codes for detailed analysis
top_5_hs = hs_counts.head(5).index

#%% [markdown]
# ### Create bar charts of most common words for each top HS code
#%%
for hs_code in top_5_hs:
    # Get description for this HS code
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    
    # Filter data for this HS code
    hs_data = df[df['Predicted HS Code'] == hs_code]
    
    # Extract all words from product descriptions
    all_words = []
    for desc_text in hs_data['productDescription']:
        all_words.extend(extract_keywords(desc_text))
    
    # Count word frequencies
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(20)
    
    print(f"\n--- Most Common Words for HS Code {hs_code} ({desc}) ---")
    for word, count in most_common:
        print(f"{word}: {count}")
    
    # Create bar chart of most common words
    if most_common:
        words, counts = zip(*most_common)
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(words)), counts, align='center')
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency')
        plt.title(f'Most Common Words for HS Code {hs_code} ({desc})')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'common_words_hs_{hs_code}.png'))
        plt.show()

#%% [markdown]
# ## 5. Analysis of Product Description Lengths
#%%
# Calculate description lengths
df['description_length'] = df['productDescription'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

# Basic statistics about description lengths
print("\n--- Description Length Statistics ---")
print(f"Mean description length: {df['description_length'].mean():.2f} characters")
print(f"Median description length: {df['description_length'].median()} characters")
print(f"Min description length: {df['description_length'].min()} characters")
print(f"Max description length: {df['description_length'].max()} characters")

#%% [markdown]
# ### Distribution of description lengths
#%%
plt.figure(figsize=(12, 6))
sns.histplot(df['description_length'], bins=50, kde=True)
plt.title('Distribution of Product Description Lengths')
plt.xlabel('Description Length (characters)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'description_length_dist.png'))
plt.show()

#%% [markdown]
# ### Description lengths by top HS codes
#%%
# Get top 10 HS codes for description length analysis
top_10_hs = hs_counts.head(10).index
df_top10 = df[df['Predicted HS Code'].isin(top_10_hs)]

# Create a mapping of HS codes to shorter labels for the plot
hs_labels = {}
for hs_code in top_10_hs:
    desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
    # Create a short label
    if len(desc) > 15:
        desc = desc[:12] + "..."
    hs_labels[hs_code] = f"{hs_code}: {desc}"

# Map the HS codes to the shorter labels
df_top10['HS_Label'] = df_top10['Predicted HS Code'].map(hs_labels)

plt.figure(figsize=(14, 8))
sns.boxplot(x='HS_Label', y='description_length', data=df_top10)
plt.title('Distribution of Product Description Lengths by HS Code (Top 10)')
plt.xlabel('HS Code')
plt.ylabel('Description Length (characters)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'description_length_by_hs.png'))
plt.show()

#%% [markdown]
# ## 6. Summary Statistics for the Dataset
#%%
print("\n--- Summary Statistics ---")
print(f"Total number of records: {len(df)}")
print(f"Number of unique HS codes: {len(hs_counts)}")
most_common_hs = hs_counts.index[0]
most_common_desc = df[df['Predicted HS Code'] == most_common_hs]['HS_Description'].iloc[0]
print(f"Most common HS code: {most_common_hs} ({most_common_desc}) - appears {hs_counts.iloc[0]} times")
print(f"Average product description length: {df['description_length'].mean():.2f} characters")
print(f"Median product description length: {df['description_length'].median()} characters")

#%% [markdown]
# ## 7. Analysis of HS Code Structure
#%%
# Check if we can extract first and second digits
try:
    # Convert to string if not already
    df['Predicted HS Code'] = df['Predicted HS Code'].astype(str)
    
    # Check if the values look like HS codes (at least 2 digits)
    if df['Predicted HS Code'].str.len().min() >= 2:
        # Extract first and second digits
        df['first_digit'] = df['Predicted HS Code'].str[0]
        df['second_digit'] = df['Predicted HS Code'].str[1]
        
        # Create a cross-tabulation
        heatmap_data = pd.crosstab(df['first_digit'], df['second_digit'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Distribution of HS Codes (First Digit vs Second Digit)')
        plt.xlabel('Second Digit')
        plt.ylabel('First Digit')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'hs_code_heatmap.png'))
        plt.show()
        
        #%% [markdown]
        # ### Distribution by HS Chapter (First Digit)
        #%%
        # Get chapter descriptions
        chapter_counts = df['first_digit'].value_counts().sort_index()
        chapter_labels = {}
        
        for digit in chapter_counts.index:
            # Find any HS code starting with this digit to get its description
            sample_hs = df[df['first_digit'] == digit]['HS_2digit'].iloc[0]
            desc = df[df['HS_2digit'] == sample_hs]['HS_Description'].iloc[0]
            # Truncate description if too long
            if len(desc) > 30:
                desc = desc[:27] + "..."
            chapter_labels[digit] = f"{digit}: {desc}"
        
        plt.figure(figsize=(16, 8))
        plt.bar(chapter_labels.values(), chapter_counts)
        plt.title('Distribution of HS Codes by Chapter (First Digit)')
        plt.xlabel('Chapter')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'hs_code_by_chapter.png'))
        plt.show()
        
        #%% [markdown]
        # ### Average Description Length by HS Chapter
        #%%
        chapter_avg_length = df.groupby('first_digit')['description_length'].mean().sort_values(ascending=False)
        
        # Create labels with descriptions
        chapter_avg_labels = {}
        for digit in chapter_avg_length.index:
            if digit in chapter_labels:
                chapter_avg_labels[digit] = chapter_labels[digit]
            else:
                chapter_avg_labels[digit] = f"Chapter {digit}"
        
        plt.figure(figsize=(16, 8))
        plt.bar([chapter_avg_labels[d] for d in chapter_avg_length.index], chapter_avg_length.values)
        plt.title('Average Description Length by HS Code Chapter')
        plt.xlabel('Chapter')
        plt.ylabel('Average Length (characters)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'avg_length_by_chapter.png'))
        plt.show()
    else:
        print("HS codes don't appear to have at least 2 digits. Skipping digit-based analysis.")
except Exception as e:
    print(f"Error in HS code structure analysis: {e}")
    print("Skipping digit-based analysis.")

#%% [markdown]
# ## 8. Additional Insights
#%%
# Count number of words in each description
df['word_count'] = df['productDescription'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

# Plot distribution of word counts
plt.figure(figsize=(12, 6))
sns.histplot(df['word_count'], bins=50, kde=True)
plt.title('Distribution of Word Counts in Product Descriptions')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'word_count_dist.png'))
plt.show()

# Correlation between word count and description length
correlation = df['word_count'].corr(df['description_length'])
print(f"\nCorrelation between word count and description length: {correlation:.4f}")

#%% [markdown]
# ## 9. Conclusion
# 
# This analysis provides insights into the distribution of predicted HS codes and the characteristics of product descriptions. Key findings include:
# 
# - The most common HS codes and their frequencies
# - Common words associated with top HS code categories
# - Distribution of product description lengths
# - Patterns in HS code assignments by chapter
# 
# These insights can help evaluate the prediction model's behavior and identify potential areas for improvement.

print(f"\nAnalysis complete. Visualizations saved to {plots_dir}") 