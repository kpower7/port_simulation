import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os
import re
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="HS Code Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Function to load data
@st.cache_data
def load_data():
    # Update this path to the correct location
    predictions_file = 'december_final_predictions_with_desc.csv'
    df = pd.read_csv(predictions_file)
    
    # Check if the date columns exist and convert them
    date_columns = ['DischargedDate', 'GateOutDate']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                st.warning(f"Error converting {col} to datetime: {e}")
    
    # Create sample date columns if they don't exist (for demonstration)
    if 'DischargedDate' not in df.columns or 'GateOutDate' not in df.columns:
        st.warning("Date columns not found in the data. Creating sample dates for demonstration.")
        
        if 'DischargedDate' not in df.columns:
            # Generate random dates in December 2022
            start_date = datetime(2022, 12, 1)
            end_date = datetime(2022, 12, 31)
            days = (end_date - start_date).days
            
            df['DischargedDate'] = [start_date + timedelta(days=np.random.randint(0, days)) for _ in range(len(df))]
        
        if 'GateOutDate' not in df.columns:
            # GateOutDate is 1-10 days after DischargedDate
            df['GateOutDate'] = [d + timedelta(days=np.random.randint(1, 11)) for d in df['DischargedDate']]
    
    # Calculate dwell time if date columns exist
    if all(col in df.columns for col in date_columns):
        try:
            df['DwellTime'] = (df['GateOutDate'] - df['DischargedDate']).dt.total_seconds() / (24 * 60 * 60)
            
            # Filter out negative or extremely large dwell times (likely data errors)
            df = df[(df['DwellTime'] >= 0) & (df['DwellTime'] <= 365)]  # Max 1 year
        except Exception as e:
            st.error(f"Error calculating dwell time: {e}")
    
    # Extract date features
    if 'GateOutDate' in df.columns:
        df['Month'] = df['GateOutDate'].dt.month
        df['Day'] = df['GateOutDate'].dt.day
        df['DayOfWeek'] = df['GateOutDate'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    
    # Calculate description lengths
    if 'productDescription' in df.columns:
        df['description_length'] = df['productDescription'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        df['word_count'] = df['productDescription'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    # Extract HS code chapter (first digit)
    df['HS_Chapter'] = df['Predicted HS Code'].astype(str).str[0]
    
    return df

# Function to extract keywords from text
def extract_keywords(text):
    if pd.isna(text):
        return []
    # Convert to lowercase and remove punctuation
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words and filter out short words
    words = [word for word in text.split() if len(word) > 3]
    return words

# Main app
def main():
    st.title("HS Code Analysis Dashboard")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Display basic info
    st.sidebar.header("Dashboard Info")
    st.sidebar.info(f"""
    - Total Records: {len(df)}
    - Unique HS Codes: {df['Predicted HS Code'].nunique()}
    - Date Range: {df['GateOutDate'].min().date()} to {df['GateOutDate'].max().date()}
    """)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["HS Code Distribution", "Temporal Analysis", "Dwell Time Analysis", "Text Analysis"])
    
    with tab1:
        st.header("HS Code Distribution Analysis")
        
        # Get top HS codes
        hs_counts = df['Predicted HS Code'].value_counts()
        top_hs_codes = hs_counts.head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 20 Most Frequent HS Codes")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get descriptions for top HS codes
            top_hs_with_desc = {}
            for hs_code in top_hs_codes.index:
                desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
                # Truncate description if too long
                if len(desc) > 20:
                    desc = desc[:17] + "..."
                top_hs_with_desc[f"{hs_code}: {desc}"] = hs_counts[hs_code]
            
            plt.barh(list(top_hs_with_desc.keys())[::-1], list(top_hs_with_desc.values())[::-1])
            plt.title('Top 20 Most Frequent HS Codes')
            plt.xlabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Pie Chart of Top 10 HS Codes vs Others")
            
            # Get descriptions for top 10 HS codes
            top_10_hs = hs_counts.head(10)
            top_10_with_desc = []
            for hs_code in top_10_hs.index:
                desc = df[df['Predicted HS Code'] == hs_code]['HS_Description'].iloc[0]
                # Truncate description if too long
                if len(desc) > 15:
                    desc = desc[:12] + "..."
                top_10_with_desc.append(f"{hs_code}: {desc}")
            
            others_count = hs_counts[10:].sum()
            sizes = list(top_10_hs) + [others_count]
            labels = top_10_with_desc + ['Others']
            
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
        
        # HS Code Structure Analysis
        st.subheader("HS Code Structure Analysis")
        
        try:
            # Create a cross-tabulation of first and second digits
            heatmap_data = pd.crosstab(df['HS_Chapter'], df['Predicted HS Code'].astype(str).str[1])
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='d')
            plt.title('Distribution of HS Codes (First Digit vs Second Digit)')
            plt.xlabel('Second Digit')
            plt.ylabel('First Digit (Chapter)')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in HS code structure analysis: {e}")
    
    with tab2:
        st.header("Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Arrivals by Day of Week")
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['DayOfWeek'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.bar(day_names, day_counts)
            plt.title('Arrivals by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Number of Arrivals')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Arrivals by Day of Month")
            
            day_of_month_counts = df['Day'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.bar(day_of_month_counts.index, day_of_month_counts.values)
            plt.title('Arrivals by Day of Month')
            plt.xlabel('Day of Month')
            plt.ylabel('Number of Arrivals')
            plt.xticks(range(1, 32))
            plt.tight_layout()
            st.pyplot(fig)
        
        # Time Series of Arrivals
        st.subheader("Time Series of Arrivals")
        
        # Group by date and count arrivals
        daily_arrivals = df.groupby(df['GateOutDate'].dt.date).size()
        
        # Convert to DataFrame for easier plotting
        daily_arrivals_df = pd.DataFrame({'Date': daily_arrivals.index, 'Count': daily_arrivals.values})
        daily_arrivals_df['Date'] = pd.to_datetime(daily_arrivals_df['Date'])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        plt.plot(daily_arrivals_df['Date'], daily_arrivals_df['Count'], marker='o', linestyle='-')
        plt.title('Daily Arrivals Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Arrivals')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.header("Dwell Time Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dwell Time Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['DwellTime'], bins=30, kde=True)
            plt.title('Distribution of Dwell Times')
            plt.xlabel('Dwell Time (days)')
            plt.ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.metric("Average Dwell Time", f"{df['DwellTime'].mean():.2f} days")
            st.metric("Median Dwell Time", f"{df['DwellTime'].median():.2f} days")
        
        with col2:
            st.subheader("Dwell Time by Day of Week")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='DayOfWeek', y='DwellTime', data=df)
            plt.title('Dwell Time Distribution by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Dwell Time (days)')
            plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            plt.tight_layout()
            st.pyplot(fig)
        
        # Dwell Time by HS Code
        st.subheader("Dwell Time by Top HS Codes")
        
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
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='HS_Label', y='DwellTime', data=df_top_hs)
        plt.title('Dwell Time Distribution by Top 10 HS Codes')
        plt.xlabel('HS Code')
        plt.ylabel('Dwell Time (days)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Dwell Time Trends Over Time
        st.subheader("Dwell Time Trends Over Time")
        
        # Calculate average dwell time by date
        dwell_by_date = df.groupby(df['GateOutDate'].dt.date)['DwellTime'].mean()
        
        # Convert to DataFrame
        dwell_by_date_df = pd.DataFrame({'Date': dwell_by_date.index, 'AvgDwellTime': dwell_by_date.values})
        dwell_by_date_df['Date'] = pd.to_datetime(dwell_by_date_df['Date'])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        plt.plot(dwell_by_date_df['Date'], dwell_by_date_df['AvgDwellTime'], marker='o', linestyle='-')
        plt.title('Average Dwell Time Trend')
        plt.xlabel('Date')
        plt.ylabel('Average Dwell Time (days)')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation Between Dwell Time and Arrival Volume
        st.subheader("Correlation: Arrival Volume vs Dwell Time")
        
        # Merge daily arrivals with average dwell time
        volume_dwell_df = pd.merge(
            daily_arrivals_df,
            dwell_by_date_df,
            on='Date',
            how='inner'
        )
        
        # Calculate correlation
        correlation = volume_dwell_df['Count'].corr(volume_dwell_df['AvgDwellTime'])
        st.metric("Correlation Coefficient", f"{correlation:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(volume_dwell_df['Count'], volume_dwell_df['AvgDwellTime'], alpha=0.7)
        plt.title('Correlation Between Daily Arrival Volume and Average Dwell Time')
        plt.xlabel('Number of Arrivals')
        plt.ylabel('Average Dwell Time (days)')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        st.header("Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Product Description Lengths")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['description_length'], bins=30, kde=True)
            plt.title('Distribution of Product Description Lengths')
            plt.xlabel('Description Length (characters)')
            plt.ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.metric("Average Description Length", f"{df['description_length'].mean():.2f} characters")
            st.metric("Median Description Length", f"{df['description_length'].median():.0f} characters")
        
        with col2:
            st.subheader("Distribution of Word Counts")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['word_count'], bins=30, kde=True)
            plt.title('Distribution of Word Counts in Product Descriptions')
            plt.xlabel('Number of Words')
            plt.ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation between word count and description length
            correlation = df['word_count'].corr(df['description_length'])
            st.metric("Word Count vs Length Correlation", f"{correlation:.4f}")
        
        # Word Frequency Analysis for Top HS Codes
        st.subheader("Word Frequency Analysis for Top HS Codes")
        
        # Get the top 5 HS codes for detailed analysis
        top_5_hs = df['Predicted HS Code'].value_counts().head(5).index
        
        # Create a selectbox to choose HS code
        selected_hs = st.selectbox(
            "Select HS Code for Word Analysis",
            top_5_hs,
            format_func=lambda x: f"{x} - {df[df['Predicted HS Code'] == x]['HS_Description'].iloc[0]}"
        )
        
        # Filter data for this HS code
        hs_data = df[df['Predicted HS Code'] == selected_hs]
        
        # Extract all words from product descriptions
        all_words = []
        for desc_text in hs_data['productDescription']:
            all_words.extend(extract_keywords(desc_text))
        
        # Count word frequencies
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(20)
        
        if most_common:
            words, counts = zip(*most_common)
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.barh(range(len(words)), counts, align='center')
            plt.yticks(range(len(words)), words)
            plt.xlabel('Frequency')
            plt.title(f'Most Common Words for HS Code {selected_hs}')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No common words found for this HS code.")
        
        # Description lengths by top HS codes
        st.subheader("Description Lengths by Top HS Codes")
        
        # Get top 10 HS codes for description length analysis
        top_10_hs = df['Predicted HS Code'].value_counts().head(10).index
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
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='HS_Label', y='description_length', data=df_top10)
        plt.title('Distribution of Product Description Lengths by HS Code (Top 10)')
        plt.xlabel('HS Code')
        plt.ylabel('Description Length (characters)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
