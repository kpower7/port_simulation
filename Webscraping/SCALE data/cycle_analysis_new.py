import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]

# Read the merged results
print("Reading cleaned merged results...")
df = pd.read_csv('merged_results_final_december.csv')

# Convert dates to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce', utc=True).dt.tz_localize(None)
df['GateOutDate'] = pd.to_datetime(df['GateOutDate'], errors='coerce', utc=True).dt.tz_localize(None)

# Calculate dwell times for containers with both dates
both_dates = df[df['DischargedDate'].notna() & df['GateOutDate'].notna()].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

# Add hour of day and day of week features
both_dates['discharge_hour'] = both_dates['DischargedDate'].dt.hour
both_dates['discharge_dow'] = both_dates['DischargedDate'].dt.dayofweek
both_dates['gateout_hour'] = both_dates['GateOutDate'].dt.hour
both_dates['gateout_dow'] = both_dates['GateOutDate'].dt.dayofweek

# Categorize containers
def categorize_container(iso_code):
    if pd.isna(iso_code):
        return 'Unknown'
    iso_code = str(iso_code).upper()
    return 'Reefer' if 'R' in iso_code else 'Dry'

both_dates['container_category'] = both_dates['IsoCode'].apply(categorize_container)

def analyze_cycles(data, category):
    print(f"\n{category} Containers Analysis:")
    
    # 1. Gaussian Mixture Model Analysis
    # Reshape data for GMM
    X = data.reshape(-1, 1)
    
    # Find optimal number of components using BIC
    n_components_range = range(1, 8)
    bic = []
    lowest_bic = np.inf
    best_gmm = None
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    
    optimal_components = n_components_range[np.argmin(bic)]
    print(f"\nOptimal number of Gaussian components: {optimal_components}")
    
    # Plot GMM results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and GMM components
    plt.subplot(2, 2, 1)
    x_plot = np.linspace(0, np.percentile(data, 95), 1000).reshape(-1, 1)
    
    # Plot histogram of actual data
    plt.hist(data[data <= np.percentile(data, 95)], bins=50, density=True, alpha=0.5, color='gray', label='Actual Data')
    
    # Plot individual components
    logprob = best_gmm.score_samples(x_plot)
    responsibilities = best_gmm.predict_proba(x_plot)
    pdf = np.exp(logprob)
    plt.plot(x_plot, pdf, 'k-', label='GMM')
    
    for i in range(optimal_components):
        pdf_component = responsibilities[:, i] * pdf
        plt.plot(x_plot, pdf_component, '--', label=f'Component {i+1}')
    
    plt.title(f'{category} Containers - Gaussian Mixture Components')
    plt.xlabel('Dwell Time (days)')
    plt.ylabel('Density')
    plt.legend()
    
    # 2. Kernel Density Estimation
    plt.subplot(2, 2, 2)
    kde = gaussian_kde(data[data <= np.percentile(data, 95)])
    x_kde = np.linspace(0, np.percentile(data, 95), 1000)
    plt.plot(x_kde, kde(x_kde), 'r-', label='KDE')
    plt.hist(data[data <= np.percentile(data, 95)], bins=50, density=True, alpha=0.5, color='gray', label='Actual Data')
    plt.title(f'{category} Containers - Kernel Density Estimation')
    plt.xlabel('Dwell Time (days)')
    plt.ylabel('Density')
    plt.legend()
    
    # 3. Time-of-Day Analysis
    plt.subplot(2, 2, 3)
    hour_counts = pd.DataFrame({
        'Discharge': both_dates[both_dates['container_category'] == category]['discharge_hour'].value_counts().sort_index(),
        'Gate Out': both_dates[both_dates['container_category'] == category]['gateout_hour'].value_counts().sort_index()
    })
    hour_counts.plot(kind='bar', alpha=0.6)
    plt.title(f'{category} Containers - Activity by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.legend()
    
    # 4. Day-of-Week Analysis
    plt.subplot(2, 2, 4)
    dow_counts = pd.DataFrame({
        'Discharge': both_dates[both_dates['container_category'] == category]['discharge_dow'].value_counts().sort_index(),
        'Gate Out': both_dates[both_dates['container_category'] == category]['gateout_dow'].value_counts().sort_index()
    })
    dow_counts.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts.plot(kind='bar', alpha=0.6)
    plt.title(f'{category} Containers - Activity by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{category.lower()}_cycle_analysis.png')
    print(f"Saved visualization to {category.lower()}_cycle_analysis.png")
    
    # Print component parameters
    print("\nGMM Component Parameters:")
    for i in range(optimal_components):
        weight = best_gmm.weights_[i]
        mean = best_gmm.means_[i][0]
        std = np.sqrt(best_gmm.covariances_[i][0][0])
        print(f"Component {i+1}:")
        print(f"  Weight: {weight:.3f}")
        print(f"  Mean: {mean:.2f} days")
        print(f"  Std Dev: {std:.2f} days")
    
    # Analyze hourly patterns
    print("\nPeak Activity Hours:")
    peak_discharge_hour = hour_counts['Discharge'].idxmax()
    peak_gateout_hour = hour_counts['Gate Out'].idxmax()
    print(f"Peak Discharge Hour: {peak_discharge_hour:02d}:00")
    print(f"Peak Gate Out Hour: {peak_gateout_hour:02d}:00")
    
    # Analyze daily patterns
    print("\nBusiest Days:")
    peak_discharge_day = dow_counts['Discharge'].idxmax()
    peak_gateout_day = dow_counts['Gate Out'].idxmax()
    print(f"Busiest Discharge Day: {peak_discharge_day}")
    print(f"Busiest Gate Out Day: {peak_gateout_day}")
    
    return best_gmm

# Print initial statistics
print(f"\nTotal containers in dataset: {len(df):,}")
print(f"Containers with both discharge and gate out dates: {len(both_dates):,}")
print(f"Average dwell time: {both_dates['dwell_time_days'].mean():.2f} days")
print(f"Median dwell time: {both_dates['dwell_time_days'].median():.2f} days")

# Count container categories
category_counts = both_dates['container_category'].value_counts()
print("\nContainer Categories:")
for category, count in category_counts.items():
    print(f"{category}: {count:,} containers ({count/len(both_dates)*100:.1f}%)")

# Analyze each container category
categories = ['Dry', 'Reefer']
for category in categories:
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days'].values
    if len(data) > 0:  # Only analyze if we have data for this category
        gmm = analyze_cycles(data, category)

# Additional analysis of time patterns
print("\nTime Pattern Analysis:")
for category in categories:
    subset = both_dates[both_dates['container_category'] == category]
    if len(subset) > 0:
        print(f"\n{category} Containers:")
        
        # Calculate correlation between discharge and gate out times
        hour_correlation = stats.pearsonr(
            subset['discharge_hour'],
            subset['gateout_hour']
        )
        
        print(f"Hour correlation coefficient: {hour_correlation[0]:.3f} (p-value: {hour_correlation[1]:.3f})")
        
        # Calculate average dwell time by hour of discharge
        avg_dwell_by_discharge_hour = subset.groupby('discharge_hour')['dwell_time_days'].mean()
        best_discharge_hour = avg_dwell_by_discharge_hour.idxmin()
        worst_discharge_hour = avg_dwell_by_discharge_hour.idxmax()
        
        print(f"Best discharge hour (shortest avg dwell time): {best_discharge_hour:02d}:00 ({avg_dwell_by_discharge_hour.min():.2f} days)")
        print(f"Worst discharge hour (longest avg dwell time): {worst_discharge_hour:02d}:00 ({avg_dwell_by_discharge_hour.max():.2f} days)")
