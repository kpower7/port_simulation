import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson
from scipy.optimize import minimize

# Set style for better visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Read the merged results
df = pd.read_csv('merged_results.csv')

# Convert dates to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce', utc=True).dt.tz_localize(None)
df['GateOutDate'] = pd.to_datetime(df['GateOutDate'], errors='coerce', utc=True).dt.tz_localize(None)

# Calculate dwell times for containers with both dates
both_dates = df[df['DischargedDate'].notna() & df['GateOutDate'].notna()].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

# Categorize containers
def categorize_container(iso_code):
    if pd.isna(iso_code):
        return 'Unknown'
    iso_code = str(iso_code).upper()
    return 'Reefer' if 'R' in iso_code else 'Dry'

both_dates['container_category'] = both_dates['IsoCode'].apply(categorize_container)

# Function to fit Poisson distribution
def fit_poisson(data):
    # Round dwell times to nearest integer (Poisson requires integer values)
    data_int = np.round(data).astype(int)
    # Calculate mean (lambda parameter for Poisson)
    lambda_param = np.mean(data_int)
    return lambda_param

# Fit Poisson distributions
categories = ['Dry', 'Reefer']
poisson_params = {}
for category in categories:
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days']
    poisson_params[category] = fit_poisson(data)

# Create visualization comparing actual distribution to Poisson
plt.figure(figsize=(15, 10))

for idx, category in enumerate(categories, 1):
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days']
    data_int = np.round(data).astype(int)
    max_days = int(np.percentile(data_int, 95))  # Use 95th percentile for better visualization
    
    plt.subplot(2, 2, idx)
    # Plot actual distribution
    counts, bins, _ = plt.hist(data_int[data_int <= max_days], 
                              bins=range(max_days + 2), 
                              density=True, 
                              alpha=0.6, 
                              label='Actual',
                              color='blue')
    
    # Generate Poisson distribution
    lambda_param = poisson_params[category]
    x = np.arange(0, max_days + 1)
    poisson_dist = poisson.pmf(x, lambda_param)
    
    # Plot Poisson distribution
    plt.plot(x, poisson_dist, 'r-', lw=2, label=f'Poisson (λ={lambda_param:.2f})')
    
    plt.title(f'{category} Containers\nActual vs Poisson Distribution')
    plt.xlabel('Dwell Time (days)')
    plt.ylabel('Probability')
    plt.legend()
    
    # Calculate goodness of fit using normalized counts
    observed = counts / np.sum(counts)
    expected = poisson_dist[:len(counts)] / np.sum(poisson_dist[:len(counts)])
    
    # Calculate KL divergence
    kl_div = np.sum(observed * np.log(observed / expected))
    print(f"\nGoodness of fit for {category} containers:")
    print(f"KL divergence: {kl_div:.4f}")
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(observed - expected))
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Calculate root mean squared error
    rmse = np.sqrt(np.mean((observed - expected) ** 2))
    print(f"Root Mean Squared Error: {rmse:.4f}")

# Q-Q plot
plt.subplot(2, 2, 3)
for category in categories:
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days']
    data_int = np.round(data).astype(int)
    lambda_param = poisson_params[category]
    
    # Generate theoretical quantiles
    theoretical_quantiles = poisson.ppf(np.linspace(0.01, 0.99, 100), lambda_param)
    actual_quantiles = np.percentile(data_int, np.linspace(1, 99, 100))
    
    plt.scatter(theoretical_quantiles, actual_quantiles, alpha=0.5, label=category)

plt.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--')
plt.xlabel('Theoretical Quantiles (Poisson)')
plt.ylabel('Sample Quantiles')
plt.title('Q-Q Plot')
plt.legend()

# Probability plot
plt.subplot(2, 2, 4)
for category in categories:
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days']
    data_int = np.round(data).astype(int)
    lambda_param = poisson_params[category]
    
    # Calculate empirical CDF
    sorted_data = np.sort(data_int)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Calculate theoretical CDF
    theoretical_cdf = poisson.cdf(sorted_data, lambda_param)
    
    plt.scatter(theoretical_cdf, empirical_cdf, alpha=0.5, label=category)

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Theoretical Probability')
plt.ylabel('Empirical Probability')
plt.title('Probability Plot')
plt.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nPoisson Distribution Parameters:")
for category in categories:
    lambda_param = poisson_params[category]
    print(f"\n{category} Containers:")
    print(f"Lambda (mean) = {lambda_param:.2f}")
    print(f"Variance = {lambda_param:.2f}")  # For Poisson, mean = variance
    print(f"Standard deviation = {np.sqrt(lambda_param):.2f}")
    
    # Calculate probability of various dwell times
    probs = [1, 2, 3, 5, 7, 10]
    print("\nProbability of dwell time <= X days (Poisson model):")
    for days in probs:
        prob = poisson.cdf(days, lambda_param)
        print(f"P(X <= {days}) = {prob:.2%}")
        
    # Calculate actual probabilities from data
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days']
    data_int = np.round(data).astype(int)
    print("\nActual probabilities from data:")
    for days in probs:
        prob = (data_int <= days).mean()
        print(f"P(X <= {days}) = {prob:.2%}")
        
    # Print comparison of actual vs predicted probabilities
    print("\nDifference between actual and predicted probabilities:")
    for days in probs:
        actual_prob = (data_int <= days).mean()
        predicted_prob = poisson.cdf(days, lambda_param)
        diff = actual_prob - predicted_prob
        print(f"Day {days}: {diff:+.2%}")
