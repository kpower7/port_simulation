import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Read the merged results
df = pd.read_csv('merged_results.csv')

# Convert dates to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce', utc=True).dt.tz_localize(None)
df['GateOutDate'] = pd.to_datetime(df['GateOutDate'], errors='coerce', utc=True).dt.tz_localize(None)

# Calculate dwell times
both_dates = df[df['DischargedDate'].notna() & df['GateOutDate'].notna()].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

def categorize_container(iso_code):
    if pd.isna(iso_code):
        return 'Unknown'
    iso_code = str(iso_code).upper()
    return 'Reefer' if 'R' in iso_code else 'Dry'

both_dates['container_category'] = both_dates['IsoCode'].apply(categorize_container)

def fourier_series(x, *params):
    """
    Compute Fourier series with n_terms
    params: [a0, a1, b1, a2, b2, ..., an, bn]
    """
    n_terms = (len(params) - 1) // 2
    result = params[0] / 2  # a0 term
    
    for i in range(n_terms):
        n = i + 1
        a = params[2*i + 1]
        b = params[2*i + 2]
        result += a * np.cos(n * x) + b * np.sin(n * x)
    
    return result

def evaluate_models(data, category, subplot_idx=1):
    print(f"\n{category} Containers Analysis:")
    
    # Filter outliers using 95th percentile
    data_filtered = data[data <= np.percentile(data, 95)]
    
    # Split data
    X_train, X_test = train_test_split(data_filtered, test_size=0.2, random_state=42)
    
    # Prepare data for density estimation
    hist_train, bin_edges = np.histogram(X_train, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit KDE
    kde = gaussian_kde(X_train)
    
    # Fit Normal distribution
    mu, std = norm.fit(X_train)
    
    # Prepare data for Fourier series
    x_scaled = bin_centers / np.max(bin_centers) * 2 * np.pi
    
    # Fit Fourier series with 5 terms
    n_terms = 5
    p0 = [1.0] + [0.1] * (2 * n_terms)
    fourier_params, _ = curve_fit(fourier_series, x_scaled, hist_train, p0=p0, maxfev=10000)
    
    # Generate points for plotting
    x_plot = np.linspace(0, np.max(data_filtered), 1000)
    x_plot_scaled = x_plot / np.max(data_filtered) * 2 * np.pi
    
    # Calculate densities
    kde_density = kde(x_plot)
    normal_density = norm.pdf(x_plot, mu, std)
    fourier_density = fourier_series(x_plot_scaled, *fourier_params)
    fourier_density = np.maximum(fourier_density, 0)
    fourier_density = fourier_density / np.trapz(fourier_density, x_plot)
    
    # Calculate errors on test set
    hist_test, _ = np.histogram(X_test, bins=50, density=True)
    x_test = bin_centers
    
    kde_test = kde(x_test)
    normal_test = norm.pdf(x_test, mu, std)
    x_test_scaled = x_test / np.max(data_filtered) * 2 * np.pi
    fourier_test = fourier_series(x_test_scaled, *fourier_params)
    fourier_test = np.maximum(fourier_test, 0)
    fourier_test = fourier_test / np.trapz(fourier_test, x_test)
    
    # Calculate errors
    kde_mae = mean_absolute_error(hist_test, kde_test)
    normal_mae = mean_absolute_error(hist_test, normal_test)
    fourier_mae = mean_absolute_error(hist_test, fourier_test)
    
    kde_rmse = np.sqrt(mean_squared_error(hist_test, kde_test))
    normal_rmse = np.sqrt(mean_squared_error(hist_test, normal_test))
    fourier_rmse = np.sqrt(mean_squared_error(hist_test, fourier_test))
    
    # Plot distributions
    plt.subplot(1, 2, subplot_idx)
    
    plt.hist(X_test, bins=50, density=True, alpha=0.3, color='gray', label='Test Data')
    plt.plot(x_plot, kde_density, 'b-', label='KDE')
    plt.plot(x_plot, normal_density, 'r-', label='Normal')
    plt.plot(x_plot, fourier_density, 'g-', label='Fourier')
    
    plt.title(f'{category} Containers')
    plt.xlabel('Dwell Time (days)')
    plt.ylabel('Density')
    plt.legend()
    
    # Print results
    print("\nModel Performance:")
    print(f"KDE - MAE: {kde_mae:.4f}, RMSE: {kde_rmse:.4f}")
    print(f"Normal - MAE: {normal_mae:.4f}, RMSE: {normal_rmse:.4f}")
    print(f"Fourier - MAE: {fourier_mae:.4f}, RMSE: {fourier_rmse:.4f}")
    
    return {
        'kde_mae': kde_mae,
        'normal_mae': normal_mae,
        'fourier_mae': fourier_mae,
        'kde_rmse': kde_rmse,
        'normal_rmse': normal_rmse,
        'fourier_rmse': fourier_rmse
    }

# Create a single figure for both plots
plt.figure(figsize=(15, 6))

# Analyze each container category
categories = ['Dry', 'Reefer']
results = {}

for idx, category in enumerate(categories):
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days'].values
    results[category] = evaluate_models(data, category, subplot_idx=idx+1)

plt.tight_layout()
plt.show()

# Print summary table
print("\nSummary Table:")
print("| Container Type | Model   | MAE     | RMSE    | Improvement vs Normal |")
print("|---------------|---------|---------|---------|---------------------|")
for category in categories:
    r = results[category]
    kde_imp = ((r['normal_mae'] - r['kde_mae']) / r['normal_mae'] * 100)
    fourier_imp = ((r['normal_mae'] - r['fourier_mae']) / r['normal_mae'] * 100)
    
    print(f"| {category:<13} | KDE     | {r['kde_mae']:.4f} | {r['kde_rmse']:.4f} | {kde_imp:>19.1f}% |")
    print(f"| {category:<13} | Fourier | {r['fourier_mae']:.4f} | {r['fourier_rmse']:.4f} | {fourier_imp:>19.1f}% |")
    print(f"| {category:<13} | Normal  | {r['normal_mae']:.4f} | {r['normal_rmse']:.4f} | {' ':>19} |")
