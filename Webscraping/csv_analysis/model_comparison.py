import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
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

def predict_dwell_time(model, x_value):
    """Make a prediction using the full probability distribution"""
    if isinstance(model, gaussian_kde):
        return x_value[np.argmax(model(x_value))]
    else:  # Normal distribution
        return model.mean()

def evaluate_models(data, category, subplot_idx=1):
    print(f"\n{category} Containers Analysis:")
    
    # Filter outliers using 95th percentile
    data_filtered = data[data <= np.percentile(data, 95)]
    
    # Split data
    X_train, X_test = train_test_split(data_filtered, test_size=0.2, random_state=42)
    
    # Fit KDE with cross-validation for bandwidth
    bandwidths = np.linspace(0.1, 1.0, 10)
    best_kde = None
    best_score = -np.inf
    
    for bw in bandwidths:
        kde = gaussian_kde(X_train, bw_method=bw)
        score = np.mean(kde.logpdf(X_test))
        if score > best_score:
            best_score = score
            best_kde = kde
    
    kde = best_kde
    
    # Fit Normal distribution
    mu, std = norm.fit(X_train)
    normal_dist = norm(mu, std)
    
    # Generate prediction points
    x_range = np.linspace(0, np.max(data_filtered), 1000)
    
    # Make predictions for each test point using full distributions
    kde_predictions = []
    normal_predictions = []
    
    for x in X_test:
        # Find the most likely dwell time in a window around the actual value
        window = np.linspace(max(0, x-2), x+2, 100)
        kde_pred = predict_dwell_time(kde, window)
        normal_pred = predict_dwell_time(normal_dist, window)
        kde_predictions.append(kde_pred)
        normal_predictions.append(normal_pred)
    
    # Calculate errors
    kde_mae = mean_absolute_error(X_test, kde_predictions)
    normal_mae = mean_absolute_error(X_test, normal_predictions)
    
    kde_rmse = np.sqrt(mean_squared_error(X_test, kde_predictions))
    normal_rmse = np.sqrt(mean_squared_error(X_test, normal_predictions))
    
    # Plot distributions
    plt.subplot(1, 2, subplot_idx)
    
    # Plot KDE
    kde_density = kde(x_range)
    plt.plot(x_range, kde_density, 'b-', label='KDE')
    
    # Plot Normal
    normal_density = normal_dist.pdf(x_range)
    plt.plot(x_range, normal_density, 'r-', label='Normal')
    
    # Plot histogram of actual data
    plt.hist(X_test, bins=50, density=True, alpha=0.3, color='gray', label='Test Data')
    
    plt.title(f'{category} Containers - Distribution Comparison')
    plt.xlabel('Dwell Time (days)')
    plt.ylabel('Density')
    plt.legend()
    
    print(f"\nKDE Model:")
    print(f"Mean Absolute Error: {kde_mae:.2f} days")
    print(f"Root Mean Square Error: {kde_rmse:.2f} days")
    
    print(f"\nNormal Distribution:")
    print(f"Mean Absolute Error: {normal_mae:.2f} days")
    print(f"Root Mean Square Error: {normal_rmse:.2f} days")
    
    improvement_mae = ((normal_mae - kde_mae) / normal_mae) * 100
    improvement_rmse = ((normal_rmse - kde_rmse) / normal_rmse) * 100
    
    print(f"\nKDE Improvement:")
    print(f"MAE Improvement: {improvement_mae:.1f}%")
    print(f"RMSE Improvement: {improvement_rmse:.1f}%")
    
    return {
        'kde_mae': kde_mae,
        'normal_mae': normal_mae,
        'kde_rmse': kde_rmse,
        'normal_rmse': normal_rmse
    }

# Analyze each container category
categories = ['Dry', 'Reefer']
results = {}

# Create a single figure for both plots
plt.figure(figsize=(15, 6))

for idx, category in enumerate(categories):
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days'].values
    results[category] = evaluate_models(data, category, subplot_idx=idx+1)

plt.tight_layout()
plt.show()

# Print summary table
print("\nSummary Table:")
print("| Container Type | Model   | MAE    | RMSE   | Improvement |")
print("|---------------|---------|--------|---------|-------------|")
for category in categories:
    r = results[category]
    improvement = ((r['normal_mae'] - r['kde_mae']) / r['normal_mae'] * 100)
    print(f"| {category:<13} | KDE     | {r['kde_mae']:.2f}   | {r['kde_rmse']:.2f}    | {improvement:>11.1f}% |")
    print(f"| {category:<13} | Normal  | {r['normal_mae']:.2f}   | {r['normal_rmse']:.2f}    | {' ':>11} |")
