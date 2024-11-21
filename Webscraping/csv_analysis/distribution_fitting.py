import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, poisson, expon, weibull_min, lognorm
import warnings
warnings.filterwarnings('ignore')

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

# Function to calculate AIC
def calculate_aic(log_likelihood, n_params):
    return 2 * n_params - 2 * log_likelihood

# Function to calculate BIC
def calculate_bic(log_likelihood, n_params, n_samples):
    return n_params * np.log(n_samples) - 2 * log_likelihood

# Function to fit distributions and calculate goodness of fit
def fit_distributions(data):
    # Remove any negative values or zeros for certain distributions
    data_positive = data[data > 0]
    
    # Dictionary to store results
    results = {}
    
    # Normal distribution
    params_normal = stats.norm.fit(data)
    log_likelihood_normal = np.sum(stats.norm.logpdf(data, *params_normal))
    results['Normal'] = {
        'params': params_normal,
        'log_likelihood': log_likelihood_normal,
        'aic': calculate_aic(log_likelihood_normal, 2),
        'bic': calculate_bic(log_likelihood_normal, 2, len(data)),
        'distribution': stats.norm,
    }
    
    # Poisson distribution (rounded data)
    data_int = np.round(data).astype(int)
    lambda_poisson = np.mean(data_int)
    log_likelihood_poisson = np.sum(stats.poisson.logpmf(data_int, lambda_poisson))
    results['Poisson'] = {
        'params': (lambda_poisson,),
        'log_likelihood': log_likelihood_poisson,
        'aic': calculate_aic(log_likelihood_poisson, 1),
        'bic': calculate_bic(log_likelihood_poisson, 1, len(data)),
        'distribution': stats.poisson,
    }
    
    # Exponential distribution
    params_exp = stats.expon.fit(data_positive)
    log_likelihood_exp = np.sum(stats.expon.logpdf(data_positive, *params_exp))
    results['Exponential'] = {
        'params': params_exp,
        'log_likelihood': log_likelihood_exp,
        'aic': calculate_aic(log_likelihood_exp, 1),
        'bic': calculate_bic(log_likelihood_exp, 1, len(data_positive)),
        'distribution': stats.expon,
    }
    
    # Weibull distribution
    params_weibull = stats.weibull_min.fit(data_positive)
    log_likelihood_weibull = np.sum(stats.weibull_min.logpdf(data_positive, *params_weibull))
    results['Weibull'] = {
        'params': params_weibull,
        'log_likelihood': log_likelihood_weibull,
        'aic': calculate_aic(log_likelihood_weibull, 2),
        'bic': calculate_bic(log_likelihood_weibull, 2, len(data_positive)),
        'distribution': stats.weibull_min,
    }
    
    # Log-normal distribution
    params_lognorm = stats.lognorm.fit(data_positive)
    log_likelihood_lognorm = np.sum(stats.lognorm.logpdf(data_positive, *params_lognorm))
    results['Log-normal'] = {
        'params': params_lognorm,
        'log_likelihood': log_likelihood_lognorm,
        'aic': calculate_aic(log_likelihood_lognorm, 2),
        'bic': calculate_bic(log_likelihood_lognorm, 2, len(data_positive)),
        'distribution': stats.lognorm,
    }
    
    return results

# Function to plot distribution comparisons
def plot_distribution_comparison(data, results, title):
    plt.figure(figsize=(15, 10))
    
    # Histogram of actual data
    plt.hist(data[data <= np.percentile(data, 95)], 
             bins=50, density=True, alpha=0.6, 
             label='Actual Data', color='gray')
    
    # Plot fitted distributions
    x = np.linspace(0, np.percentile(data, 95), 100)
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for (name, res), color in zip(results.items(), colors):
        if name == 'Poisson':
            # Special handling for Poisson
            x_discrete = np.arange(0, int(np.percentile(data, 95)) + 1)
            pmf = res['distribution'].pmf(x_discrete, *res['params'])
            plt.plot(x_discrete, pmf, color=color, 
                    label=f'{name} (AIC: {res["aic"]:.0f})', 
                    linestyle='--', alpha=0.8)
        else:
            try:
                pdf = res['distribution'].pdf(x, *res['params'])
                plt.plot(x, pdf, color=color, 
                        label=f'{name} (AIC: {res["aic"]:.0f})', 
                        linestyle='--', alpha=0.8)
            except:
                continue
    
    plt.title(title)
    plt.xlabel('Dwell Time (days)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Analyze each container category
categories = ['Dry', 'Reefer']
for category in categories:
    print(f"\n{category} Containers Analysis:")
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days'].values
    
    # Fit distributions
    results = fit_distributions(data)
    
    # Print results
    print("\nDistribution Fitting Results:")
    print(f"{'Distribution':<12} {'Log-Likelihood':>15} {'AIC':>10} {'BIC':>10}")
    print("-" * 47)
    
    # Sort by AIC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['aic'])
    for name, res in sorted_results:
        print(f"{name:<12} {res['log_likelihood']:>15.2f} {res['aic']:>10.2f} {res['bic']:>10.2f}")
    
    # Plot comparison
    plot_distribution_comparison(data, results, f"{category} Containers - Distribution Fitting")
    
    # Print parameters of best fit
    best_dist = sorted_results[0][0]
    best_params = sorted_results[0][1]['params']
    print(f"\nBest fitting distribution: {best_dist}")
    print(f"Parameters: {best_params}")
    
    # Calculate some key probabilities using the best distribution
    if best_dist != 'Poisson':
        dist = results[best_dist]['distribution']
        probs = [1, 2, 3, 5, 7, 10]
        print("\nProbabilities using best fit distribution:")
        for days in probs:
            prob = dist.cdf(days, *best_params)
            print(f"P(X <= {days}) = {prob:.2%}")
    
    # Print summary statistics
    print("\nActual Data Summary:")
    print(pd.Series(data).describe())
