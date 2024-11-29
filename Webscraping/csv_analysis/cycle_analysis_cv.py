import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde, norm
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Read the merged results
df = pd.read_csv('merged_results.csv')

# Convert dates to datetime
df['DischargedDate'] = pd.to_datetime(df['DischargedDate'], errors='coerce', utc=True).dt.tz_localize(None)
df['GateOutDate'] = pd.to_datetime(df['GateOutDate'], errors='coerce', utc=True).dt.tz_localize(None)

# Calculate dwell times for containers with both dates
both_dates = df[df['DischargedDate'].notna() & df['GateOutDate'].notna()].copy()
both_dates['dwell_time_days'] = (both_dates['GateOutDate'] - both_dates['DischargedDate']).dt.total_seconds() / (24*60*60)

def categorize_container(iso_code):
    if pd.isna(iso_code):
        return 'Unknown'
    iso_code = str(iso_code).upper()
    return 'Reefer' if 'R' in iso_code else 'Dry'

both_dates['container_category'] = both_dates['IsoCode'].apply(categorize_container)

def evaluate_kde(train_data, test_data, bandwidth):
    kde = gaussian_kde(train_data, bw_method=bandwidth)
    log_likelihood = np.mean(kde.logpdf(test_data))
    return log_likelihood

def get_baseline_score(train_data, test_data):
    mu, std = norm.fit(train_data)
    log_likelihood = np.mean(norm.logpdf(test_data, mu, std))
    return log_likelihood

def analyze_cycles_cv(data, category):
    print(f"\n{category} Containers Analysis with Cross-Validation:")
    
    # Filter outliers using 95th percentile
    data_filtered = data[data <= np.percentile(data, 95)]
    
    # Initialize K-fold cross-validation with just 2 folds
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    cv_scores = []
    baseline_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_filtered)):
        X_train = data_filtered[train_idx]
        X_test = data_filtered[test_idx]
        
        # Try different bandwidths
        bandwidths = np.linspace(0.1, 2.0, 10)  # Reduced number of bandwidths
        log_likelihoods = []
        
        for bw in bandwidths:
            log_likelihood = evaluate_kde(X_train, X_test, bw)
            log_likelihoods.append(log_likelihood)
        
        # Get baseline score
        baseline_score = get_baseline_score(X_train, X_test)
        baseline_scores.append(baseline_score)
        
        # Find optimal bandwidth and score for this fold
        optimal_bw = bandwidths[np.argmax(log_likelihoods)]
        best_score = max(log_likelihoods)
        cv_scores.append(best_score)
        
        print(f"\nFold {fold + 1}:")
        print(f"Optimal bandwidth: {optimal_bw:.3f}")
        print(f"KDE Log-likelihood: {best_score:.3f}")
        print(f"Baseline Log-likelihood: {baseline_score:.3f}")
        print(f"Improvement over baseline: {((best_score - baseline_score) / abs(baseline_score) * 100):.1f}%")
    
    # Average results across folds
    mean_cv_score = np.mean(cv_scores)
    mean_baseline = np.mean(baseline_scores)
    
    print(f"\nAverage across {len(cv_scores)} folds:")
    print(f"KDE Log-likelihood: {mean_cv_score:.3f} ± {np.std(cv_scores):.3f}")
    print(f"Baseline Log-likelihood: {mean_baseline:.3f} ± {np.std(baseline_scores):.3f}")
    print(f"Average improvement: {((mean_cv_score - mean_baseline) / abs(mean_baseline) * 100):.1f}%")
    
    return mean_cv_score, mean_baseline

# Analyze each container category
categories = ['Dry', 'Reefer']
results = {}

for category in categories:
    data = both_dates[both_dates['container_category'] == category]['dwell_time_days'].values
    kde_score, baseline_score = analyze_cycles_cv(data, category)
    results[category] = {'kde_score': kde_score, 'baseline_score': baseline_score}

# Print summary of results
print("\nFinal Summary:")
for category, metrics in results.items():
    print(f"\n{category} Containers:")
    print(f"KDE Log-likelihood: {metrics['kde_score']:.3f}")
    print(f"Baseline Log-likelihood: {metrics['baseline_score']:.3f}")
    print(f"Improvement over baseline: {((metrics['kde_score'] - metrics['baseline_score']) / abs(metrics['baseline_score']) * 100):.1f}%")
