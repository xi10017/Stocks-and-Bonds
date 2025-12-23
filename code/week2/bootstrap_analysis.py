"""
Block Bootstrap Analysis - Week 2 (VOO 2010-2025)

Block bootstrap preserves temporal dependence in time series data
by resampling blocks of consecutive observations instead of individual points.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import download_data, calculate_sharpe_ratio

# =============================================================================
# Configuration
# =============================================================================
TICKERS = {"Stock": "VOO", "Bond": "IEF", "VIX": "^VIX"}

# Optimal parameters from tuning (VOO 2010-2025)
# Note: 15 months = 15 * 21 trading days ≈ 315 days (not calendar days!)
LOOKBACK_WINDOW = 315   # 15 months (optimal for 90% cutoff)
PERCENTILE_CUTOFF = 0.90  # 90% (optimal)

N_BOOTSTRAP = 1000
BLOCK_SIZE = 21
CONFIDENCE_LEVEL = 0.95
RISK_FREE_RATE = 0.02

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# =============================================================================
# Data Download
# =============================================================================
print("=" * 70)
print("BLOCK BOOTSTRAP ANALYSIS - WEEK 2 (VOO 2010-2025)")
print("=" * 70)
print(f"\nDownloading data...")

df = download_data(TICKERS, START_DATE, END_DATE)
df["Stock_Return"] = df["Stock"].pct_change()
df["Bond_Return"] = df["Bond"].pct_change()

# Generate VIX strategy signals with optimal parameters
df["VIX_threshold"] = df["VIX"].rolling(window=LOOKBACK_WINDOW).quantile(PERCENTILE_CUTOFF)
df["Signal"] = (df["VIX"] > df["VIX_threshold"]).astype(int).shift(1)
df = df.dropna()

df["Strategy_Return"] = np.where(df["Signal"] == 1, df["Bond_Return"], df["Stock_Return"])

print(f"Data loaded: {len(df)} trading days (~{len(df)/252:.1f} years)")
print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Optimal parameters: Lookback={LOOKBACK_WINDOW} days (3mo), Cutoff={PERCENTILE_CUTOFF*100:.0f}%")

# =============================================================================
# Block Bootstrap Function
# =============================================================================
def block_bootstrap_sharpe(returns, n_bootstrap=N_BOOTSTRAP, block_size=BLOCK_SIZE):
    """Perform block bootstrap to estimate Sharpe ratio distribution."""
    n = len(returns)
    returns_array = returns.values
    sharpe_samples = []
    n_blocks = int(np.ceil(n / block_size))
    
    for _ in range(n_bootstrap):
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        bootstrap_sample = []
        for start in block_starts:
            bootstrap_sample.extend(returns_array[start:start + block_size])
        bootstrap_sample = np.array(bootstrap_sample[:n])
        sharpe = calculate_sharpe_ratio(pd.Series(bootstrap_sample), RISK_FREE_RATE)
        sharpe_samples.append(sharpe)
    
    return np.array(sharpe_samples)

# =============================================================================
# Run Bootstrap Analysis
# =============================================================================
print(f"\nRunning block bootstrap ({N_BOOTSTRAP} samples, {BLOCK_SIZE}-day blocks)...")

original_sharpe = {
    "S&P 500 (VOO)": calculate_sharpe_ratio(df["Stock_Return"], RISK_FREE_RATE),
    "10Y Treasury": calculate_sharpe_ratio(df["Bond_Return"], RISK_FREE_RATE),
    "VIX Strategy": calculate_sharpe_ratio(df["Strategy_Return"], RISK_FREE_RATE)
}

bootstrap_results = {}
for name, returns in [("S&P 500 (VOO)", df["Stock_Return"]), 
                       ("10Y Treasury", df["Bond_Return"]),
                       ("VIX Strategy", df["Strategy_Return"])]:
    print(f"  Bootstrapping {name}...")
    bootstrap_results[name] = block_bootstrap_sharpe(returns)

print("Done!")

# =============================================================================
# Calculate Confidence Intervals
# =============================================================================
alpha = 1 - CONFIDENCE_LEVEL
ci_results = {}

print(f"\n{'='*70}")
print(f"SHARPE RATIO CONFIDENCE INTERVALS ({CONFIDENCE_LEVEL*100:.0f}%)")
print(f"{'='*70}")
print(f"\n{'Strategy':<20} {'Original':>10} {'CI Lower':>10} {'CI Upper':>10} {'CI Width':>10}")
print("-" * 62)

for name in ["S&P 500 (VOO)", "10Y Treasury", "VIX Strategy"]:
    samples = bootstrap_results[name]
    ci_lower = np.percentile(samples, alpha/2 * 100)
    ci_upper = np.percentile(samples, (1 - alpha/2) * 100)
    ci_width = ci_upper - ci_lower
    
    ci_results[name] = {
        "original": original_sharpe[name],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "width": ci_width,
        "samples": samples
    }
    
    print(f"{name:<20} {original_sharpe[name]:>10.3f} {ci_lower:>10.3f} {ci_upper:>10.3f} {ci_width:>10.3f}")

# =============================================================================
# Statistical Tests
# =============================================================================
vix_samples = bootstrap_results["VIX Strategy"]
sp500_samples = bootstrap_results["S&P 500 (VOO)"]
prob_vix_beats_sp500 = np.mean(vix_samples > sp500_samples)

vix_ci = (ci_results["VIX Strategy"]["ci_lower"], ci_results["VIX Strategy"]["ci_upper"])
sp500_ci = (ci_results["S&P 500 (VOO)"]["ci_lower"], ci_results["S&P 500 (VOO)"]["ci_upper"])
ci_overlap = not (vix_ci[0] > sp500_ci[1] or sp500_ci[0] > vix_ci[1])

print(f"\n{'='*70}")
print("STATISTICAL SIGNIFICANCE")
print(f"{'='*70}")
print(f"Probability VIX Strategy Sharpe > S&P 500 Sharpe: {prob_vix_beats_sp500:.1%}")
print(f"95% Confidence Intervals Overlap: {'Yes' if ci_overlap else 'No'}")
if not ci_overlap:
    print("→ VIX Strategy Sharpe is SIGNIFICANTLY higher than S&P 500!")
else:
    print("→ Difference is NOT statistically significant at 95% confidence")

# =============================================================================
# Visualization
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {"S&P 500 (VOO)": "#1f77b4", "10Y Treasury": "#2ca02c", "VIX Strategy": "#d62728"}

# Plot 1: Histogram
ax1 = axes[0, 0]
for name in ["S&P 500 (VOO)", "10Y Treasury", "VIX Strategy"]:
    ax1.hist(bootstrap_results[name], bins=50, alpha=0.5, label=name, color=colors[name])
    ax1.axvline(original_sharpe[name], color=colors[name], linestyle='--', linewidth=2)
ax1.set_xlabel("Sharpe Ratio", fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)
ax1.set_title("Bootstrap Distribution of Sharpe Ratios (VOO 2010-2025)", fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Box plot
ax2 = axes[0, 1]
bp_data = [bootstrap_results["S&P 500 (VOO)"], bootstrap_results["10Y Treasury"], bootstrap_results["VIX Strategy"]]
bp = ax2.boxplot(bp_data, labels=["S&P 500\n(VOO)", "10Y Treasury", "VIX Strategy"], patch_artist=True)
for patch, color in zip(bp['boxes'], [colors["S&P 500 (VOO)"], colors["10Y Treasury"], colors["VIX Strategy"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel("Sharpe Ratio", fontsize=11)
ax2.set_title("Sharpe Ratio Distribution Comparison", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: CI visualization
ax3 = axes[1, 0]
strategies = ["S&P 500 (VOO)", "10Y Treasury", "VIX Strategy"]
for i, name in enumerate(strategies):
    ci = ci_results[name]
    ax3.barh(i, ci["original"], color=colors[name], alpha=0.7, height=0.5)
    ax3.errorbar(ci["original"], i, xerr=[[ci["original"]-ci["ci_lower"]], [ci["ci_upper"]-ci["original"]]], 
                 fmt='o', color='black', capsize=5, capthick=2, markersize=8)
ax3.set_yticks(range(len(strategies)))
ax3.set_yticklabels(strategies)
ax3.set_xlabel("Sharpe Ratio", fontsize=11)
ax3.set_title(f"Sharpe Ratio with 95% Confidence Intervals", fontsize=13, fontweight='bold')
ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Difference distribution
ax4 = axes[1, 1]
sharpe_diff = vix_samples - sp500_samples
ax4.hist(sharpe_diff, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax4.axvline(np.mean(sharpe_diff), color='green', linestyle='-', linewidth=2, 
            label=f'Mean diff: {np.mean(sharpe_diff):.3f}')
ci_diff_lower = np.percentile(sharpe_diff, 2.5)
ci_diff_upper = np.percentile(sharpe_diff, 97.5)
ax4.axvline(ci_diff_lower, color='orange', linestyle=':', linewidth=2)
ax4.axvline(ci_diff_upper, color='orange', linestyle=':', linewidth=2, 
            label=f'95% CI: [{ci_diff_lower:.3f}, {ci_diff_upper:.3f}]')
ax4.set_xlabel("Sharpe Difference (VIX Strategy - S&P 500)", fontsize=11)
ax4.set_ylabel("Frequency", fontsize=11)
ax4.set_title("Distribution of Sharpe Ratio Difference", fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../../results/week2/bootstrap_analysis.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Chart saved to 'results/week2/bootstrap_analysis.png'")

plt.show()

# =============================================================================
# Export Results
# =============================================================================
summary_data = []
for name in ["S&P 500 (VOO)", "10Y Treasury", "VIX Strategy"]:
    ci = ci_results[name]
    summary_data.append({
        "Strategy": name,
        "Original_Sharpe": round(ci["original"], 4),
        "CI_Lower_95": round(ci["ci_lower"], 4),
        "CI_Upper_95": round(ci["ci_upper"], 4),
        "CI_Width": round(ci["width"], 4)
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("../../data/week2/bootstrap_results.csv", index=False)
print(f"✓ Results saved to 'data/week2/bootstrap_results.csv'")

# =============================================================================
# Print Summary
# =============================================================================
print(f"\n{'='*70}")
print("SUMMARY - WEEK 2 (VOO 2010-2025)")
print(f"{'='*70}")
print(f"""
Data Period: {df.index[0].date()} to {df.index[-1].date()} (~{len(df)/252:.1f} years)
Optimal Parameters: 3mo lookback, 60% cutoff

VIX Strategy:
  - Sharpe Ratio: {original_sharpe['VIX Strategy']:.3f}
  - 95% CI: [{ci_results['VIX Strategy']['ci_lower']:.3f}, {ci_results['VIX Strategy']['ci_upper']:.3f}]

S&P 500 (VOO):
  - Sharpe Ratio: {original_sharpe['S&P 500 (VOO)']:.3f}
  - 95% CI: [{ci_results['S&P 500 (VOO)']['ci_lower']:.3f}, {ci_results['S&P 500 (VOO)']['ci_upper']:.3f}]

Statistical Significance:
  - Probability VIX beats S&P: {prob_vix_beats_sp500:.1%}
  - 95% CI Overlap: {'Yes' if ci_overlap else 'No'}
  - {'SIGNIFICANT!' if not ci_overlap else 'NOT significant at 95% confidence'}
""")
