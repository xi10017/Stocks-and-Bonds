"""
VIX Strategy Parameter Tuning

Varies two parameters:
1. Lookback window (for rolling percentile calculation)
2. Percentile cutoff (threshold for switching to bonds)

Generates heatmaps for key performance metrics.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration - ADJUSTABLE PARAMETER RANGES
# =============================================================================
# Lookback windows to test (in trading days)
LOOKBACK_WINDOWS = [63, 126, 189, 252, 315, 378]  # ~3mo, 6mo, 9mo, 1yr, 1.25yr, 1.5yr
LOOKBACK_LABELS = ['3mo', '6mo', '9mo', '1yr', '15mo', '18mo']

# Percentile cutoffs to test
PERCENTILE_CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PERCENTILE_LABELS = ['50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%']

# Tickers
STOCK_TICKER = "VOO"
BOND_TICKER = "IEF"
VIX_TICKER = "^VIX"

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# =============================================================================
# Data Download
# =============================================================================
print("=" * 60)
print("VIX STRATEGY PARAMETER TUNING")
print("=" * 60)
print(f"\nDownloading data from {START_DATE} to {END_DATE}...")

stock = yf.download(STOCK_TICKER, start=START_DATE, end=END_DATE, progress=False)
bond = yf.download(BOND_TICKER, start=START_DATE, end=END_DATE, progress=False)
vix = yf.download(VIX_TICKER, start=START_DATE, end=END_DATE, progress=False)

# Handle multi-level columns
if isinstance(stock.columns, pd.MultiIndex):
    stock = stock.droplevel(1, axis=1)
if isinstance(bond.columns, pd.MultiIndex):
    bond = bond.droplevel(1, axis=1)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix.droplevel(1, axis=1)

# Combine data
df_base = pd.concat([
    stock["Close"].rename("Stock"),
    bond["Close"].rename("Bond"),
    vix["Close"].rename("VIX")
], axis=1).dropna()

df_base["Stock_Return"] = df_base["Stock"].pct_change()
df_base["Bond_Return"] = df_base["Bond"].pct_change()

print(f"Data loaded: {len(df_base)} trading days")

# =============================================================================
# Parameter Tuning Function
# =============================================================================
def calculate_strategy_metrics(df, lookback_window, percentile_cutoff, risk_free_rate=0.02):
    """Calculate strategy metrics for given parameters."""
    df = df.copy()
    
    # Calculate rolling percentile threshold
    df["VIX_threshold"] = df["VIX"].rolling(window=lookback_window).quantile(percentile_cutoff)
    
    # Generate signal (shifted to avoid look-ahead bias)
    df["Signal"] = (df["VIX"] > df["VIX_threshold"]).astype(int).shift(1)
    
    # Drop NaN rows
    df = df.dropna()
    
    if len(df) < 252:  # Need at least 1 year of data
        return None
    
    # Calculate strategy returns
    df["Strategy_Return"] = np.where(
        df["Signal"] == 1,
        df["Bond_Return"],
        df["Stock_Return"]
    )
    
    # Calculate metrics
    returns = df["Strategy_Return"]
    
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Percent time in bonds
    pct_bonds = df["Signal"].mean()
    
    return {
        "annual_return": annual_return * 100,
        "annual_vol": annual_vol * 100,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown * 100,
        "pct_bonds": pct_bonds * 100
    }

# =============================================================================
# Run Parameter Grid
# =============================================================================
print(f"\nTesting {len(LOOKBACK_WINDOWS)} lookback windows × {len(PERCENTILE_CUTOFFS)} percentile cutoffs...")
print(f"Total combinations: {len(LOOKBACK_WINDOWS) * len(PERCENTILE_CUTOFFS)}")

# Initialize result matrices
results = {
    "annual_return": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
    "annual_vol": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
    "sharpe": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
    "max_drawdown": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
    "pct_bonds": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS)))
}

for i, lookback in enumerate(LOOKBACK_WINDOWS):
    for j, percentile in enumerate(PERCENTILE_CUTOFFS):
        metrics = calculate_strategy_metrics(df_base, lookback, percentile)
        if metrics:
            results["annual_return"][i, j] = metrics["annual_return"]
            results["annual_vol"][i, j] = metrics["annual_vol"]
            results["sharpe"][i, j] = metrics["sharpe"]
            results["max_drawdown"][i, j] = metrics["max_drawdown"]
            results["pct_bonds"][i, j] = metrics["pct_bonds"]
        else:
            for key in results:
                results[key][i, j] = np.nan

print("Parameter sweep complete!\n")

# =============================================================================
# Create Heatmaps
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Define heatmap configurations
heatmap_configs = [
    ("annual_return", "Annual Return (%)", "RdYlGn", axes[0, 0]),
    ("annual_vol", "Annual Volatility (%)", "RdYlGn_r", axes[0, 1]),  # Reversed: lower is better
    ("sharpe", "Sharpe Ratio", "RdYlGn", axes[0, 2]),
    ("max_drawdown", "Max Drawdown (%)", "RdYlGn", axes[1, 0]),  # More negative is worse, but values are negative
    ("pct_bonds", "% Time in Bonds", "Blues", axes[1, 1]),
]

for key, title, cmap, ax in heatmap_configs:
    data = results[key]
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(len(LOOKBACK_WINDOWS)):
        for j in range(len(PERCENTILE_CUTOFFS)):
            value = data[i, j]
            if not np.isnan(value):
                # Choose text color based on background
                text_color = 'white' if abs(value - np.nanmean(data)) > np.nanstd(data) else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_xticks(range(len(PERCENTILE_CUTOFFS)))
    ax.set_xticklabels(PERCENTILE_LABELS, fontsize=9)
    ax.set_yticks(range(len(LOOKBACK_WINDOWS)))
    ax.set_yticklabels(LOOKBACK_LABELS, fontsize=9)
    ax.set_xlabel("Percentile Cutoff", fontsize=11)
    ax.set_ylabel("Lookback Window", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

# Add benchmark comparison in the last subplot
ax_text = axes[1, 2]
ax_text.axis('off')

# Calculate benchmark metrics
stock_returns = df_base["Stock_Return"].dropna()
bond_returns = df_base["Bond_Return"].dropna()

def calc_benchmark(returns, name):
    total_ret = (1 + returns).prod() - 1
    years = len(returns) / 252
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol
    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min()
    return ann_ret * 100, ann_vol * 100, sharpe, max_dd * 100

stock_metrics = calc_benchmark(stock_returns, "S&P 500")
bond_metrics = calc_benchmark(bond_returns, "10Y Treasury")

# Find best parameters
best_sharpe_idx = np.unravel_index(np.nanargmax(results["sharpe"]), results["sharpe"].shape)
best_sharpe = results["sharpe"][best_sharpe_idx]
best_sharpe_params = (LOOKBACK_LABELS[best_sharpe_idx[0]], PERCENTILE_LABELS[best_sharpe_idx[1]])

best_vol_idx = np.unravel_index(np.nanargmin(results["annual_vol"]), results["annual_vol"].shape)
best_vol = results["annual_vol"][best_vol_idx]
best_vol_params = (LOOKBACK_LABELS[best_vol_idx[0]], PERCENTILE_LABELS[best_vol_idx[1]])

summary_text = f"""
PARAMETER TUNING SUMMARY
========================

BENCHMARKS (Buy & Hold)
-----------------------
S&P 500:
  Annual Return: {stock_metrics[0]:.2f}%
  Annual Volatility: {stock_metrics[1]:.2f}%
  Sharpe Ratio: {stock_metrics[2]:.3f}
  Max Drawdown: {stock_metrics[3]:.2f}%

10Y Treasury:
  Annual Return: {bond_metrics[0]:.2f}%
  Annual Volatility: {bond_metrics[1]:.2f}%
  Sharpe Ratio: {bond_metrics[2]:.3f}
  Max Drawdown: {bond_metrics[3]:.2f}%

BEST VIX STRATEGY PARAMETERS
----------------------------
Highest Sharpe Ratio: {best_sharpe:.3f}
  → Lookback: {best_sharpe_params[0]}, Cutoff: {best_sharpe_params[1]}

Lowest Volatility: {best_vol:.2f}%
  → Lookback: {best_vol_params[0]}, Cutoff: {best_vol_params[1]}

Original Parameters (1yr, 75%):
  Sharpe: {results["sharpe"][3, 5]:.3f}
  Volatility: {results["annual_vol"][3, 5]:.2f}%
"""

ax_text.text(0.1, 0.95, summary_text, transform=ax_text.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("VIX Strategy Parameter Sensitivity Analysis", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("parameter_tuning.png", dpi=150, bbox_inches="tight")
plt.show()

print("✓ Heatmaps saved to 'parameter_tuning.png'")

# =============================================================================
# Export Results to CSV
# =============================================================================
# Create a long-form DataFrame with all results
rows = []
for i, lookback in enumerate(LOOKBACK_WINDOWS):
    for j, percentile in enumerate(PERCENTILE_CUTOFFS):
        rows.append({
            "Lookback_Days": lookback,
            "Lookback_Label": LOOKBACK_LABELS[i],
            "Percentile_Cutoff": percentile,
            "Percentile_Label": PERCENTILE_LABELS[j],
            "Annual_Return_Pct": results["annual_return"][i, j],
            "Annual_Volatility_Pct": results["annual_vol"][i, j],
            "Sharpe_Ratio": results["sharpe"][i, j],
            "Max_Drawdown_Pct": results["max_drawdown"][i, j],
            "Pct_Time_Bonds": results["pct_bonds"][i, j]
        })

param_df = pd.DataFrame(rows)
param_df.to_csv("parameter_tuning.csv", index=False)
print("✓ Parameter results saved to 'parameter_tuning.csv'")

# =============================================================================
# Print Summary
# =============================================================================
print("\n" + "=" * 60)
print("PARAMETER TUNING RESULTS")
print("=" * 60)
print(f"\nBest Sharpe Ratio: {best_sharpe:.3f}")
print(f"  Parameters: Lookback = {best_sharpe_params[0]}, Cutoff = {best_sharpe_params[1]}")
print(f"\nLowest Volatility: {best_vol:.2f}%")
print(f"  Parameters: Lookback = {best_vol_params[0]}, Cutoff = {best_vol_params[1]}")
print(f"\nOriginal Parameters (1yr lookback, 75% cutoff):")
print(f"  Sharpe: {results['sharpe'][3, 5]:.3f}, Volatility: {results['annual_vol'][3, 5]:.2f}%")

