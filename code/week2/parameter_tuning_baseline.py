"""
VIX Strategy Parameter Tuning - Baseline Comparison

Creates diverging heatmaps comparing VIX strategy to:
1. S&P 500 (buy & hold stocks) baseline
2. 10Y Treasury (buy & hold bonds) baseline

Color scheme:
- Yellow = matches baseline
- Green = better than baseline  
- Red = worse than baseline
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration - ADJUSTABLE PARAMETER RANGES
# =============================================================================
LOOKBACK_WINDOWS = [63, 126, 189, 252, 315, 378]
LOOKBACK_LABELS = ['3mo', '6mo', '9mo', '1yr', '15mo', '18mo']

PERCENTILE_CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PERCENTILE_LABELS = ['50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%']

STOCK_TICKER = "VOO"
BOND_TICKER = "IEF"
VIX_TICKER = "^VIX"

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# =============================================================================
# Data Download
# =============================================================================
print("=" * 60)
print("VIX STRATEGY - BASELINE COMPARISON HEATMAPS")
print("=" * 60)
print(f"\nDownloading data...")

stock = yf.download(STOCK_TICKER, start=START_DATE, end=END_DATE, progress=False)
bond = yf.download(BOND_TICKER, start=START_DATE, end=END_DATE, progress=False)
vix = yf.download(VIX_TICKER, start=START_DATE, end=END_DATE, progress=False)

if isinstance(stock.columns, pd.MultiIndex):
    stock = stock.droplevel(1, axis=1)
if isinstance(bond.columns, pd.MultiIndex):
    bond = bond.droplevel(1, axis=1)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix.droplevel(1, axis=1)

df_base = pd.concat([
    stock["Close"].rename("Stock"),
    bond["Close"].rename("Bond"),
    vix["Close"].rename("VIX")
], axis=1).dropna()

df_base["Stock_Return"] = df_base["Stock"].pct_change()
df_base["Bond_Return"] = df_base["Bond"].pct_change()

print(f"Data loaded: {len(df_base)} trading days")

# =============================================================================
# Calculate Benchmark Metrics
# =============================================================================
def calc_metrics(returns, risk_free_rate=0.02):
    """Calculate metrics for a return series."""
    returns = returns.dropna()
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol
    cumulative = (1 + returns).cumprod()
    max_dd = ((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min()
    return {
        "annual_return": annual_return * 100,
        "annual_vol": annual_vol * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd * 100
    }

stock_baseline = calc_metrics(df_base["Stock_Return"])
bond_baseline = calc_metrics(df_base["Bond_Return"])

print(f"\nS&P 500 Baseline: Return={stock_baseline['annual_return']:.2f}%, Vol={stock_baseline['annual_vol']:.2f}%, Sharpe={stock_baseline['sharpe']:.3f}")
print(f"10Y Treasury Baseline: Return={bond_baseline['annual_return']:.2f}%, Vol={bond_baseline['annual_vol']:.2f}%, Sharpe={bond_baseline['sharpe']:.3f}")

# =============================================================================
# Calculate Strategy Metrics for All Parameters
# =============================================================================
def calculate_strategy_metrics(df, lookback_window, percentile_cutoff, risk_free_rate=0.02):
    df = df.copy()
    df["VIX_threshold"] = df["VIX"].rolling(window=lookback_window).quantile(percentile_cutoff)
    df["Signal"] = (df["VIX"] > df["VIX_threshold"]).astype(int).shift(1)
    df = df.dropna()
    
    if len(df) < 252:
        return None
    
    df["Strategy_Return"] = np.where(df["Signal"] == 1, df["Bond_Return"], df["Stock_Return"])
    return calc_metrics(df["Strategy_Return"])

print(f"\nCalculating {len(LOOKBACK_WINDOWS) * len(PERCENTILE_CUTOFFS)} parameter combinations...")

results = {
    "annual_return": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
    "annual_vol": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
    "sharpe": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
    "max_drawdown": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS)))
}

for i, lookback in enumerate(LOOKBACK_WINDOWS):
    for j, percentile in enumerate(PERCENTILE_CUTOFFS):
        metrics = calculate_strategy_metrics(df_base, lookback, percentile)
        if metrics:
            for key in results:
                results[key][i, j] = metrics[key]
        else:
            for key in results:
                results[key][i, j] = np.nan

print("Done!")

# =============================================================================
# Create Diverging Heatmap Function
# =============================================================================
def create_diverging_heatmap(ax, data, baseline_value, title, higher_is_better=True, fmt='.1f'):
    """Create a heatmap with yellow at baseline, green for better, red for worse."""
    
    # Calculate difference from baseline
    diff = data - baseline_value
    
    # Determine color mapping
    if higher_is_better:
        # Higher = green, lower = red
        vmax = max(abs(np.nanmax(diff)), abs(np.nanmin(diff)))
        vmin = -vmax
        cmap = 'RdYlGn'
    else:
        # Lower = green (better), higher = red (worse)
        # Flip the sign so positive diff (worse) shows as red
        diff = -diff
        vmax = max(abs(np.nanmax(diff)), abs(np.nanmin(diff)))
        vmin = -vmax
        cmap = 'RdYlGn'
    
    # Create heatmap
    im = ax.imshow(diff, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if higher_is_better:
        cbar.set_label('← Worse | Better →', fontsize=9)
    else:
        cbar.set_label('← Better | Worse →', fontsize=9)
    
    # Add text annotations (show actual values, not differences)
    for i in range(len(LOOKBACK_WINDOWS)):
        for j in range(len(PERCENTILE_CUTOFFS)):
            value = data[i, j]
            diff_val = data[i, j] - baseline_value
            if not np.isnan(value):
                # Color based on difference magnitude
                if abs(diff_val) < 0.1 * abs(baseline_value):
                    text_color = 'black'
                else:
                    text_color = 'white'
                ax.text(j, i, f'{value:{fmt}}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_xticks(range(len(PERCENTILE_CUTOFFS)))
    ax.set_xticklabels(PERCENTILE_LABELS, fontsize=9)
    ax.set_yticks(range(len(LOOKBACK_WINDOWS)))
    ax.set_yticklabels(LOOKBACK_LABELS, fontsize=9)
    ax.set_xlabel("Percentile Cutoff", fontsize=11)
    ax.set_ylabel("Lookback Window", fontsize=11)
    ax.set_title(f"{title}\n(Baseline: {baseline_value:{fmt}})", fontsize=12, fontweight='bold')
    
    return im

# =============================================================================
# Figure 1: Comparison vs S&P 500 Baseline
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))

create_diverging_heatmap(axes1[0, 0], results["annual_return"], stock_baseline["annual_return"],
                         "Annual Return (%)", higher_is_better=True, fmt='.1f')
create_diverging_heatmap(axes1[0, 1], results["annual_vol"], stock_baseline["annual_vol"],
                         "Annual Volatility (%)", higher_is_better=False, fmt='.1f')
create_diverging_heatmap(axes1[1, 0], results["sharpe"], stock_baseline["sharpe"],
                         "Sharpe Ratio", higher_is_better=True, fmt='.2f')
create_diverging_heatmap(axes1[1, 1], results["max_drawdown"], stock_baseline["max_drawdown"],
                         "Max Drawdown (%)", higher_is_better=True, fmt='.1f')  # Less negative is better

fig1.suptitle("VIX Strategy vs S&P 500 Baseline\n(Green = Better than S&P 500, Red = Worse)", 
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig1.savefig("parameter_tuning_vs_sp500.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved: parameter_tuning_vs_sp500.png")

# =============================================================================
# Figure 2: Comparison vs 10Y Treasury Baseline
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))

create_diverging_heatmap(axes2[0, 0], results["annual_return"], bond_baseline["annual_return"],
                         "Annual Return (%)", higher_is_better=True, fmt='.1f')
create_diverging_heatmap(axes2[0, 1], results["annual_vol"], bond_baseline["annual_vol"],
                         "Annual Volatility (%)", higher_is_better=False, fmt='.1f')
create_diverging_heatmap(axes2[1, 0], results["sharpe"], bond_baseline["sharpe"],
                         "Sharpe Ratio", higher_is_better=True, fmt='.2f')
create_diverging_heatmap(axes2[1, 1], results["max_drawdown"], bond_baseline["max_drawdown"],
                         "Max Drawdown (%)", higher_is_better=True, fmt='.1f')

fig2.suptitle("VIX Strategy vs 10Y Treasury Baseline\n(Green = Better than Bonds, Red = Worse)", 
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig("parameter_tuning_vs_bonds.png", dpi=150, bbox_inches="tight")
print("✓ Saved: parameter_tuning_vs_bonds.png")

plt.show()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("BASELINE COMPARISON SUMMARY")
print("=" * 60)

# Count how many parameter combinations beat each baseline
beats_sp500_sharpe = np.sum(results["sharpe"] > stock_baseline["sharpe"])
beats_sp500_vol = np.sum(results["annual_vol"] < stock_baseline["annual_vol"])
beats_bond_sharpe = np.sum(results["sharpe"] > bond_baseline["sharpe"])
beats_bond_return = np.sum(results["annual_return"] > bond_baseline["annual_return"])

total = len(LOOKBACK_WINDOWS) * len(PERCENTILE_CUTOFFS)

print(f"\nVS S&P 500:")
print(f"  Combinations with higher Sharpe: {beats_sp500_sharpe}/{total} ({beats_sp500_sharpe/total*100:.0f}%)")
print(f"  Combinations with lower volatility: {beats_sp500_vol}/{total} ({beats_sp500_vol/total*100:.0f}%)")

print(f"\nVS 10Y Treasury:")
print(f"  Combinations with higher Sharpe: {beats_bond_sharpe}/{total} ({beats_bond_sharpe/total*100:.0f}%)")
print(f"  Combinations with higher return: {beats_bond_return}/{total} ({beats_bond_return/total*100:.0f}%)")

