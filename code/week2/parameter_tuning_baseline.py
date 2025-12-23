"""
VIX Strategy Parameter Tuning - Baseline Comparison (VOO 2010-2025)

Creates diverging heatmaps comparing VIX strategy to:
1. S&P 500 (buy & hold stocks) baseline
2. 10Y Treasury (buy & hold bonds) baseline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import download_data, calculate_metrics

# =============================================================================
# Configuration
# =============================================================================
LOOKBACK_WINDOWS = [63, 126, 189, 252, 315, 378]
LOOKBACK_LABELS = ['3mo', '6mo', '9mo', '1yr', '15mo', '18mo']
PERCENTILE_CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PERCENTILE_LABELS = ['50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%']

TICKERS = {"Stock": "VOO", "Bond": "IEF", "VIX": "^VIX"}
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RISK_FREE_RATE = 0.02

# =============================================================================
# Data Download
# =============================================================================
print("=" * 70)
print("VIX STRATEGY - BASELINE COMPARISON HEATMAPS (VOO 2010-2025)")
print("=" * 70)
print(f"\nDownloading data...")

df_base = download_data(TICKERS, START_DATE, END_DATE)
df_base["Stock_Return"] = df_base["Stock"].pct_change()
df_base["Bond_Return"] = df_base["Bond"].pct_change()

print(f"Data loaded: {len(df_base)} trading days")

# =============================================================================
# Calculate Benchmark Metrics
# =============================================================================
stock_baseline = calculate_metrics(df_base["Stock_Return"], RISK_FREE_RATE)
bond_baseline = calculate_metrics(df_base["Bond_Return"], RISK_FREE_RATE)

print(f"\nS&P 500 (VOO) Baseline:")
print(f"  Return={stock_baseline['annual_return']*100:.2f}%, Vol={stock_baseline['annual_vol']*100:.2f}%, Sharpe={stock_baseline['sharpe']:.3f}")
print(f"10Y Treasury Baseline:")
print(f"  Return={bond_baseline['annual_return']*100:.2f}%, Vol={bond_baseline['annual_vol']*100:.2f}%, Sharpe={bond_baseline['sharpe']:.3f}")

# =============================================================================
# Parameter Tuning
# =============================================================================
def calculate_strategy_metrics(df, lookback_window, percentile_cutoff, risk_free_rate=RISK_FREE_RATE):
    df = df.copy()
    df["VIX_threshold"] = df["VIX"].rolling(window=lookback_window).quantile(percentile_cutoff)
    df["Signal"] = (df["VIX"] > df["VIX_threshold"]).astype(int).shift(1)
    df = df.dropna()
    if len(df) < 252:
        return None
    df["Strategy_Return"] = np.where(df["Signal"] == 1, df["Bond_Return"], df["Stock_Return"])
    return calculate_metrics(df["Strategy_Return"], risk_free_rate)

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
            results["annual_return"][i, j] = metrics["annual_return"] * 100
            results["annual_vol"][i, j] = metrics["annual_vol"] * 100
            results["sharpe"][i, j] = metrics["sharpe"]
            results["max_drawdown"][i, j] = metrics["max_drawdown"] * 100
        else:
            for key in results:
                results[key][i, j] = np.nan

print("Done!")

# =============================================================================
# Create Diverging Heatmap Function
# =============================================================================
def create_diverging_heatmap(ax, data, baseline_value, title, higher_is_better=True, fmt='.1f'):
    """Create a heatmap with yellow at baseline, green for better, red for worse."""
    diff = data - baseline_value
    if not higher_is_better:
        diff = -diff
    vmax = max(abs(np.nanmax(diff)), abs(np.nanmin(diff)))
    vmin = -vmax
    
    im = ax.imshow(diff, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    
    for i in range(len(LOOKBACK_WINDOWS)):
        for j in range(len(PERCENTILE_CUTOFFS)):
            value = data[i, j]
            if not np.isnan(value):
                text_color = 'black' if abs(value - baseline_value) < 0.1 * abs(baseline_value) else 'white'
                ax.text(j, i, f'{value:{fmt}}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_xticks(range(len(PERCENTILE_CUTOFFS)))
    ax.set_xticklabels(PERCENTILE_LABELS, fontsize=9)
    ax.set_yticks(range(len(LOOKBACK_WINDOWS)))
    ax.set_yticklabels(LOOKBACK_LABELS, fontsize=9)
    ax.set_xlabel("Percentile Cutoff", fontsize=11)
    ax.set_ylabel("Lookback Window", fontsize=11)
    ax.set_title(f"{title}\n(Baseline: {baseline_value:{fmt}})", fontsize=12, fontweight='bold')

# =============================================================================
# Figure 1: Comparison vs S&P 500 Baseline
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))

create_diverging_heatmap(axes1[0, 0], results["annual_return"], stock_baseline["annual_return"]*100,
                         "Annual Return (%)", higher_is_better=True, fmt='.1f')
create_diverging_heatmap(axes1[0, 1], results["annual_vol"], stock_baseline["annual_vol"]*100,
                         "Annual Volatility (%)", higher_is_better=False, fmt='.1f')
create_diverging_heatmap(axes1[1, 0], results["sharpe"], stock_baseline["sharpe"],
                         "Sharpe Ratio", higher_is_better=True, fmt='.2f')
create_diverging_heatmap(axes1[1, 1], results["max_drawdown"], stock_baseline["max_drawdown"]*100,
                         "Max Drawdown (%)", higher_is_better=True, fmt='.1f')

fig1.suptitle("VIX Strategy vs S&P 500 Baseline (VOO 2010-2025)\n(Green = Better, Red = Worse)", 
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig1.savefig("../../results/week2/parameter_tuning_vs_sp500.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: results/week2/parameter_tuning_vs_sp500.png")

# =============================================================================
# Figure 2: Comparison vs 10Y Treasury Baseline
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))

create_diverging_heatmap(axes2[0, 0], results["annual_return"], bond_baseline["annual_return"]*100,
                         "Annual Return (%)", higher_is_better=True, fmt='.1f')
create_diverging_heatmap(axes2[0, 1], results["annual_vol"], bond_baseline["annual_vol"]*100,
                         "Annual Volatility (%)", higher_is_better=False, fmt='.1f')
create_diverging_heatmap(axes2[1, 0], results["sharpe"], bond_baseline["sharpe"],
                         "Sharpe Ratio", higher_is_better=True, fmt='.2f')
create_diverging_heatmap(axes2[1, 1], results["max_drawdown"], bond_baseline["max_drawdown"]*100,
                         "Max Drawdown (%)", higher_is_better=True, fmt='.1f')

fig2.suptitle("VIX Strategy vs 10Y Treasury Baseline (VOO 2010-2025)\n(Green = Better, Red = Worse)", 
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig("../../results/week2/parameter_tuning_vs_bonds.png", dpi=150, bbox_inches="tight")
print("✓ Saved: results/week2/parameter_tuning_vs_bonds.png")

plt.show()

# =============================================================================
# Summary
# =============================================================================
beats_sp500_sharpe = np.sum(results["sharpe"] > stock_baseline["sharpe"])
beats_sp500_vol = np.sum(results["annual_vol"] < stock_baseline["annual_vol"]*100)
beats_bond_sharpe = np.sum(results["sharpe"] > bond_baseline["sharpe"])
beats_bond_return = np.sum(results["annual_return"] > bond_baseline["annual_return"]*100)
total = len(LOOKBACK_WINDOWS) * len(PERCENTILE_CUTOFFS)

print(f"\n{'='*70}")
print("BASELINE COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"\nVS S&P 500:")
print(f"  Combinations with higher Sharpe: {beats_sp500_sharpe}/{total} ({beats_sp500_sharpe/total*100:.0f}%)")
print(f"  Combinations with lower volatility: {beats_sp500_vol}/{total} ({beats_sp500_vol/total*100:.0f}%)")
print(f"\nVS 10Y Treasury:")
print(f"  Combinations with higher Sharpe: {beats_bond_sharpe}/{total} ({beats_bond_sharpe/total*100:.0f}%)")
print(f"  Combinations with higher return: {beats_bond_return}/{total} ({beats_bond_return/total*100:.0f}%)")
