"""
VIX Strategy Parameter Tuning - Week 2 (VOO 2010-2025)

Varies two parameters:
1. Lookback window (for rolling percentile calculation)
2. Percentile cutoff (threshold for switching to bonds)

Generates heatmaps for key performance metrics.
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
print("VIX STRATEGY PARAMETER TUNING - WEEK 2 (VOO 2010-2025)")
print("=" * 70)
print(f"\nDownloading data from {START_DATE} to {END_DATE}...")

df_base = download_data(TICKERS, START_DATE, END_DATE)
df_base["Stock_Return"] = df_base["Stock"].pct_change()
df_base["Bond_Return"] = df_base["Bond"].pct_change()

print(f"Data loaded: {len(df_base)} trading days (~{len(df_base)/252:.1f} years)")
print(f"Period: {df_base.index[0].date()} to {df_base.index[-1].date()}")

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
# Parameter Tuning Function
# =============================================================================
def calculate_strategy_metrics(df, lookback_window, percentile_cutoff, risk_free_rate=RISK_FREE_RATE):
    """Calculate strategy metrics for given parameters."""
    df = df.copy()
    df["VIX_threshold"] = df["VIX"].rolling(window=lookback_window).quantile(percentile_cutoff)
    df["Signal"] = (df["VIX"] > df["VIX_threshold"]).astype(int).shift(1)
    df = df.dropna()
    
    if len(df) < 252:
        return None
    
    df["Strategy_Return"] = np.where(df["Signal"] == 1, df["Bond_Return"], df["Stock_Return"])
    return calculate_metrics(df["Strategy_Return"], risk_free_rate)

# =============================================================================
# Run Parameter Grid
# =============================================================================
print(f"\nTesting {len(LOOKBACK_WINDOWS)} lookback windows × {len(PERCENTILE_CUTOFFS)} percentile cutoffs...")

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

print("Parameter sweep complete!\n")

# =============================================================================
# Find Best Parameters
# =============================================================================
best_sharpe_idx = np.unravel_index(np.nanargmax(results["sharpe"]), results["sharpe"].shape)
best_sharpe = results["sharpe"][best_sharpe_idx]
best_params = (LOOKBACK_WINDOWS[best_sharpe_idx[0]], PERCENTILE_CUTOFFS[best_sharpe_idx[1]], 
               LOOKBACK_LABELS[best_sharpe_idx[0]], PERCENTILE_LABELS[best_sharpe_idx[1]])

print(f"Optimal Parameters: Lookback={best_params[2]}, Cutoff={best_params[3]}")
print(f"Optimal Sharpe: {best_sharpe:.3f}")

# =============================================================================
# Create Heatmaps
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

heatmap_configs = [
    ("annual_return", "Annual Return (%)", "RdYlGn", axes[0, 0]),
    ("annual_vol", "Annual Volatility (%)", "RdYlGn_r", axes[0, 1]),
    ("sharpe", "Sharpe Ratio", "RdYlGn", axes[1, 0]),
    ("max_drawdown", "Max Drawdown (%)", "RdYlGn", axes[1, 1]),
]

for key, title, cmap, ax in heatmap_configs:
    data = results[key]
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    
    for i in range(len(LOOKBACK_WINDOWS)):
        for j in range(len(PERCENTILE_CUTOFFS)):
            value = data[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value - np.nanmean(data)) > np.nanstd(data) else 'black'
                fmt = '.1f' if key != 'sharpe' else '.2f'
                ax.text(j, i, f'{value:{fmt}}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_xticks(range(len(PERCENTILE_CUTOFFS)))
    ax.set_xticklabels(PERCENTILE_LABELS, fontsize=9)
    ax.set_yticks(range(len(LOOKBACK_WINDOWS)))
    ax.set_yticklabels(LOOKBACK_LABELS, fontsize=9)
    ax.set_xlabel("Percentile Cutoff", fontsize=11)
    ax.set_ylabel("Lookback Window", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

fig.suptitle("VIX Strategy Parameter Tuning (VOO 2010-2025)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../../results/week2/parameter_tuning.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: results/week2/parameter_tuning.png")

# =============================================================================
# Export CSV
# =============================================================================
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
            "Max_Drawdown_Pct": results["max_drawdown"][i, j]
        })

param_df = pd.DataFrame(rows)
param_df.to_csv("../../data/week2/parameter_tuning.csv", index=False)
print(f"✓ Saved: data/week2/parameter_tuning.csv")

plt.show()
