"""
VIX Strategy Parameter Tuning - Week 3 (SPY 2002-2025) with Transaction Costs

Uses SPY instead of VOO for longer historical data (~22 years vs 14 years)
Includes transaction costs when rebalancing between stocks and bonds.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import download_data, calculate_metrics, calculate_progressive_bond_allocation, calculate_progressive_strategy_returns, calculate_sharpe_ratio, calculate_annual_return, calculate_annual_volatility

# =============================================================================
# Configuration
# =============================================================================
LOOKBACK_WINDOWS = [63, 126, 189, 252, 315, 378]
LOOKBACK_LABELS = ['3mo', '6mo', '9mo', '1yr', '15mo', '18mo']
PERCENTILE_CUTOFFS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PERCENTILE_LABELS = ['60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%']

# Progressive allocation scaling options
SCALING_OPTIONS = ['linear', 'sigmoid', 'exponential', 'power']
SCALING_LABELS = ['Linear', 'Sigmoid', 'Exp', 'Power']

TICKERS = {"Stock": "SPY", "Bond": "IEF", "VIX": "^VIX"}
START_DATE = "2002-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RISK_FREE_RATE = 0.04  # 4% - matches comprehensive analysis in new/src/run_analysis.py
TRANSACTION_COST = 0.001  # 0.1% (10 bps) per trade - applied to absolute change in allocation

# =============================================================================
# Data Download
# =============================================================================
print("=" * 70)
print("VIX STRATEGY PARAMETER TUNING - WEEK 3 (SPY 2002-2025)")
print("WITH TRANSACTION COSTS")
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

print(f"\nS&P 500 (SPY) Baseline:")
print(f"  Return={stock_baseline['annual_return']*100:.2f}%, Sharpe={stock_baseline['sharpe']:.3f}")
print(f"10Y Treasury Baseline:")
print(f"  Return={bond_baseline['annual_return']*100:.2f}%, Sharpe={bond_baseline['sharpe']:.3f}")

# =============================================================================
# Parameter Tuning
# =============================================================================
def calculate_strategy_metrics(df, lookback_window, percentile_cutoff, 
                              scaling='linear', risk_free_rate=RISK_FREE_RATE,
                              transaction_cost=TRANSACTION_COST):
    """
    Calculate strategy metrics with progressive allocation and transaction costs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with VIX, Stock_Return, Bond_Return
    lookback_window : int
        Rolling window for VIX percentile
    percentile_cutoff : float
        Percentile threshold (e.g., 0.70 for 70th percentile)
    scaling : str
        Scaling function: 'linear', 'sigmoid', 'exponential', 'power'
    risk_free_rate : float
        Risk-free rate for Sharpe calculation
    transaction_cost : float
        Transaction cost rate (e.g., 0.001 for 0.1% or 10 bps)
        Applied to absolute change in allocation when rebalancing
    """
    df = df.copy()
    df["VIX_threshold"] = df["VIX"].rolling(window=lookback_window).quantile(percentile_cutoff)
    
    # Calculate progressive bond allocation
    # Shift threshold to avoid look-ahead bias
    df["VIX_threshold_shifted"] = df["VIX_threshold"].shift(1)
    df["Bond_Allocation"] = calculate_progressive_bond_allocation(
        df["VIX"], 
        df["VIX_threshold_shifted"],
        scaling=scaling
    )
    df = df.dropna()
    
    if len(df) < 252:
        return None
    
    # Calculate strategy returns using progressive allocation
    df["Strategy_Return"] = calculate_progressive_strategy_returns(
        df["Stock_Return"],
        df["Bond_Return"],
        df["Bond_Allocation"]
    )
    
    # Calculate transaction costs
    # Cost is applied when allocation changes (rebalancing occurs)
    # Cost = abs(change_in_allocation) * transaction_cost_rate
    df["Allocation_Change"] = df["Bond_Allocation"].diff().abs()
    df["Transaction_Cost"] = df["Allocation_Change"] * transaction_cost
    # First row has no previous allocation, so no cost
    df.loc[df.index[0], "Transaction_Cost"] = 0.0
    
    # Subtract transaction costs from strategy returns
    df["Strategy_Return_Net"] = df["Strategy_Return"] - df["Transaction_Cost"]
    
    metrics = calculate_metrics(df["Strategy_Return_Net"], risk_free_rate)
    
    # Validation: Check for suspiciously high Sharpe ratios and verify calculations
    if metrics and metrics["sharpe"] > 2.5:
        # Verify Sharpe calculation manually
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = df["Strategy_Return_Net"] - daily_rf
        manual_sharpe = (excess_returns.mean() / df["Strategy_Return_Net"].std()) * np.sqrt(252)
        
        # Verify return calculation
        sample_idx = df.index[100:105]
        for idx in sample_idx[:2]:
            stock_ret = df.loc[idx, 'Stock_Return']
            bond_ret = df.loc[idx, 'Bond_Return']
            bond_alloc = df.loc[idx, 'Bond_Allocation']
            strategy_ret = df.loc[idx, 'Strategy_Return']
            expected_ret = (1 - bond_alloc) * stock_ret + bond_alloc * bond_ret
            if abs(strategy_ret - expected_ret) > 1e-10:
                print(f"  ERROR: Return calculation mismatch at {idx.date()}: {strategy_ret:.6f} vs {expected_ret:.6f}")
        
        # Transaction cost diagnostics
        if metrics["sharpe"] > 3.0:
            total_turnover = df["Allocation_Change"].sum()
            avg_daily_turnover = df["Allocation_Change"].mean()
            total_cost = df["Transaction_Cost"].sum()
            annual_cost = total_cost * (252 / len(df))
            print(f"    Transaction Costs: Total={total_cost*100:.2f}%, Annualized={annual_cost*100:.2f}%")
            print(f"    Total Turnover: {total_turnover*100:.1f}%, Avg Daily: {avg_daily_turnover*100:.3f}%")
            print(f"    Rebalancing Days: {(df['Allocation_Change'] > 0.001).sum()} ({(df['Allocation_Change'] > 0.001).mean():.1%})")
        
        # Diagnostic info
        if metrics["sharpe"] > 3.0:
            print(f"  WARNING: Very high Sharpe ({metrics['sharpe']:.3f}, manual check: {manual_sharpe:.3f})")
            print(f"    Annual Return: {metrics['annual_return']*100:.2f}%, Vol: {metrics['annual_vol']*100:.2f}%")
            print(f"    Avg Bond Allocation: {df['Bond_Allocation'].mean():.2%}")
            print(f"    Days with >50% bonds: {(df['Bond_Allocation'] > 0.5).sum()} ({(df['Bond_Allocation'] > 0.5).mean():.1%})")
            print(f"    Days with >90% bonds: {(df['Bond_Allocation'] > 0.9).sum()} ({(df['Bond_Allocation'] > 0.9).mean():.1%})")
            print(f"    Max bond allocation: {df['Bond_Allocation'].max():.2%}")
            print(f"    Min bond allocation: {df['Bond_Allocation'].min():.2%}")
            
            # Compare to pure bonds
            bond_only_sharpe = calculate_sharpe_ratio(df['Bond_Return'], risk_free_rate)
            bond_only_return = calculate_annual_return(df['Bond_Return']) * 100
            bond_only_vol = calculate_annual_volatility(df['Bond_Return']) * 100
            print(f"    Pure Bonds (same period): Return={bond_only_return:.2f}%, Vol={bond_only_vol:.2f}%, Sharpe={bond_only_sharpe:.3f}")
            
            # Check if returns match expected weighted average
            expected_return = (1 - df['Bond_Allocation'].mean()) * calculate_annual_return(df['Stock_Return']) + df['Bond_Allocation'].mean() * calculate_annual_return(df['Bond_Return'])
            actual_return = metrics['annual_return']
            print(f"    Expected return (weighted avg): {expected_return*100:.2f}%, Actual: {actual_return*100:.2f}%")
            
            print(f"    Stock-only equivalent vol: {df['Stock_Return'].std() * np.sqrt(252) * 100:.2f}%")
    
    return metrics

print(f"\nTesting {len(LOOKBACK_WINDOWS) * len(PERCENTILE_CUTOFFS) * len(SCALING_OPTIONS)} parameter combinations...")
print(f"  - {len(LOOKBACK_WINDOWS)} lookback windows")
print(f"  - {len(PERCENTILE_CUTOFFS)} percentile cutoffs")
print(f"  - {len(SCALING_OPTIONS)} scaling functions: {', '.join(SCALING_LABELS)}")
print(f"  - Transaction Cost: {TRANSACTION_COST*100:.2f}% ({TRANSACTION_COST*10000:.0f} bps) per rebalancing trade")

# Store results for each scaling function
all_results = {}
for scaling in SCALING_OPTIONS:
    all_results[scaling] = {
        "annual_return": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
        "annual_vol": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
        "sharpe": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS))),
        "max_drawdown": np.zeros((len(LOOKBACK_WINDOWS), len(PERCENTILE_CUTOFFS)))
    }

for scaling_idx, scaling in enumerate(SCALING_OPTIONS):
    print(f"\nTesting {SCALING_LABELS[scaling_idx]} scaling...")
    for i, lookback in enumerate(LOOKBACK_WINDOWS):
        for j, percentile in enumerate(PERCENTILE_CUTOFFS):
            metrics = calculate_strategy_metrics(df_base, lookback, percentile, scaling=scaling)
            if metrics:
                all_results[scaling]["annual_return"][i, j] = metrics["annual_return"] * 100
                all_results[scaling]["annual_vol"][i, j] = metrics["annual_vol"] * 100
                all_results[scaling]["sharpe"][i, j] = metrics["sharpe"]
                all_results[scaling]["max_drawdown"][i, j] = metrics["max_drawdown"] * 100
            else:
                for key in all_results[scaling]:
                    all_results[scaling][key][i, j] = np.nan

# Find best overall parameters across all scaling functions
best_sharpe = -np.inf
best_params = None
best_scaling = None

for scaling in SCALING_OPTIONS:
    sharpe_data = all_results[scaling]["sharpe"]
    if np.any(~np.isnan(sharpe_data)):
        idx = np.unravel_index(np.nanargmax(sharpe_data), sharpe_data.shape)
        sharpe_val = sharpe_data[idx]
        if sharpe_val > best_sharpe:
            best_sharpe = sharpe_val
            best_params = (LOOKBACK_WINDOWS[idx[0]], PERCENTILE_CUTOFFS[idx[1]], 
                          LOOKBACK_LABELS[idx[0]], PERCENTILE_LABELS[idx[1]])
            best_scaling = scaling

print(f"\nOptimal Parameters: Lookback={best_params[2]}, Cutoff={best_params[3]}, Scaling={best_scaling}")
print(f"Optimal Sharpe: {best_sharpe:.3f}")

# Use best scaling for main results (for visualization)
results = all_results[best_scaling]
scaling_label = SCALING_LABELS[SCALING_OPTIONS.index(best_scaling)]

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

fig.suptitle(f"VIX Strategy Parameter Tuning (SPY 2002-2025) with Transaction Costs\nBest Scaling: {scaling_label} | TC: {TRANSACTION_COST*100:.2f}%", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../../results/week3/parameter_tuning_tc.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: results/week3/parameter_tuning_tc.png")

# =============================================================================
# Export CSV (for all scaling functions)
# =============================================================================
rows = []
for scaling in SCALING_OPTIONS:
    scaling_label = SCALING_LABELS[SCALING_OPTIONS.index(scaling)]
    for i, lookback in enumerate(LOOKBACK_WINDOWS):
        for j, percentile in enumerate(PERCENTILE_CUTOFFS):
            rows.append({
                "Scaling": scaling,
                "Scaling_Label": scaling_label,
                "Lookback_Days": lookback,
                "Lookback_Label": LOOKBACK_LABELS[i],
                "Percentile_Cutoff": percentile,
                "Percentile_Label": PERCENTILE_LABELS[j],
                "Annual_Return_Pct": all_results[scaling]["annual_return"][i, j],
                "Annual_Volatility_Pct": all_results[scaling]["annual_vol"][i, j],
                "Sharpe_Ratio": all_results[scaling]["sharpe"][i, j],
                "Max_Drawdown_Pct": all_results[scaling]["max_drawdown"][i, j]
            })

param_df = pd.DataFrame(rows)
param_df.to_csv("../../data/week3/parameter_tuning_progressive_tc.csv", index=False)
print(f"✓ Saved: data/week3/parameter_tuning_progressive_tc.csv")

# =============================================================================
# Baseline Comparison Heatmaps - All Scaling Functions Side by Side
# =============================================================================
def create_diverging_heatmap(ax, data, baseline_value, title, higher_is_better=True, fmt='.1f'):
    """Create a single diverging heatmap."""
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
                fmt_str = '.1f' if fmt == '.1f' else '.2f'
                ax.text(j, i, f'{value:{fmt_str}}', ha='center', va='center', 
                       color=text_color, fontsize=7, fontweight='bold')
    
    ax.set_xticks(range(len(PERCENTILE_CUTOFFS)))
    ax.set_xticklabels(PERCENTILE_LABELS, fontsize=8)
    ax.set_yticks(range(len(LOOKBACK_WINDOWS)))
    ax.set_yticklabels(LOOKBACK_LABELS, fontsize=8)
    ax.set_xlabel("Percentile Cutoff", fontsize=9)
    ax.set_ylabel("Lookback Window", fontsize=9)
    ax.set_title(f"{title}\n(Baseline: {baseline_value:{fmt}})", fontsize=10, fontweight='bold')
    return im

# vs S&P 500 - 4x4 grid (4 metrics × 4 scaling functions)
fig1, axes1 = plt.subplots(4, 4, figsize=(20, 18))

metrics_config = [
    ("annual_return", "Annual Return (%)", stock_baseline["annual_return"]*100, True, '.1f'),
    ("annual_vol", "Annual Volatility (%)", stock_baseline["annual_vol"]*100, False, '.1f'),
    ("sharpe", "Sharpe Ratio", stock_baseline["sharpe"], True, '.2f'),
    ("max_drawdown", "Max Drawdown (%)", stock_baseline["max_drawdown"]*100, True, '.1f'),
]

for metric_idx, (metric_key, metric_title, baseline_val, higher_better, fmt) in enumerate(metrics_config):
    for scaling_idx, scaling in enumerate(SCALING_OPTIONS):
        scaling_label = SCALING_LABELS[scaling_idx]
        ax = axes1[metric_idx, scaling_idx]
        data = all_results[scaling][metric_key]
        
        # Convert to percentage if needed
        if metric_key in ["annual_return", "annual_vol", "max_drawdown"]:
            data_display = data  # Already in percentage
        else:
            data_display = data
        
        create_diverging_heatmap(ax, data_display, baseline_val, 
                                f"{scaling_label}: {metric_title}", 
                                higher_better, fmt)
        
        # Add column labels (scaling functions) on top row
        if metric_idx == 0:
            ax.text(0.5, 1.15, scaling_label, transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add row labels (metrics) on leftmost column
        if scaling_idx == 0:
            ax.text(-0.15, 0.5, metric_title, transform=ax.transAxes,
                   ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

fig1.suptitle(f"VIX Strategy vs S&P 500 Baseline (SPY 2002-2025) with Transaction Costs\nAll Scaling Functions Comparison (Green = Better, Red = Worse) | TC: {TRANSACTION_COST*100:.2f}%", 
              fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig1.savefig("../../results/week3/parameter_tuning_vs_sp500_tc.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: results/week3/parameter_tuning_vs_sp500_tc.png")

# vs Bonds - 4x4 grid (4 metrics × 4 scaling functions)
fig2, axes2 = plt.subplots(4, 4, figsize=(20, 18))

for metric_idx, (metric_key, metric_title, baseline_val, higher_better, fmt) in enumerate(metrics_config):
    # Use bond baseline for this comparison
    if metric_key == "annual_return":
        baseline_val = bond_baseline["annual_return"]*100
    elif metric_key == "annual_vol":
        baseline_val = bond_baseline["annual_vol"]*100
    elif metric_key == "sharpe":
        baseline_val = bond_baseline["sharpe"]
    elif metric_key == "max_drawdown":
        baseline_val = bond_baseline["max_drawdown"]*100
    
    for scaling_idx, scaling in enumerate(SCALING_OPTIONS):
        scaling_label = SCALING_LABELS[scaling_idx]
        ax = axes2[metric_idx, scaling_idx]
        data = all_results[scaling][metric_key]
        
        # Convert to percentage if needed
        if metric_key in ["annual_return", "annual_vol", "max_drawdown"]:
            data_display = data  # Already in percentage
        else:
            data_display = data
        
        create_diverging_heatmap(ax, data_display, baseline_val, 
                                f"{scaling_label}: {metric_title}", 
                                higher_better, fmt)
        
        # Add column labels (scaling functions) on top row
        if metric_idx == 0:
            ax.text(0.5, 1.15, scaling_label, transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add row labels (metrics) on leftmost column
        if scaling_idx == 0:
            ax.text(-0.15, 0.5, metric_title, transform=ax.transAxes,
                   ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

fig2.suptitle(f"VIX Strategy vs 10Y Treasury Baseline (SPY 2002-2025) with Transaction Costs\nAll Scaling Functions Comparison (Green = Better, Red = Worse) | TC: {TRANSACTION_COST*100:.2f}%", 
              fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig2.savefig("../../results/week3/parameter_tuning_vs_bonds_tc.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved: results/week3/parameter_tuning_vs_bonds_tc.png")

plt.show()

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Transaction Cost: {TRANSACTION_COST*100:.2f}% ({TRANSACTION_COST*10000:.0f} bps) per rebalancing trade")
print(f"Optimal Parameters: {best_params[2]} lookback, {best_params[3]} cutoff, {best_scaling} scaling")
print(f"Optimal Sharpe: {best_sharpe:.3f} vs S&P 500: {stock_baseline['sharpe']:.3f}")
print(f"\nScaling Function Comparison:")
for scaling in SCALING_OPTIONS:
    scaling_label = SCALING_LABELS[SCALING_OPTIONS.index(scaling)]
    sharpe_data = all_results[scaling]["sharpe"]
    if np.any(~np.isnan(sharpe_data)):
        max_sharpe = np.nanmax(sharpe_data)
        print(f"  {scaling_label:8s}: Max Sharpe = {max_sharpe:.3f}")



