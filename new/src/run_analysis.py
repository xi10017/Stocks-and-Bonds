"""
Main script to run strategy performance analysis for Week 3 (SPY 2002-2025).

This script:
1. Loads the VIX strategy returns (SPY period with optimal parameters)
2. Loads S&P 500 benchmark returns
3. Runs comprehensive performance analysis
4. Saves all outputs to results/ directory
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../code'))

import pandas as pd
import numpy as np
from datetime import datetime
from utils import download_data, calculate_progressive_bond_allocation, calculate_progressive_strategy_returns
from strategy_performance_analyzer import StrategyPerformanceAnalyzer

# =============================================================================
# Configuration
# =============================================================================
TICKERS = {"Stock": "SPY", "Bond": "IEF", "VIX": "^VIX"}
START_DATE = "2002-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Optimal parameters from Week 3 tuning
LOOKBACK_WINDOW = 21 * 15  # 15 months
PERCENTILE_CUTOFF = 0.6  # 70%

# Progressive allocation scaling function
# Options: 'linear', 'sigmoid', 'exponential', 'power'
# Set to None to use binary switching (old method)
# Set to 'all' to run analysis for all scaling functions and compare
SCALING = 'sigmoid'  # Use progressive allocation with exponential scaling
# SCALING = None  # Uncomment to use binary switching instead
# SCALING = 'all'  # Uncomment to analyze all scaling functions

RISK_FREE_RATE = 0.04  # 4% as specified
OUTPUT_DIR = "../results"

# Scaling function options
SCALING_OPTIONS = ['linear', 'sigmoid', 'exponential', 'power']
SCALING_LABELS = ['Linear', 'Sigmoid', 'Exponential', 'Power']

# =============================================================================
# Load Data and Generate Strategy Returns
# =============================================================================
print("=" * 70)
print("LOADING DATA AND GENERATING STRATEGY RETURNS")
print("=" * 70)

df_base = download_data(TICKERS, START_DATE, END_DATE)
df_base["Stock_Return"] = df_base["Stock"].pct_change()
df_base["Bond_Return"] = df_base["Bond"].pct_change()

# Generate VIX strategy signals
df = df_base.copy()
df["VIX_threshold"] = df["VIX"].rolling(window=LOOKBACK_WINDOW).quantile(PERCENTILE_CUTOFF)

if SCALING == 'all':
    # For 'all', we'll generate a placeholder - actual analysis happens later
    # Just set up the base dataframe
    df = df_base.copy()
    allocation_method = "All Scaling Functions (will be analyzed separately)"
    avg_bond_allocation = None
elif SCALING is None:
    # Binary switching (old method)
    df["Signal"] = (df["VIX"] > df["VIX_threshold"]).astype(int).shift(1)
    df = df.dropna()
    df["Strategy_Return"] = np.where(df["Signal"] == 1, df["Bond_Return"], df["Stock_Return"])
    allocation_method = "Binary Switching"
    avg_bond_allocation = df["Signal"].mean()
else:
    # Progressive allocation (new method)
    df["VIX_threshold_shifted"] = df["VIX_threshold"].shift(1)  # Avoid look-ahead bias
    df["Bond_Allocation"] = calculate_progressive_bond_allocation(
        df["VIX"],
        df["VIX_threshold_shifted"],
        scaling=SCALING
    )
    df = df.dropna()
    df["Strategy_Return"] = calculate_progressive_strategy_returns(
        df["Stock_Return"],
        df["Bond_Return"],
        df["Bond_Allocation"]
    )
    allocation_method = f"Progressive Allocation ({SCALING.capitalize()} scaling)"
    avg_bond_allocation = df["Bond_Allocation"].mean()
    
    # Diagnostic: Check bond allocation distribution
    print(f"\nBond Allocation Statistics:")
    print(f"  Mean: {df['Bond_Allocation'].mean():.2%}")
    print(f"  Median: {df['Bond_Allocation'].median():.2%}")
    print(f"  Max: {df['Bond_Allocation'].max():.2%}")
    print(f"  Min: {df['Bond_Allocation'].min():.2%}")
    print(f"  Days with >50% bonds: {(df['Bond_Allocation'] > 0.5).sum()} ({(df['Bond_Allocation'] > 0.5).mean():.1%})")
    print(f"  Days with >90% bonds: {(df['Bond_Allocation'] > 0.9).sum()} ({(df['Bond_Allocation'] > 0.9).mean():.1%})")
    
    # Verify returns calculation
    print(f"\nReturns Verification (sample):")
    sample_idx = df.index[100:105]
    print(f"  Date range: {sample_idx[0].date()} to {sample_idx[-1].date()}")
    for idx in sample_idx[:3]:
        stock_ret = df.loc[idx, 'Stock_Return']
        bond_ret = df.loc[idx, 'Bond_Return']
        bond_alloc = df.loc[idx, 'Bond_Allocation']
        stock_alloc = 1 - bond_alloc
        strategy_ret = df.loc[idx, 'Strategy_Return']
        expected_ret = stock_alloc * stock_ret + bond_alloc * bond_ret
        print(f"  {idx.date()}: Stock={stock_ret:.4f}, Bond={bond_ret:.4f}, BondAlloc={bond_alloc:.2%}, Strategy={strategy_ret:.4f}, Expected={expected_ret:.4f}, Match={abs(strategy_ret - expected_ret) < 1e-10}")

print(f"Data loaded: {len(df_base)} trading days")
print(f"Period: {df_base.index[0].date()} to {df_base.index[-1].date()}")
print(f"Strategy parameters: {LOOKBACK_WINDOW} days lookback, {PERCENTILE_CUTOFF*100:.0f}% cutoff")
print(f"Allocation method: {allocation_method}")
if avg_bond_allocation is not None:
    print(f"Average bond allocation: {avg_bond_allocation:.1%}")

# =============================================================================
# Prepare DataFrames for Analysis
# =============================================================================
if SCALING != 'all':
    # Strategy returns DataFrame
    strategy_df = pd.DataFrame({
        'returns': df['Strategy_Return']
    }, index=df.index)
    
    # Benchmark returns (S&P 500) for regime analysis
    benchmark_returns = df['Stock_Return']
    
    # Bond returns for comparison
    bond_returns = df['Bond_Return']

# =============================================================================
# Run Analysis
# =============================================================================
if SCALING == 'all':
    # Run analysis for all scaling functions
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS FOR ALL SCALING FUNCTIONS")
    print("=" * 70)
    
    all_results = {}
    for scaling in SCALING_OPTIONS:
        print(f"\n{'='*70}")
        print(f"Analyzing: {scaling.capitalize()} Scaling")
        print(f"{'='*70}")
        
        # Generate strategy returns with this scaling
        df_scaling = df_base.copy()
        df_scaling["VIX_threshold"] = df_scaling["VIX"].rolling(window=LOOKBACK_WINDOW).quantile(PERCENTILE_CUTOFF)
        df_scaling["VIX_threshold_shifted"] = df_scaling["VIX_threshold"].shift(1)
        df_scaling["Bond_Allocation"] = calculate_progressive_bond_allocation(
            df_scaling["VIX"],
            df_scaling["VIX_threshold_shifted"],
            scaling=scaling
        )
        df_scaling = df_scaling.dropna()
        df_scaling["Strategy_Return"] = calculate_progressive_strategy_returns(
            df_scaling["Stock_Return"],
            df_scaling["Bond_Return"],
            df_scaling["Bond_Allocation"]
        )
        
        # Prepare for analysis
        strategy_df_scaling = pd.DataFrame({
            'returns': df_scaling['Strategy_Return']
        }, index=df_scaling.index)
        
        # Run analysis with scaling-specific output directory
        scaling_output_dir = f"{OUTPUT_DIR}/{scaling}_scaling"
        os.makedirs(scaling_output_dir, exist_ok=True)
        
        analyzer = StrategyPerformanceAnalyzer(strategy_df_scaling, risk_free_rate=RISK_FREE_RATE)
        results_scaling = analyzer.run_full_analysis(
            benchmark_returns=df_scaling['Stock_Return'],
            bond_returns=df_scaling['Bond_Return'],
            output_dir=scaling_output_dir
        )
        
        all_results[scaling] = {
            'results': results_scaling,
            'avg_bond_allocation': df_scaling['Bond_Allocation'].mean(),
            'strategy_returns': df_scaling['Strategy_Return']
        }
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("SCALING FUNCTIONS COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Scaling':<12} {'Avg Bond %':<12} {'Annual Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
    print("-" * 70)
    
    for scaling in SCALING_OPTIONS:
        res = all_results[scaling]['results']
        annual_perf = res['annual_performance']
        consistency = res['consistency_metrics']
        
        # Calculate overall metrics
        strategy_returns = all_results[scaling]['strategy_returns']
        total_return = (1 + strategy_returns).prod() - 1
        years = len(strategy_returns) / 252
        annual_return = (1 + total_return) ** (1/years) - 1
        
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        sharpe = consistency.get('Sortino_Ratio', np.nan)  # Use Sortino as proxy if available
        # Calculate actual Sharpe
        daily_rf = (1 + RISK_FREE_RATE) ** (1/252) - 1
        excess_returns = strategy_returns - daily_rf
        sharpe = (excess_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else np.nan
        
        print(f"{scaling.capitalize():<12} {all_results[scaling]['avg_bond_allocation']:>11.1%} "
              f"{annual_return:>14.2%} {sharpe:>14.3f} {max_dd:>14.2%}")
    
    print(f"\nDetailed results saved to: {OUTPUT_DIR}/[scaling]_scaling/")
    
else:
    # Run analysis for single scaling function (or binary)
    analyzer = StrategyPerformanceAnalyzer(strategy_df, risk_free_rate=RISK_FREE_RATE)
    results = analyzer.run_full_analysis(
        benchmark_returns=benchmark_returns,
        bond_returns=bond_returns,
        output_dir=OUTPUT_DIR
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  - annual_performance.csv")
print("  - annual_comparison.csv")
print("  - rolling_metrics.csv")
print("  - consistency_metrics.csv")
print("  - regime_attribution.csv")
print("  - regime_performance.csv")
print("  - statistical_tests.txt")
print("  - summary.txt")
print("  - annual_performance.png")
print("  - annual_comparison.png")
print("  - comprehensive_analysis.png")
print("  - rolling_metrics_enhanced.png")
print("  - metrics_heatmap.png")
print("  - regime_comparison.png")

