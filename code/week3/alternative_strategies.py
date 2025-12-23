"""
Alternative Market Timing Strategies for Comparison

This script implements various market timing strategies to compare
against the VIX-based strategy. Each strategy uses different signals to time
the market between stocks and bonds.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime
from utils import download_data, calculate_metrics, calculate_sharpe_ratio

# =============================================================================
# Configuration
# =============================================================================
TICKERS = {"Stock": "SPY", "Bond": "IEF", "VIX": "^VIX"}
START_DATE = "2002-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RISK_FREE_RATE = 0.02

# =============================================================================
# Strategy Implementations
# =============================================================================

def moving_average_crossover_strategy(df, short_window=50, long_window=200):
    """
    Moving Average Crossover Strategy
    
    Buy stocks when short MA > long MA, otherwise hold bonds.
    Classic trend-following strategy.
    """
    df = df.copy()
    df['MA_Short'] = df['Stock'].rolling(window=short_window).mean()
    df['MA_Long'] = df['Stock'].rolling(window=long_window).mean()
    df['Signal'] = (df['MA_Short'] > df['MA_Long']).astype(int).shift(1)
    df = df.dropna()
    
    df['Strategy_Return'] = np.where(df['Signal'] == 1, 
                                     df['Stock_Return'], 
                                     df['Bond_Return'])
    return df['Strategy_Return'], "MA Crossover"


def momentum_strategy(df, lookback=252, threshold=0.10):
    """
    Momentum Strategy
    
    Buy stocks when 1-year return > threshold, otherwise hold bonds.
    Based on momentum factor.
    """
    df = df.copy()
    df['Momentum'] = df['Stock'].pct_change(lookback)
    df['Signal'] = (df['Momentum'] > threshold).astype(int).shift(1)
    df = df.dropna()
    
    df['Strategy_Return'] = np.where(df['Signal'] == 1, 
                                     df['Stock_Return'], 
                                     df['Bond_Return'])
    return df['Strategy_Return'], "Momentum"


def volatility_regime_strategy(df, lookback=252, high_vol_threshold=0.20):
    """
    Volatility Regime Strategy (similar to VIX but using realized volatility)
    
    Use realized volatility instead of VIX. Hold bonds when volatility is high.
    """
    df = df.copy()
    df['Realized_Vol'] = df['Stock_Return'].rolling(window=lookback).std() * np.sqrt(252)
    df['Vol_Threshold'] = df['Realized_Vol'].rolling(window=lookback).quantile(0.75)
    df['Signal'] = (df['Realized_Vol'] > df['Vol_Threshold']).astype(int).shift(1)
    df = df.dropna()
    
    df['Strategy_Return'] = np.where(df['Signal'] == 1, 
                                     df['Bond_Return'], 
                                     df['Stock_Return'])
    return df['Strategy_Return'], "Volatility Regime"


def drawdown_protection_strategy(df, max_drawdown_threshold=-0.15):
    """
    Drawdown Protection Strategy
    
    Switch to bonds when portfolio drawdown exceeds threshold.
    Dynamic risk management approach.
    """
    df = df.copy()
    df['Cumulative'] = (1 + df['Stock_Return']).cumprod()
    df['Rolling_Max'] = df['Cumulative'].expanding().max()
    df['Drawdown'] = (df['Cumulative'] - df['Rolling_Max']) / df['Rolling_Max']
    
    df['Signal'] = (df['Drawdown'] < max_drawdown_threshold).astype(int).shift(1)
    df = df.dropna()
    
    df['Strategy_Return'] = np.where(df['Signal'] == 1, 
                                     df['Bond_Return'], 
                                     df['Stock_Return'])
    return df['Strategy_Return'], "Drawdown Protection"


def yield_curve_strategy(df, lookback=63):
    """
    Yield Curve Strategy (requires additional data)
    
    Uses 10Y - 2Y Treasury spread. Inverted yield curve (negative spread)
    signals recession risk - switch to bonds.
    
    Note: This is a placeholder - would need to download yield curve data
    """
    df = df.copy()
    # Placeholder: would need to download 10Y and 2Y Treasury yields
    # For now, return None to indicate data needed
    return None, "Yield Curve (needs data)"


def dual_momentum_strategy(df, stock_lookback=252, bond_lookback=252):
    """
    Dual Momentum Strategy (Antonacci's approach)
    
    Compare stock momentum vs bond momentum. Hold whichever has better
    recent performance.
    """
    df = df.copy()
    df['Stock_Momentum'] = df['Stock'].pct_change(stock_lookback)
    df['Bond_Momentum'] = df['Bond'].pct_change(bond_lookback)
    
    df['Signal'] = (df['Stock_Momentum'] > df['Bond_Momentum']).astype(int).shift(1)
    df = df.dropna()
    
    df['Strategy_Return'] = np.where(df['Signal'] == 1, 
                                     df['Stock_Return'], 
                                     df['Bond_Return'])
    return df['Strategy_Return'], "Dual Momentum"


def risk_parity_strategy(df, target_vol=0.10, rebalance_freq=21):
    """
    Risk Parity Strategy
    
    Dynamically allocate between stocks and bonds to maintain target volatility.
    More sophisticated than simple switching.
    """
    df = df.copy()
    
    # Calculate rolling volatility
    df['Stock_Vol'] = df['Stock_Return'].rolling(window=63).std() * np.sqrt(252)
    df['Bond_Vol'] = df['Bond_Return'].rolling(window=63).std() * np.sqrt(252)
    
    # Calculate allocation to maintain target volatility
    # Simplified: assume correlation = 0.3
    correlation = 0.3
    df['Stock_Weight'] = target_vol / (df['Stock_Vol'] * (1 + correlation))
    df['Stock_Weight'] = df['Stock_Weight'].clip(0, 1)
    df['Bond_Weight'] = 1 - df['Stock_Weight']
    
    # Rebalance only at specified frequency
    df['Rebalance_Day'] = (df.index.to_series().diff().dt.days % rebalance_freq == 0) | (df.index == df.index[0])
    df['Stock_Weight'] = df['Stock_Weight'].where(df['Rebalance_Day']).ffill()
    
    df = df.dropna()
    df['Strategy_Return'] = (df['Stock_Weight'] * df['Stock_Return'] + 
                             df['Bond_Weight'] * df['Bond_Return'])
    
    return df['Strategy_Return'], "Risk Parity"


def regime_switching_strategy(df, lookback=252):
    """
    Regime Switching Strategy
    
    Identify bull/bear/sideways markets using multiple indicators:
    - Price trend (moving averages)
    - Volatility regime
    - Momentum
    
    Hold stocks in bull markets, bonds in bear markets, balanced in sideways.
    """
    df = df.copy()
    
    # Multiple indicators
    df['MA_50'] = df['Stock'].rolling(50).mean()
    df['MA_200'] = df['Stock'].rolling(200).mean()
    df['Momentum'] = df['Stock'].pct_change(252)
    df['Volatility'] = df['Stock_Return'].rolling(63).std() * np.sqrt(252)
    df['Vol_Percentile'] = df['Volatility'].rolling(lookback).rank(pct=True)
    
    # Regime classification
    df['Trend_Up'] = (df['MA_50'] > df['MA_200']).astype(int)
    df['Momentum_Positive'] = (df['Momentum'] > 0).astype(int)
    df['Low_Vol'] = (df['Vol_Percentile'] < 0.5).astype(int)
    
    # Bull: trend up + positive momentum + low vol
    # Bear: opposite
    # Sideways: mixed signals
    df['Regime_Score'] = df['Trend_Up'] + df['Momentum_Positive'] + df['Low_Vol']
    df['Stock_Allocation'] = np.where(df['Regime_Score'] >= 2, 1.0,
                                      np.where(df['Regime_Score'] <= 1, 0.0, 0.5))
    df['Stock_Allocation'] = df['Stock_Allocation'].shift(1)
    
    df = df.dropna()
    df['Strategy_Return'] = (df['Stock_Allocation'] * df['Stock_Return'] + 
                             (1 - df['Stock_Allocation']) * df['Bond_Return'])
    
    return df['Strategy_Return'], "Regime Switching"


# =============================================================================
# Main Comparison
# =============================================================================

def compare_strategies():
    """Compare all strategies against benchmarks."""
    
    print("=" * 70)
    print("ALTERNATIVE TIMING STRATEGIES COMPARISON")
    print("=" * 70)
    
    # Download data
    print(f"\nDownloading data from {START_DATE} to {END_DATE}...")
    df = download_data(TICKERS, START_DATE, END_DATE)
    df["Stock_Return"] = df["Stock"].pct_change()
    df["Bond_Return"] = df["Bond"].pct_change()
    df = df.dropna()
    
    print(f"Data loaded: {len(df)} trading days")
    
    # Calculate benchmarks
    stock_metrics = calculate_metrics(df["Stock_Return"], RISK_FREE_RATE)
    bond_metrics = calculate_metrics(df["Bond_Return"], RISK_FREE_RATE)
    
    print(f"\nBenchmarks:")
    print(f"  S&P 500: Return={stock_metrics['annual_return']*100:.2f}%, "
          f"Sharpe={stock_metrics['sharpe']:.3f}, "
          f"Vol={stock_metrics['annual_vol']*100:.2f}%")
    print(f"  10Y Treasury: Return={bond_metrics['annual_return']*100:.2f}%, "
          f"Sharpe={bond_metrics['sharpe']:.3f}, "
          f"Vol={bond_metrics['annual_vol']*100:.2f}%")
    
    # Test strategies
    strategies = [
        moving_average_crossover_strategy,
        momentum_strategy,
        volatility_regime_strategy,
        drawdown_protection_strategy,
        dual_momentum_strategy,
        regime_switching_strategy,
    ]
    
    results = []
    
    print(f"\n{'='*70}")
    print("STRATEGY RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<25} {'Return':<10} {'Sharpe':<10} {'Vol':<10} {'Max DD':<10}")
    print("-" * 70)
    
    for strategy_func in strategies:
        try:
            returns, name = strategy_func(df)
            if returns is not None and len(returns) > 252:
                metrics = calculate_metrics(returns, RISK_FREE_RATE)
                results.append({
                    'Strategy': name,
                    'Return': metrics['annual_return'] * 100,
                    'Sharpe': metrics['sharpe'],
                    'Vol': metrics['annual_vol'] * 100,
                    'Max_DD': metrics['max_drawdown'] * 100
                })
                print(f"{name:<25} {metrics['annual_return']*100:>9.2f}% "
                      f"{metrics['sharpe']:>9.3f} {metrics['annual_vol']*100:>9.2f}% "
                      f"{metrics['max_drawdown']*100:>9.2f}%")
        except Exception as e:
            print(f"{strategy_func.__name__}: Error - {e}")
    
    # Add benchmarks
    results.append({
        'Strategy': 'S&P 500 (Buy & Hold)',
        'Return': stock_metrics['annual_return'] * 100,
        'Sharpe': stock_metrics['sharpe'],
        'Vol': stock_metrics['annual_vol'] * 100,
        'Max_DD': stock_metrics['max_drawdown'] * 100
    })
    results.append({
        'Strategy': '10Y Treasury (Buy & Hold)',
        'Return': bond_metrics['annual_return'] * 100,
        'Sharpe': bond_metrics['sharpe'],
        'Vol': bond_metrics['annual_vol'] * 100,
        'Max_DD': bond_metrics['max_drawdown'] * 100
    })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Sharpe', ascending=False)
    results_df.to_csv("../../data/week3/alternative_strategies_comparison.csv", index=False)
    print(f"\n✓ Saved: data/week3/alternative_strategies_comparison.csv")
    
    print(f"\n{'='*70}")
    print("RANKED BY SHARPE RATIO")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    compare_strategies()

