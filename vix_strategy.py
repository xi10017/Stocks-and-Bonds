"""
VIX-Based Stock/Bond Switching Strategy

Strategy:
- When VIX > 75th percentile (rolling 1-year window) → Hold Bonds (IEF)
- Otherwise → Hold Stocks (VOO)

Benchmarks:
- Buy and Hold S&P 500 (VOO)
- Buy and Hold 10-Year Treasury Bonds (IEF)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =============================================================================
# Configuration
# =============================================================================
STOCK_TICKER = "VOO"      # Vanguard S&P 500 ETF
BOND_TICKER = "IEF"       # iShares 7-10 Year Treasury Bond ETF
VIX_TICKER = "^VIX"       # CBOE Volatility Index

ROLLING_WINDOW = 252      # Trading days in a year
VIX_PERCENTILE = 0.75     # 75th percentile threshold (3rd quartile)

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# =============================================================================
# Data Download
# =============================================================================
print("=" * 60)
print("VIX-BASED STOCK/BOND SWITCHING STRATEGY")
print("=" * 60)
print(f"\nDownloading data from {START_DATE} to {END_DATE}...")

# Download data
stock = yf.download(STOCK_TICKER, start=START_DATE, end=END_DATE, progress=False)
bond = yf.download(BOND_TICKER, start=START_DATE, end=END_DATE, progress=False)
vix = yf.download(VIX_TICKER, start=START_DATE, end=END_DATE, progress=False)
tbill = yf.download("^IRX", start=START_DATE, end=END_DATE, progress=False)  # 13-week T-bill rate

# Handle multi-level columns if present
if isinstance(stock.columns, pd.MultiIndex):
    stock = stock.droplevel(1, axis=1)
if isinstance(bond.columns, pd.MultiIndex):
    bond = bond.droplevel(1, axis=1)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix.droplevel(1, axis=1)
if isinstance(tbill.columns, pd.MultiIndex):
    tbill = tbill.droplevel(1, axis=1)

# Use Close prices (yfinance now auto-adjusts by default)
stock_prices = stock["Close"].rename("Stock")
bond_prices = bond["Close"].rename("Bond")
vix_prices = vix["Close"].rename("VIX")
# T-bill rate is annualized percentage, convert to daily decimal
tbill_rate = (tbill["Close"] / 100 / 252).rename("Rf_daily")  # Convert annual % to daily rate

# Combine into single DataFrame
df = pd.concat([stock_prices, bond_prices, vix_prices, tbill_rate], axis=1).dropna()
print(f"Data loaded: {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")

# =============================================================================
# Calculate Returns
# =============================================================================
df["Stock_Return"] = df["Stock"].pct_change()
df["Bond_Return"] = df["Bond"].pct_change()

# =============================================================================
# Generate Trading Signal
# =============================================================================
# Calculate rolling 75th percentile of VIX over past year
df["VIX_75pct"] = df["VIX"].rolling(window=ROLLING_WINDOW).quantile(VIX_PERCENTILE)

# Signal: 1 = Hold Bonds (high volatility), 0 = Hold Stocks (low volatility)
# Use previous day's signal to avoid look-ahead bias
df["High_Vol_Signal"] = (df["VIX"] > df["VIX_75pct"]).astype(int).shift(1)

# Drop rows with NaN (need ROLLING_WINDOW days for percentile calculation)
df = df.dropna()

print(f"\nStrategy period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total trading days: {len(df)}")

# =============================================================================
# Calculate Strategy Returns
# =============================================================================
# Strategy: Hold bonds when VIX is high, otherwise hold stocks
df["Strategy_Return"] = np.where(
    df["High_Vol_Signal"] == 1,
    df["Bond_Return"],
    df["Stock_Return"]
)

# Cumulative returns (growth of $1)
df["Stock_Cumulative"] = (1 + df["Stock_Return"]).cumprod()
df["Bond_Cumulative"] = (1 + df["Bond_Return"]).cumprod()
df["Strategy_Cumulative"] = (1 + df["Strategy_Return"]).cumprod()

# =============================================================================
# Performance Metrics
# =============================================================================
def calculate_metrics(returns, name, risk_free_rate=0.02):
    """Calculate performance metrics for a return series."""
    # Annualized return
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    # Annualized volatility
    annual_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        "Name": name,
        "Total Return": f"{total_return:.2%}",
        "Annual Return": f"{annual_return:.2%}",
        "Annual Volatility": f"{annual_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_drawdown:.2%}"
    }

# Calculate metrics for all strategies
stock_metrics = calculate_metrics(df["Stock_Return"], "Buy & Hold S&P 500 (VOO)")
bond_metrics = calculate_metrics(df["Bond_Return"], "Buy & Hold 10Y Treasury (IEF)")
strategy_metrics = calculate_metrics(df["Strategy_Return"], "VIX Switching Strategy")

# =============================================================================
# Display Results
# =============================================================================
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

metrics_df = pd.DataFrame([stock_metrics, bond_metrics, strategy_metrics])
metrics_df = metrics_df.set_index("Name")
print("\n", metrics_df.to_string())

# Signal statistics
bond_days = df["High_Vol_Signal"].sum()
stock_days = len(df) - bond_days
print(f"\n" + "=" * 60)
print("SIGNAL STATISTICS")
print("=" * 60)
print(f"Days holding Bonds (VIX > 75th percentile): {bond_days} ({bond_days/len(df):.1%})")
print(f"Days holding Stocks (VIX <= 75th percentile): {stock_days} ({stock_days/len(df):.1%})")

# VIX statistics
print(f"\n" + "=" * 60)
print("VIX STATISTICS")
print("=" * 60)
print(f"VIX Mean: {df['VIX'].mean():.2f}")
print(f"VIX Median: {df['VIX'].median():.2f}")
print(f"VIX Min: {df['VIX'].min():.2f}")
print(f"VIX Max: {df['VIX'].max():.2f}")
print(f"Current VIX 75th Percentile Threshold: {df['VIX_75pct'].iloc[-1]:.2f}")

# =============================================================================
# Plotting
# =============================================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 18))

# Plot 1: Cumulative Returns
ax1 = axes[0]
ax1.plot(df.index, df["Stock_Cumulative"], label="S&P 500 (VOO)", color="#1f77b4", linewidth=1.5)
ax1.plot(df.index, df["Bond_Cumulative"], label="10Y Treasury (IEF)", color="#2ca02c", linewidth=1.5)
ax1.plot(df.index, df["Strategy_Cumulative"], label="VIX Strategy", color="#d62728", linewidth=2)
ax1.set_ylabel("Growth of $1", fontsize=11)
ax1.set_title("Cumulative Returns Comparison", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(df.index[0], df.index[-1])

# Plot 2: VIX and Threshold
ax2 = axes[1]
ax2.plot(df.index, df["VIX"], label="VIX", color="#7f7f7f", linewidth=1, alpha=0.8)
ax2.plot(df.index, df["VIX_75pct"], label="75th Percentile (1Y Rolling)", 
         color="#ff7f0e", linewidth=1.5, linestyle="--")
ax2.fill_between(df.index, 0, df["VIX"], 
                  where=(df["High_Vol_Signal"] == 1), 
                  color="#d62728", alpha=0.3, label="High Vol → Hold Bonds")
ax2.set_ylabel("VIX Level", fontsize=11)
ax2.set_title("VIX Index and 75th Percentile Threshold", fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(df.index[0], df.index[-1])

# Plot 3: Rolling Sharpe Ratios - UNADJUSTED (mean/std, no risk-free subtraction)
rolling_window_sharpe = 252

stock_rolling_sharpe_unadj = (df["Stock_Return"].rolling(rolling_window_sharpe).mean() * 252) / \
                              (df["Stock_Return"].rolling(rolling_window_sharpe).std() * np.sqrt(252))
bond_rolling_sharpe_unadj = (df["Bond_Return"].rolling(rolling_window_sharpe).mean() * 252) / \
                             (df["Bond_Return"].rolling(rolling_window_sharpe).std() * np.sqrt(252))
strategy_rolling_sharpe_unadj = (df["Strategy_Return"].rolling(rolling_window_sharpe).mean() * 252) / \
                                 (df["Strategy_Return"].rolling(rolling_window_sharpe).std() * np.sqrt(252))

ax3 = axes[2]
ax3.plot(df.index, stock_rolling_sharpe_unadj, label="S&P 500", color="#1f77b4", linewidth=1.2, alpha=0.8)
ax3.plot(df.index, bond_rolling_sharpe_unadj, label="10Y Treasury", color="#2ca02c", linewidth=1.2, alpha=0.8)
ax3.plot(df.index, strategy_rolling_sharpe_unadj, label="VIX Strategy", color="#d62728", linewidth=1.8)
ax3.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
ax3.set_ylabel("Rolling Sharpe Ratio", fontsize=11)
ax3.set_title("1-Year Rolling Sharpe Ratio - UNADJUSTED (Return/Volatility)", fontsize=13, fontweight="bold")
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(df.index[0], df.index[-1])

# Plot 4: Rolling Sharpe Ratios - TREASURY ADJUSTED (using actual 13-week T-bill rate)
# Calculate rolling average of risk-free rate
rolling_rf = df["Rf_daily"].rolling(rolling_window_sharpe).mean()

stock_rolling_sharpe_adj = ((df["Stock_Return"].rolling(rolling_window_sharpe).mean() - rolling_rf) * 252) / \
                            (df["Stock_Return"].rolling(rolling_window_sharpe).std() * np.sqrt(252))
bond_rolling_sharpe_adj = ((df["Bond_Return"].rolling(rolling_window_sharpe).mean() - rolling_rf) * 252) / \
                           (df["Bond_Return"].rolling(rolling_window_sharpe).std() * np.sqrt(252))
strategy_rolling_sharpe_adj = ((df["Strategy_Return"].rolling(rolling_window_sharpe).mean() - rolling_rf) * 252) / \
                               (df["Strategy_Return"].rolling(rolling_window_sharpe).std() * np.sqrt(252))

ax4 = axes[3]
ax4.plot(df.index, stock_rolling_sharpe_adj, label="S&P 500", color="#1f77b4", linewidth=1.2, alpha=0.8)
ax4.plot(df.index, bond_rolling_sharpe_adj, label="10Y Treasury", color="#2ca02c", linewidth=1.2, alpha=0.8)
ax4.plot(df.index, strategy_rolling_sharpe_adj, label="VIX Strategy", color="#d62728", linewidth=1.8)
ax4.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
ax4.set_ylabel("Rolling Sharpe Ratio", fontsize=11)
ax4.set_title("1-Year Rolling Sharpe Ratio - TREASURY ADJUSTED (Rf = 13-week T-bill)", fontsize=13, fontweight="bold")
ax4.legend(loc="upper right", fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(df.index[0], df.index[-1])

# Plot 5: Rolling 12-Month Returns (first derivative of cumulative, smoothed)
rolling_return_window = 252
stock_rolling_return = (1 + df["Stock_Return"]).rolling(rolling_return_window).apply(
    lambda x: x.prod() - 1, raw=True) * 100  # Convert to percentage
bond_rolling_return = (1 + df["Bond_Return"]).rolling(rolling_return_window).apply(
    lambda x: x.prod() - 1, raw=True) * 100
strategy_rolling_return = (1 + df["Strategy_Return"]).rolling(rolling_return_window).apply(
    lambda x: x.prod() - 1, raw=True) * 100

ax5 = axes[4]
ax5.plot(df.index, stock_rolling_return, label="S&P 500", color="#1f77b4", linewidth=1.2, alpha=0.8)
ax5.plot(df.index, bond_rolling_return, label="10Y Treasury", color="#2ca02c", linewidth=1.2, alpha=0.8)
ax5.plot(df.index, strategy_rolling_return, label="VIX Strategy", color="#d62728", linewidth=1.8)
ax5.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
ax5.set_ylabel("Rolling 12-Month Return (%)", fontsize=11)
ax5.set_title("Trailing 12-Month Returns (Interval Performance)", fontsize=13, fontweight="bold")
ax5.legend(loc="upper right", fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlabel("Date", fontsize=11)
ax5.set_xlim(df.index[0], df.index[-1])

plt.tight_layout()
plt.savefig("strategy_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n✓ Chart saved to 'strategy_analysis.png'")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("STRATEGY SUMMARY")
print("=" * 60)
print("""
This strategy switches between stocks and bonds based on market volatility:
  • When VIX > 75th percentile (rolling 1-year) → Hold Bonds (defensive)
  • When VIX ≤ 75th percentile → Hold Stocks (risk-on)

The hypothesis is that this should provide:
  ✓ Similar or slightly lower returns than buy-and-hold stocks
  ✓ Significantly lower volatility
  ✓ Better risk-adjusted returns (higher Sharpe ratio)
  ✓ Lower maximum drawdown (downside protection)

No transaction costs assumed.
""")

