# Week 4 Results

This directory contains comprehensive analysis results for the VIX-based dynamic asset allocation strategy with optimal parameters.

## Directory Structure

### `indepth/`
**Comprehensive Strategy Performance Analysis**

Contains detailed performance visualizations for the baseline strategy (315 days lookback, 65% cutoff, exponential scaling) using only historical VIX data (no predictions).

**Files:**
- `annual_performance.png` - Annual return, volatility, Sharpe ratio, and max drawdown by year
- `annual_comparison.png` - Strategy returns vs S&P 500 vs Bonds by year
- `comprehensive_analysis.png` - Multi-panel overview of key performance metrics
- `metrics_heatmap.png` - Heatmap showing Return, Sharpe, and Volatility across all years (2003-2025)
- `rolling_metrics_enhanced.png` - Rolling 1-year, 2-year, and 3-year metrics for Sharpe, volatility, and max drawdown
- `regime_comparison.png` - Performance breakdown by market regime (Bull/Bear/Sideways)

**Key Findings:**
- Optimal Sharpe Ratio: 0.597 (vs S&P 500: 0.442)
- Annual Return: 10.48%
- Volatility: 11.17%
- Max Drawdown: -24.81%

---

### `parametertuning/vsspy/`
**Parameter Tuning Heatmaps vs S&P 500 Baseline**

Comprehensive parameter optimization results comparing strategy performance against S&P 500 across different parameter combinations.

**Files:**
- `parameter_tuning.png` - Main heatmap showing optimal parameters (4 metrics: return, volatility, Sharpe, drawdown)
- `parameter_tuning_vs_sp500.png` - Full 16-heatmap comparison (4 metrics × 4 scaling functions) vs S&P 500
- `parameter_tuning_vs_sp500_return.png` - Annual return comparison (4 scaling functions side-by-side)
- `parameter_tuning_vs_sp500_volatility.png` - Volatility comparison (4 scaling functions side-by-side)
- `parameter_tuning_vs_sp500_sharpe.png` - Sharpe ratio comparison (4 scaling functions side-by-side)
- `parameter_tuning_vs_sp500_drawdown.png` - Max drawdown comparison (4 scaling functions side-by-side)

**Optimal Parameters Found:**
- Lookback Window: 315 days (15 months)
- Percentile Cutoff: 65%
- Scaling Function: Exponential
- Optimal Sharpe: 0.597

**Parameter Space Tested:**
- Lookback Windows: 3mo, 6mo, 9mo, 1yr, 15mo, 18mo
- Percentile Cutoffs: 5% to 95% (in 5% increments)
- Scaling Functions: Linear, Sigmoid, Exponential, Power

---

### `parametertuning/vsbonds/`
**Parameter Tuning Heatmaps vs 10Y Treasury Baseline**

Comprehensive parameter optimization results comparing strategy performance against 10-Year Treasury bonds across different parameter combinations.

**Files:**
- `parameter_tuning_vs_bonds.png` - Full 16-heatmap comparison (4 metrics × 4 scaling functions) vs Bonds
- `parameter_tuning_vs_bonds_return.png` - Annual return comparison (4 scaling functions side-by-side)
- `parameter_tuning_vs_bonds_volatility.png` - Volatility comparison (4 scaling functions side-by-side)
- `parameter_tuning_vs_bonds_sharpe.png` - Sharpe ratio comparison (4 scaling functions side-by-side)
- `parameter_tuning_vs_bonds_drawdown.png` - Max drawdown comparison (4 scaling functions side-by-side)

**Purpose:**
These heatmaps help understand how the strategy performs relative to a bond-only portfolio, showing the risk-adjusted benefits of the dynamic allocation approach.

---

## Analysis Period

- **Data Range**: 2002-01-01 to 2026-01-05 (~23.4 years)
- **Trading Days**: ~5,897 days
- **Strategy Type**: Baseline (no VIX predictions, using yesterday's actual VIX)

## Methodology

1. **Parameter Tuning**: Tested 456 parameter combinations (6 lookback windows × 19 percentile cutoffs × 4 scaling functions)
2. **Optimization Criterion**: Maximize Sharpe ratio
3. **Validation**: Walk-forward approach with realistic data shifting (yesterday's VIX for today's allocation)

## Notes

- All results use the **baseline strategy** (no SAMoVAR predictions)
- Results are based on **realistic trading constraints** (no look-ahead bias)
- Heatmaps use diverging color scales (Green = Better, Red = Worse) relative to baseline benchmarks
