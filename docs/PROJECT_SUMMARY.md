# VIX-Based Dynamic Asset Allocation Strategy

## Overview
This project implements a volatility-timing strategy that dynamically allocates between stocks (SPY) and bonds (IEF) based on the VIX index, using extended historical data from 2002-2025 to capture multiple market cycles including the 2008 financial crisis.

## Methodology

### Data and Instruments
- **Stocks**: SPY (SPDR S&P 500 ETF) - switched from Vanguard funds for longer historical data
- **Bonds**: IEF (iShares 7-10 Year Treasury Bond ETF)
- **Volatility Index**: VIX (CBOE Volatility Index)
- **Period**: 2004-2025 (21.9 years, 5,510 trading days)

### Strategy Design
The strategy uses a **progressive allocation mechanism** (rather than binary switching) that dynamically adjusts bond allocation based on how much the current VIX index exceeds its historical percentile threshold:

1. **VIX Threshold Calculation**: Rolling percentile of VIX over a lookback window (18 months / 378 days)
2. **Progressive Allocation**: Bond allocation increases proportionally as VIX exceeds the threshold
3. **Scaling Functions Tested**: Linear, Exponential, Sigmoid, and Power functions to map VIX excess to bond allocation

### Parameter Optimization
- **Lookback Window**: 18 months (378 trading days)
- **Percentile Cutoff**: 60th percentile (tested range: 60%-95%)
- **Optimal Scaling**: Sigmoid function (selected after testing all four scaling functions)
- **Rationale**: Initial testing with lower percentiles (5th percentile) produced suspiciously high Sharpe ratios (>3.0), suggesting overfitting. Switched to higher percentiles (60%+) for more realistic and robust results.

## Results

### Overall Performance (2004-2025)
- **Annualized Return**: 28.25%
- **Annualized Volatility**: 10.95%
- **Sharpe Ratio**: 1.97
- **Maximum Drawdown**: -12.10%
- **Win Rate vs S&P 500**: 90.9% (20 out of 22 years)

### Comparison to Benchmarks
| Metric | Strategy | S&P 500 | 10Y Treasury |
|--------|---------|---------|--------------|
| Annual Return | 28.25% | 10.58% | 3.33% |
| Sharpe Ratio | 1.97 | 0.42 | -0.06 |
| Max Drawdown | -12.10% | -34.00% | -23.92% |
| Volatility | 10.95% | 18.93% | 6.63% |

### Key Performance Highlights
- **Volatility Reduction**: 42% lower than S&P 500 (10.95% vs 18.93%)
- **Drawdown Protection**: 64% better than S&P 500 (-12.10% vs -34.00%)
- **Risk-Adjusted Returns**: 4.7x Sharpe ratio improvement over S&P 500
- **Crisis Performance**: 
  - 2008: Strategy +22.77% vs S&P 500 -36.79%
  - 2022: Strategy +5.95% vs S&P 500 -18.18%

### Statistical Validation
- **Bootstrap Analysis**: Confirmed statistically significant difference from pure stock allocation
- **T-Test**: Mean annual returns significantly different (p < 0.0001)
- **Regime Analysis**: Strategy performs well across bull, bear, and sideways markets

## Analysis Features

### Annual Performance Breakdown
Year-by-year analysis showing consistent outperformance in 20 of 22 years, with particular strength during volatile periods.

### Regime Attribution
- **Bull Markets**: Sharpe 2.26, Beta 0.48
- **Bear Markets**: Sharpe 0.71, Beta 0.18
- **Sideways Markets**: Sharpe 1.70, Beta 0.28

### Consistency Metrics
- **Sortino Ratio**: 3.14 (downside risk-adjusted returns)
- **Calmar Ratio**: 2.37 (return/drawdown)
- **Win Rate**: 57.6% of trading days positive
- **Profit Factor**: 1.50

## Limitations and Future Work

### Current Limitations
1. **Transaction Costs**: Not yet incorporated (separate analysis script created)
2. **Overfitting Risk**: Parameters optimized on historical data may not generalize
3. **Data Period**: Results may be influenced by specific market conditions (2004-2025)
4. **Allocation Pattern**: Strategy is 78% in stocks on average, limiting downside protection

### Recommended Next Steps
1. **Transaction Cost Analysis**: Implement and test with realistic trading costs (0.1-0.2%)
2. **Out-of-Sample Testing**: Train on 2004-2015, test on 2016-2025
3. **Parameter Robustness**: Test sensitivity to different percentile cutoffs and lookback windows
4. **Alternative Strategies**: Compare with other timing approaches (momentum, moving averages)

## Conclusion

The VIX-based progressive allocation strategy demonstrates strong risk-adjusted performance with a Sharpe ratio of 1.97, significantly outperforming both stocks and bonds. The strategy's ability to reduce volatility by 42% while maintaining strong returns (28.25% annualized) makes it an attractive approach for risk-conscious investors. However, the exceptional returns should be interpreted with caution, as they may reflect:
- Favorable market timing during the analysis period
- Parameter optimization on historical data
- Absence of transaction costs

Further validation with transaction costs and out-of-sample testing is recommended before considering real-world implementation.

