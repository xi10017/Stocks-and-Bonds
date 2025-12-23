# VIX-Based Dynamic Asset Allocation Strategy

A comprehensive quantitative analysis of a volatility-timing strategy that dynamically allocates between stocks and bonds based on the VIX index using progressive allocation mechanisms.

## Project Structure

```
StocksBonds/
├── code/
│   ├── utils.py                    # Shared utility functions
│   ├── week1/
│   │   └── vix_strategy.py         # Original binary switching strategy
│   ├── week2/
│   │   ├── parameter_tuning.py      # Parameter sweep (VOO 2010-2025)
│   │   ├── parameter_tuning_baseline.py  # Baseline comparison heatmaps
│   │   └── bootstrap_analysis.py   # Bootstrap confidence intervals
│   └── week3/
│       ├── parameter_tuning_spy.py  # Parameter sweep (SPY 2002-2025)
│       ├── parameter_tuning_spy_tc.py  # Parameter tuning with transaction costs
│       ├── bootstrap_analysis_spy.py  # Bootstrap with extended data
│       └── alternative_strategies.py  # Alternative timing strategies
├── new/
│   ├── src/
│   │   ├── run_analysis.py         # Main analysis script
│   │   └── strategy_performance_analyzer.py  # Comprehensive performance analyzer
│   ├── results/                    # Comprehensive analysis results
│   └── README.md                   # Analyzer documentation
├── data/                           # CSV data files
├── results/                        # Visualization outputs
└── docs/                           # Documentation and communication materials
```

## Strategy Description

**Progressive Allocation:** Instead of binary switching, the strategy uses a "dial" mechanism where bond allocation increases proportionally based on how much VIX exceeds its historical percentile threshold.

**Scaling Functions:** Tested linear, exponential, sigmoid, and power functions to map VIX excess to bond allocation.

**Instruments:**
- Stocks: SPY (SPDR S&P 500 ETF) - switched from VOO for longer historical data
- Bonds: IEF (iShares 7-10 Year Treasury Bond ETF)
- Volatility: ^VIX (CBOE Volatility Index)

**Optimal Parameters:**
- Lookback Window: 18 months (378 trading days)
- Percentile Cutoff: 60th percentile
- Scaling Function: Sigmoid

## Key Features

1. **Progressive Allocation**: Smooth transition between stocks and bonds (not binary)
2. **Comprehensive Analysis**: Year-to-year breakdown, rolling metrics, regime analysis
3. **Standard Sharpe Ratio Calculation**: Uses industry-standard arithmetic mean approach
4. **Block Bootstrap**: Preserves time-series dependence for confidence intervals
5. **Parameter Tuning**: Tests multiple combinations of lookback windows, percentile cutoffs, and scaling functions
6. **Transaction Costs**: Separate analysis with transaction costs included
7. **Statistical Validation**: T-tests, F-tests, correlation analysis

## Usage

### Week 1: Original Strategy
```bash
cd code/week1
python vix_strategy.py
```

### Week 2: Parameter Tuning (VOO 2010-2025)
```bash
cd code/week2
python parameter_tuning.py              # Basic heatmaps
python parameter_tuning_baseline.py     # Baseline comparisons
python bootstrap_analysis.py           # Confidence intervals
```

### Week 3: Extended Data (SPY 2002-2025)
```bash
cd code/week3
python parameter_tuning_spy.py          # Parameter sweep
python parameter_tuning_spy_tc.py       # Parameter tuning with transaction costs
python bootstrap_analysis_spy.py       # Bootstrap analysis
python alternative_strategies.py       # Compare alternative timing strategies
```

### Comprehensive Performance Analysis
```bash
cd new/src
python run_analysis.py                  # Run full comprehensive analysis
```

## Dependencies

See `requirements.txt`:
- yfinance
- pandas
- numpy
- matplotlib

## Key Results

### Week 3 (SPY 2002-2025, 22 years) - Progressive Allocation
- **Optimal Parameters**: 18mo lookback, 60% cutoff, Sigmoid scaling
- **Annualized Return**: 28.25%
- **Sharpe Ratio**: 1.97 vs S&P 500: 0.42
- **Volatility**: 10.95% (42% lower than S&P 500)
- **Max Drawdown**: -12.10% vs S&P 500: -34.00%
- **Win Rate**: 90.9% of years beat S&P 500
- **Bootstrap Validation**: Statistically significant difference from pure stocks

## Notes

- All Sharpe ratios use the **standard arithmetic mean formula** (industry standard)
- Block bootstrap uses 21-day blocks to preserve monthly patterns
- Transaction costs analysis available in `parameter_tuning_spy_tc.py` (0.1% per trade)
- Signal uses previous day's data to avoid look-ahead bias
- Progressive allocation tested with 4 scaling functions: linear, exponential, sigmoid, power

## Documentation

See `docs/` folder for:
- Project summary and assessment
- Email templates and communication materials
- Strategy evaluation and alternative approaches

