# Strategy Performance Analyzer

Production-ready Python module for comprehensive strategy performance analysis.

## Features

- **Annual Performance**: Year-over-Year breakdown with Total Return, Volatility, Max Drawdown, and Sharpe Ratio
- **Rolling Metrics**: 1-year, 2-year, and 3-year rolling Sharpe ratios
- **Consistency Metrics**: Win Rate, Profit Factor, and Value at Risk (VaR 95%)
- **Regime Attribution**: Bull/Bear market analysis with Beta calculation
- **Professional Visualizations**: 2x2 subplot layout with equity curve, rolling Sharpe, drawdown, and monthly returns heatmap

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import pandas as pd
from strategy_performance_analyzer import StrategyPerformanceAnalyzer

# Load your strategy returns
df = pd.DataFrame({
    'returns': your_returns_series
}, index=your_datetime_index)

# Initialize analyzer
analyzer = StrategyPerformanceAnalyzer(df, risk_free_rate=0.04)

# Run full analysis
results = analyzer.run_full_analysis(output_dir='results/')
```

### With Benchmark for Regime Analysis

```python
# Include benchmark returns for Bull/Bear regime analysis
benchmark_returns = pd.Series(your_benchmark_returns, index=your_datetime_index)

results = analyzer.run_full_analysis(
    benchmark_returns=benchmark_returns,
    output_dir='results/'
)
```

### Run Complete Analysis (SPY Strategy)

```bash
cd src
python run_analysis.py
```

## Input Format

The analyzer expects a DataFrame with:
- **DatetimeIndex**: Date index
- **'returns' column**: Daily returns in decimal format (e.g., 0.01 for 1%)

## Outputs

### CSV Files
- `annual_performance.csv`: Year-over-year metrics
- `rolling_metrics.csv`: Rolling Sharpe ratios
- `consistency_metrics.csv`: Win rate, profit factor, VaR
- `regime_attribution.csv`: Bull/Bear performance and Beta

### Visualizations
- `annual_performance.png`: 2x2 bar charts (Return, Volatility, Drawdown, Sharpe)
- `comprehensive_analysis.png`: 2x2 subplot (Equity Curve, Rolling Sharpe, Drawdown, Monthly Heatmap)

## Code Structure

- `strategy_performance_analyzer.py`: Main analyzer class with all metrics
- `run_analysis.py`: Example script for VIX strategy analysis

## Metrics Explained

### Win Rate
Percentage of days with positive returns. Higher is better.

### Profit Factor
Ratio of gross gains to gross losses. Values > 1 indicate profitable strategy.

### Value at Risk (VaR 95%)
The maximum expected loss with 95% confidence. Represents the 5th percentile of returns.

### Beta
Sensitivity to benchmark. Beta > 1 means more volatile than benchmark; Beta < 1 means less volatile.

## Author

Senior Quantitative Developer



