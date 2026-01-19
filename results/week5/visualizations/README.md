# Week 5 Visualizations

This directory contains visualizations organized into two main categories:

## 📊 Prediction Models (`prediction_models/`)

Visualizations comparing VIX prediction models (AR3, SAMoVAR, and Naive baseline):

### AR3 vs SAMoVAR Comparisons:
- `ar3_vs_samovar_timeseries.png` - Time series comparison of predictions vs actual VIX
- `ar3_vs_samovar_scatter_plots.png` - Scatter plots showing prediction accuracy
- `ar3_vs_samovar_error_distributions.png` - Distribution of prediction errors
- `ar3_vs_samovar_error_over_time.png` - Error trends over the analysis period
- `ar3_vs_samovar_error_by_vix_level.png` - Performance breakdown by VIX level (low/medium/high)
- `ar3_vs_samovar_metrics_comparison.png` - Side-by-side comparison of key metrics (MAE, RMSE, etc.)
- `ar3_vs_samovar_head_to_head.png` - Head-to-head win/loss comparison

### Three-Way Comparisons (Naive, AR3, SAMoVAR):
- `naive_ar3_samovar_timeseries.png` - Time series with all three models
- `naive_ar3_samovar_scatter.png` - Scatter plots for all three models
- `naive_ar3_samovar_error_distributions.png` - Error distributions for all models
- `naive_ar3_samovar_metrics_comparison.png` - Comprehensive metrics comparison
- `naive_ar3_samovar_win_proportions.png` - Win rate analysis across all model pairs

**Analysis Period:** 2005-2019 (3,597 trading days)

## 📈 Strategy Metrics (`strategy_metrics/`)

Visualizations for the VIX-based trading strategy performance analysis:

- `metrics_barcharts_naive.png` - Annual performance bar charts for the "yesterday's VIX" strategy
  - Shows Return (%), Sharpe Ratio, and Volatility (%) by year
  - Uses bar charts instead of heatmaps for better readability of single-dimensional metrics

**Note:** This uses the baseline strategy (yesterday's VIX for today's allocation) without model predictions.
