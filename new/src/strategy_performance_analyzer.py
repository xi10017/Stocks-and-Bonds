"""
Production-Ready Strategy Performance Analyzer

Analyzes a trading strategy's performance with comprehensive metrics including:
- Annual performance breakdown
- Rolling metrics (1yr, 2yr, 3yr)
- Consistency metrics (Win Rate, Profit Factor, VaR)
- Regime attribution (Bull/Bear Beta)
- Professional visualizations

Author: Senior Quantitative Developer
Date: 2025-12-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class StrategyPerformanceAnalyzer:
    """
    Production-ready strategy performance analyzer.
    
    Assumes input DataFrame has:
    - DatetimeIndex
    - Column 'returns' (decimal format, e.g., 0.01 for 1%)
    - Optional: 'benchmark_returns' for regime analysis
    """
    
    def __init__(self, df: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex and 'returns' column
        risk_free_rate : float
            Annual risk-free rate (default 0.04 = 4%)
        """
        self.df = df.copy()
        self.df.index = pd.to_datetime(self.df.index)
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Validate input
        if 'returns' not in self.df.columns:
            raise ValueError("DataFrame must have 'returns' column")
        
        # Calculate cumulative returns
        self.df['cumulative'] = (1 + self.df['returns']).cumprod()
        
    # =========================================================================
    # Annual Performance Metrics
    # =========================================================================
    
    def calculate_annual_performance(self) -> pd.DataFrame:
        """
        Calculate Year-over-Year performance metrics.
        
        Returns:
        --------
        pd.DataFrame
            Columns: Year, Total_Return, Volatility, Max_Drawdown, Sharpe_Ratio
        """
        annual_data = []
        
        for year in self.df.index.year.unique():
            year_data = self.df[self.df.index.year == year]['returns']
            
            if len(year_data) < 20:  # Skip partial years with too few observations
                continue
            
            # Total return (geometric)
            total_return = (1 + year_data).prod() - 1
            
            # Annualized volatility
            volatility = year_data.std() * np.sqrt(252)
            
            # Max drawdown
            cumulative = (1 + year_data).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (annualized)
            excess_returns = year_data - self.daily_rf
            sharpe = (excess_returns.mean() / year_data.std()) * np.sqrt(252) if year_data.std() > 0 else np.nan
            
            annual_data.append({
                'Year': year,
                'Total_Return': total_return,
                'Volatility': volatility,
                'Max_Drawdown': max_drawdown,
                'Sharpe_Ratio': sharpe
            })
        
        return pd.DataFrame(annual_data).set_index('Year')
    
    # =========================================================================
    # Rolling Metrics
    # =========================================================================
    
    def calculate_rolling_sharpe(self, window_days: int) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Parameters:
        -----------
        window_days : int
            Rolling window in trading days (e.g., 252 for 1 year)
        
        Returns:
        --------
        pd.Series
            Rolling Sharpe ratio
        """
        excess_returns = self.df['returns'] - self.daily_rf
        rolling_mean = excess_returns.rolling(window=window_days).mean()
        rolling_std = self.df['returns'].rolling(window=window_days).std()
        
        # Annualize
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        return rolling_sharpe
    
    def calculate_rolling_volatility(self, window_days: int) -> pd.Series:
        """
        Calculate rolling volatility (annualized).
        
        Parameters:
        -----------
        window_days : int
            Rolling window in trading days
        
        Returns:
        --------
        pd.Series
            Rolling annualized volatility
        """
        return self.df['returns'].rolling(window=window_days).std() * np.sqrt(252)
    
    def calculate_rolling_max_drawdown(self, window_days: int) -> pd.Series:
        """
        Calculate rolling maximum drawdown (vectorized for performance).
        
        Parameters:
        -----------
        window_days : int
            Rolling window in trading days
        
        Returns:
        --------
        pd.Series
            Rolling maximum drawdown
        """
        # Calculate cumulative returns
        cumulative = self.df['cumulative']
        
        # Rolling maximum
        rolling_max = cumulative.rolling(window=window_days).max()
        
        # Drawdown
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # Rolling minimum of drawdown (most negative = worst drawdown)
        rolling_max_dd = drawdown.rolling(window=window_days).min()
        
        return rolling_max_dd
    
    def calculate_rolling_metrics(self) -> pd.DataFrame:
        """
        Calculate 1-year, 2-year, and 3-year rolling Sharpe ratios, volatility, and max drawdown.
        
        Returns:
        --------
        pd.DataFrame
            Columns: rolling_sharpe_1yr, rolling_sharpe_2yr, rolling_sharpe_3yr,
                     rolling_vol_1yr, rolling_vol_2yr, rolling_vol_3yr,
                     rolling_maxdd_1yr, rolling_maxdd_2yr, rolling_maxdd_3yr
        """
        return pd.DataFrame({
            'rolling_sharpe_1yr': self.calculate_rolling_sharpe(252),
            'rolling_sharpe_2yr': self.calculate_rolling_sharpe(504),
            'rolling_sharpe_3yr': self.calculate_rolling_sharpe(756),
            'rolling_vol_1yr': self.calculate_rolling_volatility(252),
            'rolling_vol_2yr': self.calculate_rolling_volatility(504),
            'rolling_vol_3yr': self.calculate_rolling_volatility(756),
            'rolling_maxdd_1yr': self.calculate_rolling_max_drawdown(252),
            'rolling_maxdd_2yr': self.calculate_rolling_max_drawdown(504),
            'rolling_maxdd_3yr': self.calculate_rolling_max_drawdown(756)
        })
    
    # =========================================================================
    # Consistency Metrics
    # =========================================================================
    
    def calculate_win_rate(self) -> float:
        """
        Calculate win rate (percentage of positive return days).
        
        Returns:
        --------
        float
            Win rate as decimal (e.g., 0.55 for 55%)
        """
        return (self.df['returns'] > 0).mean()
    
    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross gains / gross losses).
        
        Formula: Sum of positive returns / |Sum of negative returns|
        
        Returns:
        --------
        float
            Profit factor (typically > 1 is good)
        """
        positive_returns = self.df[self.df['returns'] > 0]['returns'].sum()
        negative_returns = abs(self.df[self.df['returns'] < 0]['returns'].sum())
        
        if negative_returns == 0:
            return np.inf if positive_returns > 0 else np.nan
        
        return positive_returns / negative_returns
    
    def calculate_var_95(self) -> float:
        """
        Calculate Value at Risk (VaR) at 95% confidence level.
        
        VaR represents the maximum expected loss over a given time period
        with 95% confidence.
        
        Returns:
        --------
        float
            VaR 95% as decimal (negative value)
        """
        return np.percentile(self.df['returns'], 5)
    
    def calculate_coefficient_of_variation(self, annual_perf: pd.DataFrame = None) -> float:
        """
        Calculate coefficient of variation (CV) of annual returns.
        
        CV = std(annual_returns) / mean(annual_returns)
        Measures consistency: lower CV = more consistent returns.
        
        Parameters:
        -----------
        annual_perf : pd.DataFrame, optional
            Annual performance DataFrame. If None, calculates it.
        
        Returns:
        --------
        float
            Coefficient of variation
        """
        if annual_perf is None:
            annual_perf = self.calculate_annual_performance()
        
        annual_returns = annual_perf['Total_Return']
        mean_return = annual_returns.mean()
        
        if mean_return == 0:
            return np.inf if annual_returns.std() > 0 else np.nan
        
        return annual_returns.std() / abs(mean_return)
    
    def calculate_downside_deviation(self) -> float:
        """
        Calculate downside deviation (annualized).
        
        Downside deviation only considers negative returns.
        Used in Sortino ratio calculation.
        
        Formula: sqrt(mean(min(returns, 0)^2)) * sqrt(252)
        
        Returns:
        --------
        float
            Annualized downside deviation
        """
        negative_returns = self.df['returns'].copy()
        negative_returns[negative_returns > 0] = 0
        downside_var = (negative_returns ** 2).mean()
        return np.sqrt(downside_var) * np.sqrt(252)
    
    def calculate_calmar_ratio(self, annual_perf: pd.DataFrame = None) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Higher is better. Measures return per unit of maximum drawdown risk.
        
        Parameters:
        -----------
        annual_perf : pd.DataFrame, optional
            Annual performance DataFrame. If None, calculates overall.
        
        Returns:
        --------
        float
            Calmar ratio
        """
        if annual_perf is None:
            # Calculate overall
            total_return = (1 + self.df['returns']).prod() - 1
            years = len(self.df) / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            
            cumulative = self.df['cumulative']
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
        else:
            # Use annual data
            annual_return = annual_perf['Total_Return'].mean()
            max_drawdown = abs(annual_perf['Max_Drawdown'].min())
        
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else np.nan
        
        return annual_return / max_drawdown
    
    def calculate_sortino_ratio(self) -> float:
        """
        Calculate Sortino ratio (annualized).
        
        Sortino = (mean(excess_returns) / downside_deviation) * sqrt(252)
        
        Similar to Sharpe but only penalizes downside volatility.
        
        Returns:
        --------
        float
            Annualized Sortino ratio
        """
        excess_returns = self.df['returns'] - self.daily_rf
        mean_excess = excess_returns.mean()
        downside_dev = self.calculate_downside_deviation() / np.sqrt(252)  # Convert to daily
        
        if downside_dev == 0:
            return np.nan
        
        return (mean_excess / downside_dev) * np.sqrt(252)
    
    def calculate_consistency_metrics(self, annual_perf: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate all consistency metrics.
        
        Parameters:
        -----------
        annual_perf : pd.DataFrame, optional
            Annual performance DataFrame for CV calculation
        
        Returns:
        --------
        dict
            Dictionary with Win_Rate, Profit_Factor, VaR_95, CV, 
            Downside_Deviation, Calmar_Ratio, Sortino_Ratio
        """
        if annual_perf is None:
            annual_perf = self.calculate_annual_performance()
        
        return {
            'Win_Rate': self.calculate_win_rate(),
            'Profit_Factor': self.calculate_profit_factor(),
            'VaR_95': self.calculate_var_95(),
            'Coefficient_of_Variation': self.calculate_coefficient_of_variation(annual_perf),
            'Downside_Deviation': self.calculate_downside_deviation(),
            'Calmar_Ratio': self.calculate_calmar_ratio(annual_perf),
            'Sortino_Ratio': self.calculate_sortino_ratio()
        }
    
    # =========================================================================
    # Annual Performance Comparison
    # =========================================================================
    
    def calculate_annual_comparison(self, benchmark_returns: pd.Series,
                                   bond_returns: pd.Series = None) -> pd.DataFrame:
        """
        Calculate annual performance comparison between strategy, S&P 500, and bonds.
        
        Parameters:
        -----------
        benchmark_returns : pd.Series
            Benchmark returns (e.g., S&P 500)
        bond_returns : pd.Series, optional
            Bond returns for comparison
        
        Returns:
        --------
        pd.DataFrame
            Annual returns for each strategy indexed by year
        """
        strategy_annual = self.calculate_annual_performance()
        comparison_data = []
        
        for year in strategy_annual.index:
            # Strategy return
            strategy_return = strategy_annual.loc[year, 'Total_Return']
            
            # Benchmark return
            year_bench = benchmark_returns[benchmark_returns.index.year == year]
            benchmark_return = (1 + year_bench).prod() - 1 if len(year_bench) > 0 else np.nan
            
            # Bond return
            if bond_returns is not None:
                year_bond = bond_returns[bond_returns.index.year == year]
                bond_return = (1 + year_bond).prod() - 1 if len(year_bond) > 0 else np.nan
            else:
                bond_return = np.nan
            
            comparison_data.append({
                'Year': year,
                'Strategy': strategy_return,
                'S&P_500': benchmark_return,
                'Bonds': bond_return
            })
        
        return pd.DataFrame(comparison_data).set_index('Year')
    
    def calculate_win_rate_vs_benchmark(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate percentage of years strategy beats benchmark.
        
        Parameters:
        -----------
        benchmark_returns : pd.Series
            Benchmark returns (e.g., S&P 500)
        
        Returns:
        --------
        float
            Win rate as decimal (e.g., 0.65 for 65%)
        """
        comparison = self.calculate_annual_comparison(benchmark_returns)
        wins = (comparison['Strategy'] > comparison['S&P_500']).sum()
        total = len(comparison.dropna(subset=['Strategy', 'S&P_500']))
        
        return wins / total if total > 0 else np.nan
    
    def calculate_consistency_std(self, annual_perf: pd.DataFrame = None) -> float:
        """
        Calculate standard deviation of annual returns (consistency measure).
        
        Lower = more consistent returns across years.
        
        Parameters:
        -----------
        annual_perf : pd.DataFrame, optional
            Annual performance DataFrame
        
        Returns:
        --------
        float
            Standard deviation of annual returns
        """
        if annual_perf is None:
            annual_perf = self.calculate_annual_performance()
        
        return annual_perf['Total_Return'].std()
    
    # =========================================================================
    # Regime Attribution
    # =========================================================================
    
    def classify_regimes(self, benchmark_returns: pd.Series = None, 
                        threshold: float = 0.05) -> pd.Series:
        """
        Classify market regimes as Bull, Bear, or Sideways based on annual performance.
        
        Bull: Annual return > threshold (default 5%)
        Bear: Annual return < -threshold
        Sideways: -threshold <= Annual return <= threshold
        
        Parameters:
        -----------
        benchmark_returns : pd.Series, optional
            Benchmark returns (e.g., S&P 500). If None, uses strategy returns.
        threshold : float
            Threshold for regime classification (default 0.05 = 5%)
        
        Returns:
        --------
        pd.Series
            Regime classification ('Bull', 'Bear', or 'Sideways') indexed by year
        """
        if benchmark_returns is None:
            benchmark_returns = self.df['returns']
        
        regimes = []
        for year in benchmark_returns.index.year.unique():
            year_returns = benchmark_returns[benchmark_returns.index.year == year]
            annual_return = (1 + year_returns).prod() - 1
            
            if annual_return > threshold:
                regime = 'Bull'
            elif annual_return < -threshold:
                regime = 'Bear'
            else:
                regime = 'Sideways'
            
            regimes.append({
                'Year': year,
                'Regime': regime
            })
        
        return pd.DataFrame(regimes).set_index('Year')['Regime']
    
    def calculate_beta(self, strategy_returns: pd.Series, 
                       benchmark_returns: pd.Series) -> float:
        """
        Calculate Beta (sensitivity to benchmark).
        
        Beta = Cov(strategy, benchmark) / Var(benchmark)
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            Strategy returns
        benchmark_returns : pd.Series
            Benchmark returns
        
        Returns:
        --------
        float
            Beta coefficient
        """
        # Align indices
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 20:
            return np.nan
        
        strategy = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]
        
        covariance = strategy.cov(benchmark)
        benchmark_variance = benchmark.var()
        
        if benchmark_variance == 0:
            return np.nan
        
        return covariance / benchmark_variance
    
    def test_regime_performance(self, benchmark_returns: pd.Series = None) -> pd.DataFrame:
        """
        Test if strategy performs better in volatile/bear markets.
        
        Compares strategy performance metrics across Bull, Bear, and Sideways regimes.
        
        Parameters:
        -----------
        benchmark_returns : pd.Series, optional
            Benchmark returns for regime classification
        
        Returns:
        --------
        pd.DataFrame
            Performance comparison across regimes
        """
        if benchmark_returns is None:
            benchmark_returns = self.df['returns']
        
        regimes = self.classify_regimes(benchmark_returns)
        regime_stats = []
        
        for regime_type in ['Bull', 'Bear', 'Sideways']:
            regime_years = regimes[regimes == regime_type].index
            
            if len(regime_years) == 0:
                continue
            
            # Get all returns for this regime
            regime_returns = pd.Series(dtype=float)
            for year in regime_years:
                year_returns = self.df[self.df.index.year == year]['returns']
                regime_returns = pd.concat([regime_returns, year_returns])
            
            if len(regime_returns) == 0:
                continue
            
            # Calculate metrics
            total_return = (1 + regime_returns).prod() - 1
            volatility = regime_returns.std() * np.sqrt(252)
            sharpe = ((regime_returns.mean() - self.daily_rf) / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else np.nan
            
            # Max drawdown
            cumulative = (1 + regime_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            regime_stats.append({
                'Regime': regime_type,
                'Total_Return': total_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe,
                'Max_Drawdown': max_drawdown,
                'Years': len(regime_years),
                'Trading_Days': len(regime_returns)
            })
        
        return pd.DataFrame(regime_stats).set_index('Regime')
    
    # =========================================================================
    # Statistical Tests
    # =========================================================================
    
    def perform_statistical_tests(self, benchmark_returns: pd.Series) -> Dict[str, Dict]:
        """
        Perform statistical tests comparing strategy to benchmark.
        
        Tests:
        1. T-test: Is mean annual return significantly different?
        2. F-test: Is variance significantly lower?
        3. Correlation: Strategy returns vs market returns
        
        Parameters:
        -----------
        benchmark_returns : pd.Series
            Benchmark returns (e.g., S&P 500)
        
        Returns:
        --------
        dict
            Dictionary containing test results with p-values and statistics
        """
        from scipy import stats
        
        # Calculate annual returns for both
        strategy_annual = self.calculate_annual_performance()
        benchmark_annual = []
        
        for year in strategy_annual.index:
            year_bench = benchmark_returns[benchmark_returns.index.year == year]
            annual_return = (1 + year_bench).prod() - 1 if len(year_bench) > 0 else np.nan
            benchmark_annual.append(annual_return)
        
        strategy_returns_annual = strategy_annual['Total_Return'].values
        benchmark_returns_annual = np.array(benchmark_annual)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(strategy_returns_annual) | np.isnan(benchmark_returns_annual))
        strategy_clean = strategy_returns_annual[valid_mask]
        benchmark_clean = benchmark_returns_annual[valid_mask]
        
        results = {}
        
        # 1. T-test: Mean annual return
        if len(strategy_clean) > 1 and len(benchmark_clean) > 1:
            t_stat, t_pvalue = stats.ttest_rel(strategy_clean, benchmark_clean)
            results['T_Test'] = {
                'statistic': t_stat,
                'pvalue': t_pvalue,
                'significant': t_pvalue < 0.05,
                'interpretation': 'Mean annual returns are significantly different' if t_pvalue < 0.05 else 'No significant difference in mean annual returns'
            }
        else:
            results['T_Test'] = {'error': 'Insufficient data for t-test'}
        
        # 2. F-test: Variance comparison
        if len(strategy_clean) > 1 and len(benchmark_clean) > 1:
            strategy_var = strategy_clean.var()
            benchmark_var = benchmark_clean.var()
            
            if benchmark_var > 0:
                f_stat = strategy_var / benchmark_var
                # F-test (two-tailed)
                df1 = len(strategy_clean) - 1
                df2 = len(benchmark_clean) - 1
                f_pvalue = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
                
                results['F_Test'] = {
                    'statistic': f_stat,
                    'pvalue': f_pvalue,
                    'strategy_variance': strategy_var,
                    'benchmark_variance': benchmark_var,
                    'variance_ratio': f_stat,
                    'significant': f_pvalue < 0.05,
                    'interpretation': f'Strategy variance is {"significantly" if f_pvalue < 0.05 else "not significantly"} {"lower" if f_stat < 1 else "higher"} than benchmark'
                }
            else:
                results['F_Test'] = {'error': 'Benchmark variance is zero'}
        else:
            results['F_Test'] = {'error': 'Insufficient data for F-test'}
        
        # 3. Correlation
        if len(strategy_clean) > 1 and len(benchmark_clean) > 1:
            correlation, corr_pvalue = stats.pearsonr(strategy_clean, benchmark_clean)
            results['Correlation'] = {
                'correlation': correlation,
                'pvalue': corr_pvalue,
                'significant': corr_pvalue < 0.05,
                'interpretation': f'{"Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"} {"positive" if correlation > 0 else "negative"} correlation'
            }
        else:
            results['Correlation'] = {'error': 'Insufficient data for correlation'}
        
        return results
    
    def calculate_regime_attribution(self, benchmark_returns: pd.Series = None) -> pd.DataFrame:
        """
        Calculate strategy performance and Beta in Bull vs Bear regimes.
        
        Parameters:
        -----------
        benchmark_returns : pd.Series, optional
            Benchmark returns for regime classification
        
        Returns:
        --------
        pd.DataFrame
            Regime performance summary
        """
        if benchmark_returns is None:
            benchmark_returns = self.df['returns']
        
        regimes = self.classify_regimes(benchmark_returns)
        
        regime_stats = []
        for regime_type in ['Bull', 'Bear', 'Sideways']:
            regime_years = regimes[regimes == regime_type].index
            
            # Get all returns for this regime
            regime_returns = pd.Series(dtype=float)
            for year in regime_years:
                year_returns = self.df[self.df.index.year == year]['returns']
                regime_returns = pd.concat([regime_returns, year_returns])
            
            if len(regime_returns) == 0:
                continue
            
            # Calculate metrics
            total_return = (1 + regime_returns).prod() - 1
            volatility = regime_returns.std() * np.sqrt(252)
            sharpe = ((regime_returns.mean() - self.daily_rf) / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else np.nan
            
            # Calculate Beta
            benchmark_regime = pd.Series(dtype=float)
            for year in regime_years:
                year_bench = benchmark_returns[benchmark_returns.index.year == year]
                benchmark_regime = pd.concat([benchmark_regime, year_bench])
            
            beta = self.calculate_beta(regime_returns, benchmark_regime)
            
            regime_stats.append({
                'Regime': regime_type,
                'Total_Return': total_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe,
                'Beta': beta,
                'Years': len(regime_years),
                'Trading_Days': len(regime_returns)
            })
        
        return pd.DataFrame(regime_stats).set_index('Regime')
    
    # =========================================================================
    # Visualizations
    # =========================================================================
    
    def plot_annual_performance(self, annual_perf: pd.DataFrame, 
                                save_path: str = None) -> None:
        """
        Create Seaborn bar chart of annual performance with meaningful colors.
        
        Colors are value-based:
        - Total Return: Green (positive), Red (negative), intensity by magnitude
        - Volatility: Red (high = bad), Green (low = good)
        - Max Drawdown: Red (large = bad), intensity by magnitude
        - Sharpe Ratio: Green (positive/high), Red (negative/low)
        
        Parameters:
        -----------
        annual_perf : pd.DataFrame
            Annual performance DataFrame
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        annual_perf_reset = annual_perf.reset_index()
        
        # Helper function to normalize values for color intensity
        def normalize_for_color(values, reverse=False):
            """Normalize values to 0-1 range for color mapping."""
            if len(values) == 0:
                return values
            min_val, max_val = values.min(), values.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(values), index=values.index)
            normalized = (values - min_val) / (max_val - min_val)
            if reverse:
                normalized = 1 - normalized
            return normalized
        
        # 1. Total Return - Green for positive, Red for negative
        ax1 = axes[0, 0]
        bars1 = ax1.bar(annual_perf_reset['Year'], annual_perf_reset['Total_Return'],
                       color=['#2ecc71' if x >= 0 else '#e74c3c' for x in annual_perf_reset['Total_Return']],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_title('Annual Total Return', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Return (Decimal)', fontsize=10)
        ax1.set_xlabel('Year', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Volatility - Red for high (bad), Green for low (good)
        ax2 = axes[0, 1]
        vol_normalized = normalize_for_color(annual_perf_reset['Volatility'], reverse=True)
        colors2 = plt.cm.RdYlGn(vol_normalized)
        bars2 = ax2.bar(annual_perf_reset['Year'], annual_perf_reset['Volatility'],
                       color=colors2, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_title('Annual Volatility', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volatility (Decimal)', fontsize=10)
        ax2.set_xlabel('Year', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Max Drawdown - Red intensity based on magnitude (more negative = darker red)
        ax3 = axes[1, 0]
        # Drawdowns are negative, so we normalize the absolute value
        drawdown_abs = abs(annual_perf_reset['Max_Drawdown'])
        drawdown_normalized = normalize_for_color(drawdown_abs, reverse=False)
        colors3 = plt.cm.Reds(0.3 + 0.7 * drawdown_normalized)  # Range from light to dark red
        bars3 = ax3.bar(annual_perf_reset['Year'], annual_perf_reset['Max_Drawdown'],
                       color=colors3, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_title('Annual Max Drawdown', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (Decimal)', fontsize=10)
        ax3.set_xlabel('Year', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Sharpe Ratio - Green for positive/high, Red for negative/low
        ax4 = axes[1, 1]
        sharpe_values = annual_perf_reset['Sharpe_Ratio']
        # Create diverging colormap: red for negative, green for positive
        # Normalize to -2 to 2 range (typical Sharpe range)
        sharpe_clipped = sharpe_values.clip(-2, 2)
        sharpe_normalized = (sharpe_clipped + 2) / 4  # Map -2 to 2 -> 0 to 1
        colors4 = plt.cm.RdYlGn(sharpe_normalized)
        bars4 = ax4.bar(annual_perf_reset['Year'], sharpe_values,
                       color=colors4, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_title('Annual Sharpe Ratio', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio', fontsize=10)
        ax4.set_xlabel('Year', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_comprehensive_analysis(self, annual_perf: pd.DataFrame,
                                   rolling_metrics: pd.DataFrame,
                                   save_path: str = None) -> None:
        """
        Create 2x2 subplot layout with comprehensive visualizations.
        
        Parameters:
        -----------
        annual_perf : pd.DataFrame
            Annual performance DataFrame
        rolling_metrics : pd.DataFrame
            Rolling metrics DataFrame
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cumulative Equity Curve (Log Scale)
        ax1 = axes[0, 0]
        ax1.semilogy(self.df.index, self.df['cumulative'], 
                    linewidth=2, color='#2E86AB', label='Strategy')
        ax1.set_title('Cumulative Equity Curve (Log Scale)', 
                     fontsize=13, fontweight='bold')
        ax1.set_ylabel('Cumulative Value (Log Scale)', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe Ratio Comparison
        ax2 = axes[0, 1]
        ax2.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe_1yr'], 
                label='1-Year', linewidth=1.5, alpha=0.8)
        ax2.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe_2yr'], 
                label='2-Year', linewidth=1.5, alpha=0.8)
        ax2.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe_3yr'], 
                label='3-Year', linewidth=1.5, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Rolling Sharpe Ratio Comparison', 
                      fontsize=13, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown 'Underwater' Chart
        ax3 = axes[1, 0]
        cumulative = self.df['cumulative']
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        ax3.fill_between(self.df.index, drawdown, 0, 
                        color='#D32F2F', alpha=0.6)
        ax3.plot(self.df.index, drawdown, color='#B71C1C', linewidth=1)
        ax3.set_title('Drawdown (Underwater Chart)', 
                     fontsize=13, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax4 = axes[1, 1]
        monthly_returns = self.df['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_pivot = monthly_returns.groupby([
            monthly_returns.index.year, 
            monthly_returns.index.month
        ]).first().unstack(level=1)
        
        sns.heatmap(monthly_pivot * 100, annot=True, fmt='.1f', 
                   cmap='RdYlGn', center=0, ax=ax4, 
                   cbar_kws={'label': 'Return (%)'})
        ax4.set_title('Monthly Returns Heatmap', 
                     fontsize=13, fontweight='bold')
        ax4.set_xlabel('Month', fontsize=11)
        ax4.set_ylabel('Year', fontsize=11)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_annual_comparison(self, comparison_df: pd.DataFrame,
                              save_path: str = None) -> None:
        """
        Create bar chart comparing annual returns: Strategy vs S&P 500 vs Bonds.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Annual comparison DataFrame from calculate_annual_comparison()
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.25
        
        bars1 = ax.bar(x - width, comparison_df['Strategy'] * 100, width,
                      label='Strategy', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x, comparison_df['S&P_500'] * 100, width,
                      label='S&P 500', color='#A23B72', alpha=0.8)
        
        if 'Bonds' in comparison_df.columns and not comparison_df['Bonds'].isna().all():
            bars3 = ax.bar(x + width, comparison_df['Bonds'] * 100, width,
                          label='Bonds', color='#F18F01', alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Annual Return (%)', fontsize=11)
        ax.set_title('Annual Returns Comparison: Strategy vs S&P 500 vs Bonds',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_heatmap(self, annual_perf: pd.DataFrame,
                            save_path: str = None) -> None:
        """
        Create heatmap: Year × Metric (Return, Sharpe, Volatility).
        
        Parameters:
        -----------
        annual_perf : pd.DataFrame
            Annual performance DataFrame
        save_path : str, optional
            Path to save figure
        """
        # Prepare data for heatmap
        heatmap_data = annual_perf[['Total_Return', 'Sharpe_Ratio', 'Volatility']].copy()
        heatmap_data.columns = ['Return', 'Sharpe', 'Volatility']
        
        # Normalize for better visualization (optional - can show raw values)
        fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_data) * 0.3)))
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0, ax=ax, cbar_kws={'label': 'Value'},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title('Annual Performance Heatmap: Year × Metric',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Metric', fontsize=11)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_regime_comparison(self, regime_perf: pd.DataFrame,
                              save_path: str = None) -> None:
        """
        Create visualization comparing strategy performance across regimes.
        
        Parameters:
        -----------
        regime_perf : pd.DataFrame
            Regime performance DataFrame from test_regime_performance()
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        regimes = regime_perf.index
        x = np.arange(len(regimes))
        
        # 1. Total Return
        ax1 = axes[0, 0]
        colors1 = ['#2ecc71' if x >= 0 else '#e74c3c' for x in regime_perf['Total_Return']]
        ax1.bar(regimes, regime_perf['Total_Return'] * 100, color=colors1, alpha=0.7, edgecolor='black')
        ax1.set_title('Total Return by Regime', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Return (%)', fontsize=10)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Volatility
        ax2 = axes[0, 1]
        ax2.bar(regimes, regime_perf['Volatility'] * 100, color='#e67e22', alpha=0.7, edgecolor='black')
        ax2.set_title('Volatility by Regime', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volatility (%)', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Sharpe Ratio
        ax3 = axes[1, 0]
        colors3 = plt.cm.RdYlGn([(x + 2) / 4 for x in regime_perf['Sharpe_Ratio'].clip(-2, 2)])
        ax3.bar(regimes, regime_perf['Sharpe_Ratio'], color=colors3, alpha=0.7, edgecolor='black')
        ax3.set_title('Sharpe Ratio by Regime', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio', fontsize=10)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Max Drawdown
        ax4 = axes[1, 1]
        drawdown_abs = abs(regime_perf['Max_Drawdown'])
        colors4 = plt.cm.Reds(0.3 + 0.7 * (drawdown_abs / drawdown_abs.max()))
        ax4.bar(regimes, regime_perf['Max_Drawdown'] * 100, color=colors4, alpha=0.7, edgecolor='black')
        ax4.set_title('Max Drawdown by Regime', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_rolling_metrics_enhanced(self, rolling_metrics: pd.DataFrame,
                                     save_path: str = None) -> None:
        """
        Enhanced rolling metrics plot including volatility and max drawdown.
        
        Parameters:
        -----------
        rolling_metrics : pd.DataFrame
            Rolling metrics DataFrame
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # 1. Rolling Sharpe Ratios
        ax1 = axes[0]
        ax1.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe_1yr'],
                label='1-Year', linewidth=1.5, alpha=0.8)
        ax1.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe_2yr'],
                label='2-Year', linewidth=1.5, alpha=0.8)
        ax1.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe_3yr'],
                label='3-Year', linewidth=1.5, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('Rolling Sharpe Ratios', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Volatility
        ax2 = axes[1]
        ax2.plot(rolling_metrics.index, rolling_metrics['rolling_vol_1yr'] * 100,
                label='1-Year', linewidth=1.5, alpha=0.8)
        ax2.plot(rolling_metrics.index, rolling_metrics['rolling_vol_2yr'] * 100,
                label='2-Year', linewidth=1.5, alpha=0.8)
        ax2.plot(rolling_metrics.index, rolling_metrics['rolling_vol_3yr'] * 100,
                label='3-Year', linewidth=1.5, alpha=0.8)
        ax2.set_title('Rolling Volatility', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Volatility (%)', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Max Drawdown
        ax3 = axes[2]
        ax3.plot(rolling_metrics.index, rolling_metrics['rolling_maxdd_1yr'] * 100,
                label='1-Year', linewidth=1.5, alpha=0.8)
        ax3.plot(rolling_metrics.index, rolling_metrics['rolling_maxdd_2yr'] * 100,
                label='2-Year', linewidth=1.5, alpha=0.8)
        ax3.plot(rolling_metrics.index, rolling_metrics['rolling_maxdd_3yr'] * 100,
                label='3-Year', linewidth=1.5, alpha=0.8)
        ax3.set_title('Rolling Max Drawdown', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Max Drawdown (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    # =========================================================================
    # Summary Text File Generation
    # =========================================================================
    
    def _save_summary_text(self, output_dir: str, annual_perf: pd.DataFrame,
                          annual_comparison: pd.DataFrame = None,
                          win_rate_vs_benchmark: float = None,
                          consistency_std: float = None,
                          consistency: Dict = None,
                          regime_attribution: pd.DataFrame = None,
                          regime_performance: pd.DataFrame = None,
                          statistical_tests: Dict = None) -> None:
        """
        Save comprehensive summary text file with all key metrics.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save summary
        annual_perf : pd.DataFrame
            Annual performance DataFrame
        annual_comparison : pd.DataFrame, optional
            Annual comparison DataFrame
        win_rate_vs_benchmark : float, optional
            Win rate vs benchmark
        consistency_std : float, optional
            Consistency (std dev of annual returns)
        consistency : dict, optional
            Consistency metrics dictionary
        regime_attribution : pd.DataFrame, optional
            Regime attribution DataFrame
        regime_performance : pd.DataFrame, optional
            Regime performance DataFrame
        statistical_tests : dict, optional
            Statistical tests results
        """
        summary_path = f"{output_dir}/summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("COMPREHENSIVE STRATEGY PERFORMANCE SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall Performance Summary
            f.write("OVERALL PERFORMANCE SUMMARY\n")
            f.write("-" * 70 + "\n")
            total_return = (1 + self.df['returns']).prod() - 1
            years = len(self.df) / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            annual_vol = self.df['returns'].std() * np.sqrt(252)
            sharpe = ((self.df['returns'].mean() - self.daily_rf) / self.df['returns'].std()) * np.sqrt(252)
            
            cumulative = self.df['cumulative']
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            f.write(f"Total Return: {total_return:.2%}\n")
            f.write(f"Annualized Return: {annual_return:.2%}\n")
            f.write(f"Annualized Volatility: {annual_vol:.2%}\n")
            f.write(f"Sharpe Ratio: {sharpe:.3f}\n")
            f.write(f"Maximum Drawdown: {max_drawdown:.2%}\n")
            f.write(f"Period: {self.df.index[0].date()} to {self.df.index[-1].date()}\n")
            f.write(f"Trading Days: {len(self.df)}\n\n")
            
            # Annual Performance Comparison
            if annual_comparison is not None and win_rate_vs_benchmark is not None:
                f.write("ANNUAL PERFORMANCE COMPARISON\n")
                f.write("-" * 70 + "\n")
                f.write(f"Win Rate vs S&P 500: {win_rate_vs_benchmark:.1%}\n")
                f.write(f"  (Percentage of years strategy beats S&P 500)\n")
                if consistency_std is not None:
                    f.write(f"Consistency (Std Dev of Annual Returns): {consistency_std:.4f}\n")
                    f.write(f"  (Lower = more consistent returns across years)\n")
                f.write("\n")
            
            # Consistency Metrics
            if consistency is not None:
                f.write("CONSISTENCY METRICS\n")
                f.write("-" * 70 + "\n")
                for key, value in consistency.items():
                    if key == 'Win_Rate':
                        f.write(f"{key}: {value:.1%}\n")
                        f.write(f"  (Percentage of positive return days)\n")
                    elif key == 'VaR_95':
                        f.write(f"{key}: {value:.4f} ({value*100:.2f}%)\n")
                        f.write(f"  (Value at Risk at 95% confidence level)\n")
                    elif key == 'Downside_Deviation':
                        f.write(f"{key}: {value:.4f} ({value*100:.2f}%)\n")
                        f.write(f"  (Annualized downside volatility)\n")
                    elif key == 'Coefficient_of_Variation':
                        f.write(f"{key}: {value:.3f}\n")
                        f.write(f"  (Std dev / mean of annual returns; lower = more consistent)\n")
                    elif key == 'Calmar_Ratio':
                        f.write(f"{key}: {value:.3f}\n")
                        f.write(f"  (Annual return / max drawdown; higher = better)\n")
                    elif key == 'Sortino_Ratio':
                        f.write(f"{key}: {value:.3f}\n")
                        f.write(f"  (Return / downside deviation; higher = better)\n")
                    elif key == 'Profit_Factor':
                        f.write(f"{key}: {value:.3f}\n")
                        f.write(f"  (Gross gains / gross losses; >1 is good)\n")
                f.write("\n")
            
            # Regime Performance
            if regime_performance is not None:
                f.write("REGIME PERFORMANCE ANALYSIS\n")
                f.write("-" * 70 + "\n")
                f.write(regime_performance.to_string())
                f.write("\n\n")
            
            if regime_attribution is not None:
                f.write("REGIME ATTRIBUTION (Beta Analysis)\n")
                f.write("-" * 70 + "\n")
                f.write(regime_attribution.to_string())
                f.write("\n\n")
            
            # Statistical Tests Summary
            if statistical_tests is not None:
                f.write("STATISTICAL TESTS SUMMARY\n")
                f.write("-" * 70 + "\n")
                for test_name, test_results in statistical_tests.items():
                    if 'error' not in test_results:
                        f.write(f"{test_name}:\n")
                        if 'interpretation' in test_results:
                            f.write(f"  {test_results['interpretation']}\n")
                        if 'pvalue' in test_results:
                            f.write(f"  p-value: {test_results['pvalue']:.4f}\n")
                            f.write(f"  Significant: {'Yes' if test_results.get('significant', False) else 'No'} (α=0.05)\n")
                        f.write("\n")
            
            # Annual Performance Table
            f.write("ANNUAL PERFORMANCE BREAKDOWN\n")
            f.write("-" * 70 + "\n")
            f.write(annual_perf.to_string())
            f.write("\n\n")
            
            if annual_comparison is not None:
                f.write("ANNUAL RETURNS COMPARISON (Strategy vs S&P 500 vs Bonds)\n")
                f.write("-" * 70 + "\n")
                f.write(annual_comparison.to_string())
                f.write("\n")
        
        print(f"Summary saved to: {summary_path}")
    
    # =========================================================================
    # Main Analysis Function
    # =========================================================================
    
    def run_full_analysis(self, benchmark_returns: pd.Series = None,
                          bond_returns: pd.Series = None,
                          output_dir: str = None) -> Dict:
        """
        Run complete performance analysis with all features from todo list.
        
        Parameters:
        -----------
        benchmark_returns : pd.Series, optional
            Benchmark returns (e.g., S&P 500) for comparison and regime analysis
        bond_returns : pd.Series, optional
            Bond returns for comparison
        output_dir : str, optional
            Directory to save outputs
        
        Returns:
        --------
        dict
            Dictionary containing all analysis results
        """
        print("=" * 70)
        print("COMPREHENSIVE STRATEGY PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # 1. Annual Performance
        print("\n[1/6] Calculating annual performance...")
        annual_perf = self.calculate_annual_performance()
        
        # 2. Annual Performance Comparison
        annual_comparison = None
        win_rate_vs_benchmark = None
        consistency_std = None
        if benchmark_returns is not None:
            print("[2/6] Calculating annual performance comparison...")
            annual_comparison = self.calculate_annual_comparison(benchmark_returns, bond_returns)
            win_rate_vs_benchmark = self.calculate_win_rate_vs_benchmark(benchmark_returns)
            consistency_std = self.calculate_consistency_std(annual_perf)
        
        # 3. Rolling Metrics
        print("[3/6] Calculating rolling metrics...")
        rolling_metrics = self.calculate_rolling_metrics()
        
        # 4. Consistency Metrics (enhanced)
        print("[4/6] Calculating enhanced consistency metrics...")
        consistency = self.calculate_consistency_metrics(annual_perf)
        
        # 5. Regime Analysis
        regime_attribution = None
        regime_performance = None
        if benchmark_returns is not None:
            print("[5/6] Calculating regime attribution and performance...")
            regime_attribution = self.calculate_regime_attribution(benchmark_returns)
            regime_performance = self.test_regime_performance(benchmark_returns)
        
        # 6. Statistical Tests
        statistical_tests = None
        if benchmark_returns is not None:
            print("[6/6] Performing statistical tests...")
            try:
                statistical_tests = self.perform_statistical_tests(benchmark_returns)
            except ImportError:
                print("Warning: scipy not available. Skipping statistical tests.")
                print("Install with: pip install scipy")
        
        # Print Summary
        print("\n" + "=" * 70)
        print("ANNUAL PERFORMANCE SUMMARY")
        print("=" * 70)
        print(annual_perf.to_string())
        
        if annual_comparison is not None:
            print("\n" + "=" * 70)
            print("ANNUAL PERFORMANCE COMPARISON")
            print("=" * 70)
            print(annual_comparison.to_string())
            print(f"\nWin Rate vs S&P 500: {win_rate_vs_benchmark:.1%}")
            print(f"Consistency (Std Dev of Annual Returns): {consistency_std:.4f}")
        
        print("\n" + "=" * 70)
        print("ENHANCED CONSISTENCY METRICS")
        print("=" * 70)
        for key, value in consistency.items():
            if key == 'Win_Rate':
                print(f"{key}: {value:.1%}")
            elif key == 'VaR_95':
                print(f"{key}: {value:.4f} ({value*100:.2f}%)")
            elif key == 'Downside_Deviation':
                print(f"{key}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"{key}: {value:.3f}")
        
        if regime_attribution is not None:
            print("\n" + "=" * 70)
            print("REGIME ATTRIBUTION")
            print("=" * 70)
            print(regime_attribution.to_string())
        
        if regime_performance is not None:
            print("\n" + "=" * 70)
            print("REGIME PERFORMANCE TEST")
            print("=" * 70)
            print(regime_performance.to_string())
        
        if statistical_tests is not None:
            print("\n" + "=" * 70)
            print("STATISTICAL TESTS")
            print("=" * 70)
            for test_name, test_results in statistical_tests.items():
                print(f"\n{test_name}:")
                if 'error' in test_results:
                    print(f"  Error: {test_results['error']}")
                else:
                    for key, value in test_results.items():
                        if key not in ['significant', 'interpretation']:
                            if isinstance(value, float):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
                    if 'interpretation' in test_results:
                        print(f"  → {test_results['interpretation']}")
        
        # Visualizations
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        if output_dir:
            # Core visualizations
            self.plot_annual_performance(annual_perf, 
                                       f"{output_dir}/annual_performance.png")
            self.plot_comprehensive_analysis(annual_perf, rolling_metrics,
                                           f"{output_dir}/comprehensive_analysis.png")
            self.plot_rolling_metrics_enhanced(rolling_metrics,
                                             f"{output_dir}/rolling_metrics_enhanced.png")
            self.plot_metrics_heatmap(annual_perf,
                                    f"{output_dir}/metrics_heatmap.png")
            
            # Comparison visualizations
            if annual_comparison is not None:
                self.plot_annual_comparison(annual_comparison,
                                          f"{output_dir}/annual_comparison.png")
            
            if regime_performance is not None:
                self.plot_regime_comparison(regime_performance,
                                          f"{output_dir}/regime_comparison.png")
        else:
            self.plot_annual_performance(annual_perf)
            self.plot_comprehensive_analysis(annual_perf, rolling_metrics)
            self.plot_rolling_metrics_enhanced(rolling_metrics)
            self.plot_metrics_heatmap(annual_perf)
            
            if annual_comparison is not None:
                self.plot_annual_comparison(annual_comparison)
            
            if regime_performance is not None:
                self.plot_regime_comparison(regime_performance)
        
        # Save Data
        if output_dir:
            print(f"\nSaving results to {output_dir}/...")
            annual_perf.to_csv(f"{output_dir}/annual_performance.csv")
            # Drop rows where all rolling metrics are NaN (before enough history)
            rolling_metrics_clean = rolling_metrics.dropna(how='all')
            rolling_metrics_clean.to_csv(f"{output_dir}/rolling_metrics.csv", na_rep='NaN')
            pd.DataFrame([consistency]).to_csv(f"{output_dir}/consistency_metrics.csv", index=False)
            
            if annual_comparison is not None:
                annual_comparison.to_csv(f"{output_dir}/annual_comparison.csv")
            
            if regime_attribution is not None:
                regime_attribution.to_csv(f"{output_dir}/regime_attribution.csv")
            
            if regime_performance is not None:
                regime_performance.to_csv(f"{output_dir}/regime_performance.csv")
            
            if statistical_tests is not None:
                # Save statistical tests as JSON-like text
                import json
                with open(f"{output_dir}/statistical_tests.txt", 'w') as f:
                    for test_name, test_results in statistical_tests.items():
                        f.write(f"{test_name}:\n")
                        if 'error' in test_results:
                            f.write(f"  Error: {test_results['error']}\n")
                        else:
                            for key, value in test_results.items():
                                if isinstance(value, (int, float)):
                                    f.write(f"  {key}: {value:.6f}\n")
                                else:
                                    f.write(f"  {key}: {value}\n")
                        f.write("\n")
            
            # Save comprehensive summary text file
            self._save_summary_text(
                output_dir=output_dir,
                annual_perf=annual_perf,
                annual_comparison=annual_comparison,
                win_rate_vs_benchmark=win_rate_vs_benchmark,
                consistency_std=consistency_std,
                consistency=consistency,
                regime_attribution=regime_attribution,
                regime_performance=regime_performance,
                statistical_tests=statistical_tests
            )
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        
        return {
            'annual_performance': annual_perf,
            'annual_comparison': annual_comparison,
            'win_rate_vs_benchmark': win_rate_vs_benchmark,
            'consistency_std': consistency_std,
            'rolling_metrics': rolling_metrics,
            'consistency_metrics': consistency,
            'regime_attribution': regime_attribution,
            'regime_performance': regime_performance,
            'statistical_tests': statistical_tests
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Load your strategy returns
    # df = pd.read_csv('strategy_returns.csv', index_col=0, parse_dates=True)
    # analyzer = StrategyPerformanceAnalyzer(df, risk_free_rate=0.04)
    # results = analyzer.run_full_analysis(output_dir='../results')
    pass

