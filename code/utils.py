"""
Shared utility functions for VIX strategy analysis.

All scripts should use these functions to ensure consistency.
"""

import pandas as pd
import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate annualized Sharpe ratio using standard arithmetic mean approach.
    
    This is the industry-standard formula used by Bloomberg, Morningstar, etc.
    
    Formula: Sharpe = (mean(excess_returns) / std(excess_returns)) * sqrt(252)
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns (as decimals, e.g., 0.01 for 1%)
    risk_free_rate : float
        Annual risk-free rate (default 0.02 for 2%)
    
    Returns:
    --------
    float
        Annualized Sharpe ratio
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate daily excess returns
    excess_returns = returns - daily_rf
    
    # Mean and std dev of excess returns
    mean_excess = excess_returns.mean()
    std_dev = excess_returns.std()
    
    # Avoid division by zero
    if std_dev == 0:
        return np.nan
    
    # Annualize: Sharpe = (mean_excess / std_dev) * sqrt(252)
    return (mean_excess / std_dev) * np.sqrt(252)


def calculate_annual_return(returns):
    """
    Calculate compound annual growth rate (CAGR).
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns (as decimals)
    
    Returns:
    --------
    float
        Annualized return (as decimal, e.g., 0.10 for 10%)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    return (1 + total_return) ** (1 / years) - 1


def calculate_annual_volatility(returns):
    """
    Calculate annualized volatility.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns (as decimals)
    
    Returns:
    --------
    float
        Annualized volatility (as decimal)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    return returns.std() * np.sqrt(252)


def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns (as decimals)
    
    Returns:
    --------
    float
        Maximum drawdown (as decimal, negative value)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def calculate_metrics(returns, risk_free_rate=0.02):
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns (as decimals)
    risk_free_rate : float
        Annual risk-free rate (default 0.02)
    
    Returns:
    --------
    dict
        Dictionary with metrics:
        - annual_return: Annualized return (as decimal)
        - annual_vol: Annualized volatility (as decimal)
        - sharpe: Sharpe ratio
        - max_drawdown: Maximum drawdown (as decimal, negative)
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan
        }
    
    return {
        "annual_return": calculate_annual_return(returns),
        "annual_vol": calculate_annual_volatility(returns),
        "sharpe": calculate_sharpe_ratio(returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(returns)
    }


def download_data(tickers, start_date, end_date):
    """
    Download and clean financial data from yfinance.
    
    Parameters:
    -----------
    tickers : dict
        Dictionary mapping names to ticker symbols, e.g. {"Stock": "VOO"}
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns for each ticker's Close price
    """
    import yfinance as yf
    import warnings
    warnings.filterwarnings('ignore')
    
    data_dict = {}
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
        data_dict[name] = data["Close"].rename(name)
    
    df = pd.concat(data_dict, axis=1).dropna()
    return df


def calculate_progressive_bond_allocation(vix, threshold, scaling='linear', 
                                        max_excess=None, power=2.0, steepness=5.0):
    """
    Calculate progressive bond allocation based on VIX excess over threshold.
    
    Instead of binary switching (0% or 100% bonds), this allocates a portion
    to bonds based on how much VIX exceeds the threshold.
    
    Parameters:
    -----------
    vix : pd.Series or float
        VIX index values
    threshold : pd.Series or float
        VIX threshold (e.g., rolling percentile)
    scaling : str
        Scaling function type:
        - 'linear': Linear scaling (bond_allocation = excess / max_excess)
        - 'sigmoid': Smooth S-curve transition (sigmoid function)
        - 'exponential': Exponential scaling (faster transition)
        - 'power': Power scaling (configurable curve, default power=2)
    max_excess : float, optional
        Maximum expected excess for normalization (used in linear scaling).
        If None, uses rolling 95th percentile of historical excess to avoid look-ahead bias.
    power : float
        Power parameter for 'power' scaling (default 2.0)
    steepness : float
        Steepness parameter for 'sigmoid' and 'exponential' scaling (default 5.0)
    
    Returns:
    --------
    pd.Series or float
        Bond allocation (0.0 to 1.0, where 1.0 = 100% bonds)
    """
    # Calculate excess (how much VIX exceeds threshold)
    excess = vix - threshold
    excess = excess.clip(lower=0)  # Only consider positive excess
    
    # If no excess, return 0% bonds
    if isinstance(excess, pd.Series):
        mask = excess > 0
        if not mask.any():
            return pd.Series(0.0, index=excess.index)
    else:
        if excess <= 0:
            return 0.0
    
    # Normalize excess for scaling functions
    if max_excess is None:
        if isinstance(excess, pd.Series):
            # Use rolling window to calculate max_excess to avoid look-ahead bias
            # Use 252-day (1 year) rolling window for 95th percentile
            # This ensures we only use past data for normalization
            rolling_max_excess = excess.rolling(window=252, min_periods=63).quantile(0.95)
            
            # Fill NaN values at the beginning with expanding window
            expanding_max_excess = excess.expanding(min_periods=1).quantile(0.95)
            max_excess_series = rolling_max_excess.fillna(expanding_max_excess)
            
            # Ensure we have a minimum value (avoid division by zero)
            # Use a reasonable minimum based on typical VIX excess
            max_excess_series = max_excess_series.clip(lower=5.0)  # Minimum 5 VIX points excess
            
            # Normalize using rolling max_excess
            normalized_excess = excess / max_excess_series
        else:
            max_excess = excess if excess > 0 else 5.0
            normalized_excess = excess / max_excess
    else:
        # Use provided max_excess (scalar)
        if isinstance(excess, pd.Series):
            normalized_excess = excess / max_excess
        else:
            normalized_excess = excess / max_excess
    
    # Apply scaling function
    if scaling == 'linear':
        bond_allocation = normalized_excess.clip(upper=1.0)
    
    elif scaling == 'sigmoid':
        # Sigmoid: smooth S-curve transition
        # Formula: 1 / (1 + exp(-steepness * (normalized_excess - 0.5)))
        # Shifted so that normalized_excess=0.5 maps to ~50% allocation
        bond_allocation = 1 / (1 + np.exp(-steepness * (normalized_excess - 0.5)))
        bond_allocation = bond_allocation.clip(upper=1.0)
    
    elif scaling == 'exponential':
        # Exponential: faster transition
        # Formula: 1 - exp(-steepness * normalized_excess)
        bond_allocation = 1 - np.exp(-steepness * normalized_excess)
        bond_allocation = bond_allocation.clip(upper=1.0)
    
    elif scaling == 'power':
        # Power scaling: configurable curve
        # Formula: normalized_excess^power
        bond_allocation = np.power(normalized_excess.clip(upper=1.0), power)
    
    else:
        raise ValueError(f"Unknown scaling type: {scaling}. Use 'linear', 'sigmoid', 'exponential', or 'power'")
    
    return bond_allocation


def calculate_progressive_strategy_returns(stock_returns, bond_returns, bond_allocation):
    """
    Calculate strategy returns using progressive allocation.
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Stock returns (daily)
    bond_returns : pd.Series
        Bond returns (daily)
    bond_allocation : pd.Series
        Bond allocation (0.0 to 1.0)
    
    Returns:
    --------
    pd.Series
        Strategy returns
    """
    stock_allocation = 1 - bond_allocation
    return stock_allocation * stock_returns + bond_allocation * bond_returns

