from datetime import datetime
from collections import defaultdict
import numpy as np

from src.feature_engineering.utils import calculate_return

from src.config.settings import RETURN_ROUND_TO


# Not needed
def calculate_retvol(prices_daily):
    """
    Calculates the monthly return volatility (retvol) based on daily price data.

    For each month, this function computes the standard deviation of the daily returns,
    where each return is calculated as the percentage change between the price on two consecutive days.

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries where each dictionary contains:
            - "date" (str): The date in "YYYY-MM-DD" format.
            - "price" (float): The closing price for the day.

    Returns
    -------
    dict
        - key (str): The month in "YYYY-MM" format.
        - value (float): The standard deviation of daily returns for that month.

    Notes
    -----
    - Assumes that `prices_daily` is sorted in descending order (latest date first).
    - Uses the global variable `return_decimal_approximation` for rounding in the return calculation.
    - months_sorted with fewer than 2 return entries will still be included, but the standard deviation may be 0 or NaN.
    """
    daily_returns_monthly = defaultdict(list)

    for i in range(len(prices_daily) - 1):
        entry_later = prices_daily[i]
        entry_earlier = prices_daily[i + 1]

        date = datetime.strptime(entry_later["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"

        daily_returns_monthly[month].append(
            calculate_return(
                entry_later["price"],
                entry_earlier["price"],
                round_to=RETURN_ROUND_TO,
            )
        )  # Add return to current month

    retvol = {
        key_month: np.std(daily_returns_monthly[key_month])
        for key_month in daily_returns_monthly.keys()
    }  # Get std deviation of each month's daily returns

    # Check the last entry in the list separately to handle the case where it's the only entry in the month
    last_entry = prices_daily[-1]
    date_last = datetime.strptime(last_entry["date"], "%Y-%m-%d")
    month_last = f"{date_last.year}-{date_last.month:02d}"

    # If the last entry is the only one in the month, set its return as None
    if month_last not in retvol:
        retvol[month_last] = None

    return retvol


def calculate_retvol_std(daily_returns_monthly):
    """
    Computes the monthly return volatility (retvol) as the standard deviation of daily returns.

    For each month in the input dictionary, this function calculates the standard deviation
    of daily returns, representing the volatility of stock returns over that month.

    Parameters
    ----------
    daily_returns_monthly : dict
        A dictionary where:
            - key (str): A month in "YYYY-MM" format.
            - value (list of float or None): Daily return values for that month.
              A value of None indicates that no daily returns were computed for that month.

    Returns
    -------
    dict
        A dictionary where:
            - key (str): The month in "YYYY-MM" format.
            - value (float or None): The standard deviation of daily returns (volatility) for that month.
              Returns `None` if there are no valid daily returns for that month.

    Notes
    -----
    - Standard deviation is computed using NumPy's `np.std()` function.
    - Months with `None` as the value in `daily_returns_monthly` or empty return
    """
    # Get standard deviation of the monthly daily returns, but if the month has no returns, None is assigned
    retvol = {
        key_month: (np.std(daily_returns_monthly[key_month]))
        for key_month in daily_returns_monthly
    }

    return retvol


def calculate_idiovol(
    weeks_sorted,
    months_sorted,
    month_latest_week,
    weekly_returns,
    market_weekly_returns,
    interval=156,
    increment=4,
    current=False,
):
    """
    Calculate idiosyncratic volatility (idiovol) for each month using a 3-year rolling window
    of weekly returns, measured as the standard deviation of the difference between a stock's
    return and the market return over the same period.

    Parameters
    ----------
    weeks_sorted: list
        Sorted list of weeks in "YYYY-WW" format.

    months_sorted
        A list of strings representing months_sorted in "YYYY-MM" format, ordered from most recent to oldest

    month_latest_week
        A dictionary mapping each month to its corresponding last week as a (year, week) tuple

    weekly_returns
        A dictionary mapping (year, week) tuples to the stock's weekly returns (float)

    market_weekly_returns
        A dictionary mapping (year, week) tuples to the market's weekly returns (float)

    interval
        Number of weeks in the rolling window used to calculate idiovol (default is 156, or approximately 3 years)

    increment
        Step size in weeks between calculations, typically 4 to move month by month (default is 4)

    current : bool, optional
        - If True, starts with the current month in the sliding window.
        - If False (default), starts with the previous month in the sliding window.

    Returns
    --------
    dict
        A dictionary mapping each month (string) to its corresponding idiovol value (float), or None if insufficient data
    """

    idiovol_by_month = {}

    # Generate aligned lists of stock and market returns
    weekly_returns_list = [weekly_returns[k] for k in weeks_sorted]
    market_returns_list = [market_weekly_returns[k] for k in weeks_sorted]

    month_current_index = 0
    if current:
        month_start = months_sorted[month_current_index]
    else:
        month_start = months_sorted[1]

    week_start = month_latest_week[month_start]

    try:
        week_start = weeks_sorted.index(week_start)
    except ValueError:
        print(f"Week {week_start} not found in weekly_returns keys.")
        return {m: None for m in months_sorted}

    while month_current_index < len(months_sorted):

        if week_start + interval < len(weeks_sorted):
            # Get stock and market returns for the interval provided
            stock_window = weekly_returns_list[week_start : week_start + interval]
            market_window = market_returns_list[week_start : week_start + interval]

            # Compute residuals and standard deviation
            residuals_squared = [
                (s - m) ** 2 for s, m in zip(stock_window, market_window)
            ]
            idiovol = (sum(residuals_squared) / interval) ** 0.5

            idiovol_by_month[months_sorted[month_current_index]] = idiovol
        else:
            # If not enough data, assign None
            idiovol_by_month[months_sorted[month_current_index]] = None

        week_start += increment
        month_current_index += 1

    return idiovol_by_month


def calculate_beta_betasq(
    weeks_sorted,
    months_sorted,
    month_latest_week,
    weekly_returns,
    market_weekly_returns,
    interval=156,
    increment=4,
    current=False,
):
    """
    Calculate rolling market beta for each month using weekly returns over the past 3 years.

    Beta is defined as the covariance of stock and market returns divided by the variance of the market returns.
    It measures the sensitivity of the stock's return to market movements.

    Parameters
    ----------
    weeks_sorted: list
        Sorted list of weeks in "YYYY-WW" format.

    months_sorted : list
        List of month strings (e.g., "YYYY-MM") ordered from most recent to oldest.

    month_latest_week : dict
        Mapping of each month to the last week of that month as a (year, week) tuple.

    weekly_returns : dict
        Mapping of (year, week) tuples to the stock's weekly returns (float).

    market_weekly_returns : dict
        Mapping of (year, week) tuples to the market's weekly returns (float).

    interval : int
        The number of weeks in the rolling window (default is 156 weeks, approximately 3 years).

    increment : int
        The step size in weeks to move the rolling window forward (default is 4 weeks).

    current : bool, optional
        - If True, starts with the current month in the sliding window.
        - If False (default), starts with the previous month in the sliding window.

    Returns
    -------
    dict
        A dictionary mapping each month (string) to its corresponding beta value (float).
        If there is insufficient data, the value will be None.

    dict
        A dictionary mapping each month (string) to the squared beta value (float).
        If there is insufficient data, the value will be None.
    """

    beta_by_month = {}
    betasq_by_month = {}

    # Get a sorted list of the returns rather than the week:returns, for easier access
    stock_returns_list = [weekly_returns[k] for k in weeks_sorted]
    market_returns_list = [market_weekly_returns[k] for k in weeks_sorted]

    current_index = 0
    if current:
        month_start = months_sorted[current_index]
    else:
        month_start = months_sorted[1]

    week_start = month_latest_week[month_start]

    # Get the week to start from, which is the latest week from the previous month (1) since current_index is 0
    try:
        start = weeks_sorted.index(week_start)
    except ValueError:
        print(f"Week {week_start} not found.")
        return {m: None for m in months_sorted}

    while current_index < len(months_sorted):
        if start + interval < len(weeks_sorted):
            # Getting the stock and market returns' windows
            stock_window = stock_returns_list[start : start + interval]
            market_window = market_returns_list[start : start + interval]

            stock_np = np.array(stock_window)
            market_np = np.array(market_window)

            # Remove NaNs and ensure we have at least 52 valid points
            mask = ~np.isnan(stock_np) & ~np.isnan(market_np)
            if mask.sum() >= 52:
                stock_np = stock_np[mask]
                market_np = market_np[mask]

                # Returns covariance matrix, so will just get the covariance between them
                cov = np.cov(stock_np, market_np)[0][1]
                var = np.var(market_np)

                beta = cov / var if var != 0 else None

                beta_by_month[months_sorted[current_index]] = beta
                betasq_by_month[months_sorted[current_index]] = beta
            else:
                beta_by_month[months_sorted[current_index]] = None
                betasq_by_month[months_sorted[current_index]] = None
        else:
            beta_by_month[months_sorted[current_index]] = None
            betasq_by_month[months_sorted[current_index]] = None

        start += increment
        current += 1

    return beta_by_month, betasq_by_month
