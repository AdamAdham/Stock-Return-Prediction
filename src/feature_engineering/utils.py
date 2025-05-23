from datetime import datetime
from src.config.settings import RETURN_ROUND_TO
from collections import defaultdict

import pandas as pd
import numpy as np


def calculate_return(
    price_later: float, price_earlier: float, round_to: int | None = None
) -> float | None:
    """
    Computes the percentage return between two price points.

    The percentage return is calculated as the relative change from the earlier price to the later price.
    The result can optionally be rounded to a specified number of decimal places.

    Parameters
    ----------
    price_later : float
        The price at the later time point.
    price_earlier : float
        The price at the earlier time point.
    round_to : int, optional
        The number of decimal places to round the result to.
        If None (default), the result is returned as a raw float.

    Returns
    -------
    float
        The calculated percentage return between the two price points. If `round_to` is specified,
        the result is rounded to the specified number of decimal places. Returns `None` if `price_earlier` is zero to avoid division by zero.
    """

    if price_earlier == 0:
        return None  # Prevent division by zero

    return_calc = ((price_later - price_earlier) / price_earlier) * 100
    if round_to is None:
        return return_calc
    else:
        return round(return_calc, round_to)


# Momentum Variables


def get_monthly_price(prices_daily: list[dict]) -> tuple[list[str], dict[str, float]]:
    """
    Aggregates daily stock prices into monthly prices by selecting the latest available price for each month.

    The function iterates through a list of daily stock price entries, extracts the month and year
    from each entry's date, and stores the latest price encountered for each month.

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries where each dictionary contains:
        - "date" (str or datetime): The date in "YYYY-MM-DD" format or as a datetime object assumes sorted from latest to earliest.
        - "price" (float): The stock price on that date.

    Returns
    -------
    tuple
        A tuple containing:
        - months_sorted (list of str): A list of unique months_sorted (formatted as "YYYY-MM") found in the dataset and acts as index to prices_monthly (returned in the same order of prices_daily dates).
        - prices_monthly (dict): A dictionary mapping each month ("YYYY-MM") to its latest available price.

    Notes
    -----
    - If the "date" field is a string, it is converted into a datetime object.
    - The function assumes that the daily prices is sorted latest to earliest dates.
    """

    prices_monthly = {}
    months_sorted = []
    seen = set()

    # Get latest price of each month
    for entry in prices_daily:
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"

        if month not in seen:
            seen.add(month)
            months_sorted.append(month)
            prices_monthly[month] = entry["price"]

    return months_sorted, prices_monthly


def get_weekly_monthly_summary(
    prices_daily: list[dict],
) -> tuple[
    list[str],  # weeks_sorted
    dict[str, float],  # prices_weekly
    list[str],  # months_sorted
    dict[str, str],  # month_latest_week
    dict[str, float],  # prices_monthly
    dict[str, dict[str, float]],  # dollar_volume_monthly
    dict[str, float],  # vol_sum_monthly
    dict[str, int],  # zero_trading_days_count_monthly
    dict[str, int],  # trading_days_count_monthly
    dict[str, float | None],  # maxret_monthly
    dict[str, list[float]],  # daily_returns_monthly
]:
    """
    Computes a comprehensive weekly and monthly summary from a list of daily stock data.

    This function aggregates daily price and volume data into multiple temporal resolutions and metrics,
    including:
    - Latest weekly and monthly closing prices
    - Monthly dollar volume (price × volume)
    - Monthly trading day statistics
    - Daily volume lookup
    - Weekly price return reference
    - Monthly maximum daily return

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries representing daily stock data. Each dictionary should contain:
            - "date" (str): The date of the entry in "YYYY-MM-DD" format. Assumed to be sorted from latest to earliest.
            - "price" (float): Closing price of the stock on the given date.
            - "volume" (float): Trading volume on the given date.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - weeks_sorted : list of str
            List of unique ISO weeks (formatted as "YYYY-WW"), sorted from latest to earliest.
        - prices_weekly : dict
            Dictionary mapping each ISO week to the latest available stock price in that week.
        - months_sorted : list of str
            List of unique months (formatted as "YYYY-MM") in the order they appear in the input (latest to earliest).
        - month_latest_week : dict
            Mapping from each month to its latest week identifier (used for aligning weekly/monthly calculations).
        - prices_monthly : dict
            Mapping from each month to the latest available closing price.
        - dollar_volume_monthly : dict
            Mapping from each month to a dictionary with total and count of daily dollar volumes:
            {'sum': float, 'count': int}.
        - vol_sum_monthly : dict
            Mapping from each month to the total trading volume.
        - zero_trading_days_count_monthly : dict
            Mapping from each month to the count of days with zero trading volume.
        - trading_days_count_monthly : dict
            Mapping from each month to the total number of trading days.
        - maxret_monthly : dict
            Mapping from each month to the highest single-day return observed that month.
            If a month has only one day, return is `None`.
        - daily_returns_monthly : dict
            Mapping from each month "YYYY-MM" to a list of daily returns.
    """
    months_sorted = []
    seen = set()
    prices_monthly = {}

    dollar_volume_monthly = {}

    prices_weekly = {}
    month_latest_week = {}
    weeks_sorted = []

    vol_sum_monthly = {}
    vol_daily = {}

    zero_trading_days_count_monthly = {}
    trading_days_count_monthly = {}

    maxret_monthly = {}

    daily_returns_monthly = defaultdict(list)

    # Get latest price of each month
    for i, entry in enumerate(prices_daily):
        # Parse date into month, week and day
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"
        year, week_number, _ = date.isocalendar()
        week_key = f"{year}-{week_number:02d}"  # "YYYYY-WW" str due to not being able to dump json
        day = f"{date.year}-{date.month:02d}-{date.day:02d}"

        price = entry["price"]
        volume = entry["volume"]

        dollar_volume = price * volume  # Calculate dollar value

        # Monthly prices handling
        if month not in seen:
            seen.add(month)
            months_sorted.append(month)
            prices_monthly[month] = price

        # Dollar volume handling
        if month not in dollar_volume_monthly:
            dollar_volume_monthly[month] = {"sum": dollar_volume, "count": 1}
        else:
            dollar_volume_monthly[month]["sum"] += dollar_volume
            dollar_volume_monthly[month]["count"] += 1

        # Weekly returns, latest week handling
        if week_key not in prices_weekly:
            prices_weekly[week_key] = price
            weeks_sorted.append(week_key)

        # Latest week in the month
        if month not in month_latest_week:
            month_latest_week[month] = week_key

        # Sum the daily volume for the month
        vol_sum_monthly[month] = vol_sum_monthly.get(month, 0) + volume

        # Count total trading days
        trading_days_count_monthly[month] = trading_days_count_monthly.get(month, 0) + 1

        # Initialize zerotrade for first occurrences
        if month not in zero_trading_days_count_monthly:
            zero_trading_days_count_monthly[month] = 0

        # Count zero trading days
        if volume == 0:
            zero_trading_days_count_monthly[month] += 1

        # Create a daily volume lookup to use to calculate the daily turnover when loop through the shares
        vol_daily[day] = volume

        if i < len(prices_daily) - 1:
            price_previous = prices_daily[i + 1]["price"]
            return_current = calculate_return(
                price,
                price_previous,
                round_to=RETURN_ROUND_TO,
            )
        # If last return set it to None
        else:
            return_current = None

        # Calculate maxret_current
        # If this is the first calculated return in this month or
        # the the current calculated return is not None (not last price in list) and
        # is higher than the previously highest return
        if month not in maxret_monthly or (
            return_current is not None and maxret_monthly[month] < return_current
        ):
            maxret_monthly[month] = return_current

        # Get daily_returns
        if return_current is not None:
            daily_returns_monthly[month].append(return_current)
        elif month not in daily_returns_monthly:
            # First value of the month but it is last value of the prices so return is empty array
            # since no returns present and so the np.std([]) to be None
            daily_returns_monthly[month] = []

    return (
        weeks_sorted,
        prices_weekly,
        months_sorted,
        month_latest_week,
        prices_monthly,
        dollar_volume_monthly,
        vol_sum_monthly,
        zero_trading_days_count_monthly,
        trading_days_count_monthly,
        maxret_monthly,
        daily_returns_monthly,
    )


def get_shares_monthly(outstanding_shares: dict[str, float]):
    """
    Aggregate daily outstanding shares into monthly outstanding shares by choosing the outstanding share for
    the last trading day of that month.

    Parameters
    ----------
    outstanding_shares : dict
        A dictionary where keys are dates in "YYYY-MM-DD" format, and values are the number of outstanding
        shares on that specific date.

    Returns
    -------
    dict
        A dictionary where keys are month identifiers in the format "YYYY-MM", and values are the number of
        outstanding shares for that month (taken from the first occurrence of each month).
    """
    shares_monthly = {}

    shares_keys = sorted(
        outstanding_shares.keys(), reverse=True
    )  # Dates sorted from latest to earliest
    for key_date in shares_keys:
        date = datetime.strptime(key_date, "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"

        shares = outstanding_shares[key_date]
        if month not in shares_monthly:
            shares_monthly[month] = shares

    return shares_monthly


def get_returns_weekly(
    weeks_sorted: list[str], price_weekly: dict[str, float]
) -> dict[str, float | None]:
    """
    Calculate the weekly stock returns based on the closing prices of the last trading day of each week.

    The return for each week is calculated as the percentage change in the closing price from the previous
    week's last trading day to the current week's last trading day.

    Parameters
    ----------
    weeks_sorted : list of str
        A sorted list of week identifiers in "YYYY-WW" format, where each element represents a calendar week.
        The list should be sorted in descending order (latest week first).

    price_weekly : dict
        A dictionary where keys are week identifiers (in "YYYY-WW" format), and values are the closing prices
        of the stock on the last trading day of each week.

    Returns
    -------
    dict
        A dictionary where keys are week identifiers (in "YYYY-WW" format), and values are the calculated
        weekly returns (float), based on the percentage change in the closing prices between consecutive weeks.
        The last week in the sorted list will have a return value of None as it has no next week to compare to.

    """
    returns_weekly = {}
    for i in range(len(weeks_sorted) - 1):
        price_current = price_weekly[weeks_sorted[i]]
        price_previous = price_weekly[weeks_sorted[i + 1]]

        returns_weekly[weeks_sorted[i]] = calculate_return(
            price_current, price_previous, round_to=RETURN_ROUND_TO
        )
    # Assign None to remaining week
    returns_weekly[weeks_sorted[-1]] = None

    return returns_weekly


# Liquidity Variables


def get_dollar_volume_monthly(prices_daily: list[dict]) -> dict[str, dict[str, float]]:
    """
    Aggregates daily dollar trading volume into monthly totals and counts.

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries, each containing:
            - "date" (str): The trading date in "YYYY-MM-DD" format.
            - "price" (float): The closing price for the day.
            - "volume" (float): The trading volume for the day.

    Returns
    -------
    dict
        A dictionary where each key is a month in "YYYY-MM" format and each value is
        another dictionary with:
            - "sum": Total dollar volume for the month.
            - "count": Number of trading days in the month.

    Notes
    -----
    - Assumes all entries in `prices_daily` contain valid "date", "price", and "volume" keys.
    - Dates must be in "YYYY-MM-DD" format for proper parsing.
    """

    dollar_volume_monthly = {}

    for entry in prices_daily:
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"

        dollar_volume = entry["price"] * entry["volume"]  # Calculate value

        # Check if first occurrence of current month
        if month not in dollar_volume_monthly:
            dollar_volume_monthly[month] = {"sum": dollar_volume, "count": 1}
        else:
            dollar_volume_monthly[month]["sum"] += dollar_volume
            dollar_volume_monthly[month]["count"] += 1

    return dollar_volume_monthly


def get_market_cap_monthly(market_caps: list[dict]) -> dict[str, float]:
    """
    Extract the market capitalization (price) at the last available trading day of each month.

    This function processes a list of daily market capitalization data and retrieves
    the price corresponding to the last trading day for each month. It assumes that
    the input data is ordered in descending date order (i.e., latest date first).

    Parameters
    ----------
    market_caps : list of dict
        A list of dictionaries containing daily market capitalization data with keys:
        - "date" (str): The date of the entry in the format "YYYY-MM-DD".
        - "marketCap" (float or int): The market capitalization on that day.

    Returns
    -------
    dict
        A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
        and values are the market capitalization (price) from the last trading day of that month.
    """
    market_cap_monthly = {}

    # Get latest price of each month
    for entry in market_caps:
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"

        if month not in market_cap_monthly:
            market_cap_monthly[month] = entry["marketCap"]

    return market_cap_monthly


# Risk Measures:


def get_rolling_returns_weekly(
    weeks_sorted: list,
    months_sorted: list[str],
    month_latest_week: dict[str, str],
    returns_weekly: dict[str, float],
    interval: int = 156,
    increment: int = 4,
    current: bool = False,
) -> dict[str, float]:
    """
    Compute rolling average of weekly returns for each month over a specified interval.

    This function calculates the average of weekly stock returns over a fixed interval
    (default 156 weeks, approximately 3 years) for each month in the `months_sorted` list.
    The calculation starts from the last trading week prior to each month and rolls backward.
    The function moves forward by a set increment (default 4 weeks) for each new month.

    Parameters
    ----------
    weeks_sorted: list
        Sorted list of weeks in "YYYY-WW" format.

    months_sorted : list of str
        A list of month identifiers in the format "YYYY-MM", sorted in descending order (most recent first).

    month_latest_week : dict
        A dictionary mapping each month ("YYYY-MM") to the latest trading week prior to the month's end.
        Each value is a tuple of "YYYY-WW".

    returns_weekly : dict
        A dictionary where keys are tuples "YYYY-WW" and values are the weekly return (float)
        for that week.

    interval : int, optional
        The number of weeks to include in the rolling average (default is 156 weeks).

    increment : int, optional
        The number of weeks to move forward in each step (default is 4 weeks, roughly one month).

    current : bool, optional
        - If True, uses current month's weeks.
        - If False (default), starts with previous month.

    Returns
    -------
    dict
        A dictionary where keys are month identifiers ("YYYY-MM") and values are the rolling average
        of weekly returns over the specified interval. If insufficient data exists to compute the
        average for a given month, the value will be None.
    """

    rolling_returns_weekly = {}

    returns_weekly_list = [returns_weekly[k] for k in weeks_sorted]

    current_index = 0
    if current:
        month_start = months_sorted[current_index]
    else:
        month_start = months_sorted[1]

    week_start = month_latest_week[month_start]

    # Get index of the latest week of the month to start which is the previous one since in paper states "prior to month end."
    start = weeks_sorted.index(week_start)

    while current_index < len(months_sorted):

        if start + interval < len(returns_weekly_list):
            # Rolling average of the 156 weekly returns
            rolling_returns_weekly[months_sorted[current_index]] = (
                sum(returns_weekly_list[start : start + interval]) / interval
            )
        else:
            # Not enough data
            rolling_returns_weekly[months_sorted[current_index]] = None

        start += increment
        current_index += 1

    return rolling_returns_weekly


def handle_market_returns_weekly(
    stock: dict, market_return_details: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """
    Updates the weekly market returns data by adding the returns of a given stock
    to the existing totals for each week.

    This function accumulates the market returns for each week in the `market_return_details`
    dictionary, where the returns are summed, and the count is incremented based on the week.
    The function helps in tracking the weekly market returns for all stocks, which can later
    be averaged across all stocks.

    Parameters
    ----------
    stock : dict
        A dictionary representing a stock, containing:
            - ["subfeatures"]["returns_weekly"]: A dictionary where keys are weeks
            (in "YYYY-WW" format) and values are the market returns for the respective week.
    market_return_details : dict
        A dictionary that stores the accumulated market returns, where keys are weeks (in "YYYY-WW" format)
        and values are another dictionary containing:
            - 'sum': The total market return for that week.
            - 'count': The number of stocks that have contributed to the weekly return.

    Returns
    -------
    dict
        A dictionary that stores the accumulated market returns, where keys are weeks (in "YYYY-WW" format)
        and values are another dictionary containing:
            - 'sum': Updated total market return for that week.
            - 'count': Updated number of stocks that have contributed to the weekly return.
    """
    returns_weekly = stock["subfeatures"]["weekly"]["returns_weekly"].items()
    for week, returns in returns_weekly:
        if week not in market_return_details:
            market_return_details[week] = {"sum": 0, "count": 0}

        # Add the value if returns are not None
        if returns is not None:
            market_return_details[week]["sum"] += returns
            market_return_details[week]["count"] += 1

    return market_return_details


def invalidate_weeks_by_valid_months(
    weekly_returns_df, month_to_weeks, symbol_to_valid_months
):
    """
    Invalidate weekly returns for each symbol in weekly_returns_df based on whether the week
    belongs to a valid month for that symbol.

    Parameters:
    - weekly_returns_df: pd.DataFrame
        Rows: week keys in "YYYY-WW" format
        Columns: stock symbols
        Values: weekly returns

    - month_to_weeks: dict
        Format: { "YYYY-MM": ["YYYY-WW", ...] }

    - symbol_to_valid_months: dict
        Format: { "SYMBOL": ["YYYY-MM", ...] }

    Returns:
    - pd.DataFrame: same shape as weekly_returns_df but with invalid weeks set to NaN
    """
    # Build reverse mapping: symbol -> set of valid weeks
    symbol_to_valid_weeks = {}
    for symbol, valid_months in symbol_to_valid_months.items():
        valid_weeks = set()
        for month in valid_months:
            valid_weeks.update(month_to_weeks.get(month, []))
        symbol_to_valid_weeks[symbol] = valid_weeks

    # Create a copy to avoid modifying original
    result_df = weekly_returns_df.copy()

    # Invalidate weeks
    for symbol in result_df.columns:
        if symbol not in symbol_to_valid_weeks:
            # Invalidate all weeks if no valid months
            result_df[symbol] = np.nan
        else:
            valid_weeks = symbol_to_valid_weeks[symbol]
            mask = ~result_df.index.isin(valid_weeks)
            result_df.loc[mask, symbol] = np.nan

    return result_df


def month_to_weeks_mapper(
    first_date: str = "1962-01-01", last_date: str = "2025-05-31"
) -> dict[str, set]:
    """
    Maps each month in the given date range to the set of ISO weeks that overlap with it.

    Parameters
    ----------
    first_date : str, optional
        The start date in "YYYY-MM-DD" format (default is "1962-01-01").
    last_date : str, optional
        The end date in "YYYY-MM-DD" format (default is "2025-05-31").

    Returns
    -------
    dict[str, set]
        A dictionary where each key is a month in "YYYY-MM" format and the value is a set of
        week identifiers in "YYYY-WW" ISO week format that overlap with that month.
    """
    all_dates = pd.date_range(start=first_date, end=last_date, freq="D")

    month_to_weeks = {}

    for date in all_dates:
        month = f"{date.year}-{date.month:02d}"
        year, week_number, _ = date.isocalendar()
        week_key = f"{year}-{week_number:02d}"  # Week string in "YYYY-WW" format

        if month not in month_to_weeks:
            month_to_weeks[month] = set()

        month_to_weeks[month].add(week_key)
    month_to_weeks
