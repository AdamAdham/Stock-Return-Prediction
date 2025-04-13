from datetime import datetime
from src.config.settings import RETURN_ROUND_TO


def calculate_return(price_later, price_earlier, round_to=None):
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


def get_monthly_price(prices_daily):
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
        - months (list of str): A list of unique months (formatted as "YYYY-MM") found in the dataset and acts as index to prices_monthly (returned in the same order of prices_daily dates).
        - prices_monthly (dict): A dictionary mapping each month ("YYYY-MM") to its latest available price.

    Example
    -------
    >>> prices_daily = [
    ...     {"date": "2025-03-29", "price": 150.5},
    ...     {"date": "2025-03-30", "price": 152.0},
    ...     {"date": "2025-04-01", "price": 155.0}
    ... ]
    >>> get_monthly_price(prices_daily)
    (["2025-03", "2025-04"], {"2025-03": 150.5, "2025-04": 155.0})

    Notes
    -----
    - If the "date" field is a string, it is converted into a datetime object.
    - The function assumes that the daily prices is sorted latest to earliest dates.
    """

    prices_monthly = {}
    months = []
    seen = set()

    # Get latest price of each month
    for entry in prices_daily:
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"

        if month not in seen:
            seen.add(month)
            months.append(month)
            prices_monthly[month] = entry["price"]

    return months, prices_monthly


# Liquidity Variables


def get_dollar_volume_monthly(prices_daily):
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

    dollar_volume_by_month = {}

    for entry in prices_daily:
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"

        dollar_volume = entry["price"] * entry["volume"]  # Calculate value

        # Check if first occurence of current month
        if month not in dollar_volume_by_month:
            dollar_volume_by_month[month] = {"sum": dollar_volume, "count": 1}
        else:
            dollar_volume_by_month[month]["sum"] += dollar_volume
            dollar_volume_by_month[month]["count"] += 1

    return dollar_volume_by_month


def get_volume_shares_statistics(prices_daily, shares_daily):
    """
    Calculate monthly total trading volume, last trading day's outstanding shares,
    number of trading days, number of zero trading days, and daily turnover.

    This function processes daily stock trading data and extracts the total
    volume traded per month, the number of outstanding shares for each month (taken from the last
    trading day of the month), counts the number of zero trading days, counts the total number of
    trading days for each month, and computes the daily turnover.

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries containing daily stock trading data with keys:
        - "date" (str): Date in the format "YYYY-MM-DD".
        - "volume" (int): The number of shares traded on that day.

    shares_daily : dict
        Key is "date" (str): Date in the format "YYYY-MM-DD" and value is outstanding shares of that day.

    Returns
    -------
    tuple
        A tuple containing five dictionaries:
        - vol_sum (dict): A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
          and values are the total volume of shares traded during that month.
        - shares_monthly (dict): A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
          and values are the number of outstanding shares at the last trading day of that month.
        - zero_trading_days_count (dict): A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
          and values are the number of days with zero trading volume in that month.
        - trading_days_count (dict): A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
          and values are the number of trading days in that month.
        - daily_turnover (dict): A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
          and values are dictionaries where each key is a day identifier (str) in the format "YYYY-MM-DD"
          and the value is the daily turnover for that day.
    """

    vol_sum = {}
    shares_monthly = {}
    zero_trading_days_count = {}
    trading_days_count = {}
    daily_turnover = {}

    vol_daily = {}

    # Get sum of daily volume traded in each month, count zero trading days, count total trading days, and calculate daily turnover
    for entry in prices_daily:
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"
        day = f"{date.year}-{date.month:02d}-{date.day:02d}"

        volume = entry["volume"]

        # Sum the daily volume for the month
        vol_sum[month] = vol_sum.get(month, 0) + volume

        # Count total trading days
        trading_days_count[month] = trading_days_count.get(month, 0) + 1

        # Initialize the month
        if month not in zero_trading_days_count:
            zero_trading_days_count[month] = 0

        # Count zero trading days
        if volume == 0:
            zero_trading_days_count[month] += 1

        # Create a daily volume lookup to use to calculate the daily turnover when loop through the shares
        vol_daily[day] = volume

    # Get the number of outstanding shares of each month by getting the shares data at the last trading day
    # Get the daily turnover
    shares_keys = sorted(
        shares_daily.keys(), reverse=True
    )  # Dates sorted from latest to earliest
    for key_date in shares_keys:
        date = datetime.strptime(key_date, "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"
        day = f"{date.year}-{date.month:02d}-{date.day:02d}"

        shares = shares_daily[key_date]

        if month not in shares_monthly:
            shares_monthly[month] = shares

        if day in vol_daily:
            daily_turnover[day] = vol_daily[day] / shares if shares else None
        else:
            daily_turnover[day] = None
            print(f"Shares data available but no matching volume data on {day}")

    return (
        vol_sum,
        shares_monthly,
        zero_trading_days_count,
        trading_days_count,
        daily_turnover,
    )


def get_market_cap_monthly(market_caps):
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


def get_stock_returns_weekly(prices_daily):
    """
    Calculate the weekly stock returns based on the closing prices of the
    last trading day of each week.

    The return for each week is calculated as the percentage change in
    the closing price from the previous week's last trading day to the
    current week's last trading day.

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries containing daily stock trading data with keys:
        - "date" (str): The date in the format "YYYY-MM-DD".
        - "price" (float): The closing price of the stock on that date.

    Returns
    ---------
    tuple
        A tuple containing two dictionaries:
        - month_latest_week (dict): A dictionary where keys are month identifiers
          (in the format "YYYY-MM") and values are the last trading week of that month
          (represented as a tuple of year and week number).
        - weekly_returns (dict): A dictionary where keys are tuples representing the
          year and week number (e.g., (2022, 1) for the first week of 2022), and values
          are the calculated weekly returns (float), based on the percentage change
          between the closing prices of consecutive weeks.
    """

    # Group by calendar week (year, week number)
    weekly_prices = {}
    month_latest_week = {}

    # Get the closing price of each week
    for entry in prices_daily:
        date = datetime.strptime(entry["date"], "%Y-%m-%d")
        month = f"{date.year}-{date.month:02d}"
        year, week_number, _ = date.isocalendar()
        week_key = f"{year}-{week_number:02d}"  # (year, week_number), str due to not being able to dump json

        if week_key not in weekly_prices:
            weekly_prices[week_key] = entry["price"]

        if month not in month_latest_week:
            month_latest_week[month] = week_key

    # Compute weekly returns
    weekly_returns = {}
    weeks = list(sorted(weekly_prices.keys(), reverse=True))

    for i in range(len(weeks) - 1):
        price_current = weekly_prices[weeks[i]]
        price_previous = weekly_prices[weeks[i + 1]]

        weekly_returns[weeks[i]] = calculate_return(
            price_current, price_previous, round_to=RETURN_ROUND_TO
        )

    # Assign None to remaining week
    weekly_returns[weeks[-1]] = None

    return month_latest_week, weekly_returns


def get_previous_week(year, week, keys):
    """
    Get the previous calendar week, properly handling year transitions.

    This function computes the previous week number for a given (year, week) pair.
    If the current week is the first of the year, it rolls back to the last valid week
    of the previous year (either week 52 or 53, depending on the calendar).

    Parameters
    year (int)
        The current year.
    week (int)
        The current ISO calendar week number.
    keys (set or list of tuple)
        A collection of (year, week) tuples to determine if week 53 exists for a given year.

    Returns
    tuple
        A tuple (year, week) representing the previous week.
    """
    week -= 1

    # First week in the year, so year has finished, so decrement the year and initialize week to 53 if present 52 otherwise
    if week <= 0:
        year -= 1
        week = 53 if (year, 53) in keys else 52
    return year, week


def get_rolling_weekly_returns(
    months, month_latest_week, weekly_returns, interval=156, increment=4
):
    """
    Compute rolling average of weekly returns for each month over a specified interval.

    This function calculates the average of weekly stock returns over a fixed interval
    (default 156 weeks, approximately 3 years) for each month in the `months` list.
    The calculation starts from the last trading week prior to each month and rolls backward.
    The function moves forward by a set increment (default 4 weeks) for each new month.

    Parameters
    ----------
    months : list of str
        A list of month identifiers in the format "YYYY-MM", sorted in descending order (most recent first).

    month_latest_week : dict
        A dictionary mapping each month ("YYYY-MM") to the latest trading week prior to the month's end.
        Each value is a tuple of (year, week_number).

    weekly_returns : dict
        A dictionary where keys are tuples (year, week_number) and values are the weekly return (float)
        for that week.

    interval : int, optional
        The number of weeks to include in the rolling average (default is 156 weeks).

    increment : int, optional
        The number of weeks to move forward in each step (default is 4 weeks, roughly one month).

    Returns
    -------
    dict
        A dictionary where keys are month identifiers ("YYYY-MM") and values are the rolling average
        of weekly returns over the specified interval. If insufficient data exists to compute the
        average for a given month, the value will be None.
    """

    rolling_weekly_returns = {}

    sorted_keys = sorted(weekly_returns.keys(), reverse=True)  # sorts by (year, week)
    weekly_returns_list = [weekly_returns[k] for k in sorted_keys]

    current = 0
    month_start = months[1]
    week_start = month_latest_week[month_start]

    # Get index of the latest week of the month to start which is the previous one since in paper states "prior to month end."
    start = sorted_keys.index(week_start)

    while current < len(months):

        if start + interval < len(weekly_returns_list):
            # Rolling average of the 156 weekly returns
            rolling_weekly_returns[months[current]] = (
                sum(weekly_returns_list[start : start + interval]) / interval
            )
        else:
            # Not enough data
            rolling_weekly_returns[months[current]] = None

        start += increment
        current += 1

    return rolling_weekly_returns


# Differnce is the window starts from current month
# month_start = months[current]
def get_rolling_weekly_returns_current(
    months, month_latest_week, weekly_returns, interval=156, increment=4
):
    """
    Compute rolling average of weekly returns for each month over a specified interval.

    This function calculates the average of weekly stock returns over a fixed interval
    (default 156 weeks, approximately 3 years) for each month in the `months` list.
    The calculation starts from the last trading week prior to each month and rolls backward.
    The function moves forward by a set increment (default 4 weeks) for each new month.

    Parameters
    ----------
    months : list of str
        A list of month identifiers in the format "YYYY-MM", sorted in descending order (most recent first).

    month_latest_week : dict
        A dictionary mapping each month ("YYYY-MM") to the latest trading week prior to the month's end.
        Each value is a tuple of (year, week_number).

    weekly_returns : dict
        A dictionary where keys are tuples (year, week_number) and values are the weekly return (float)
        for that week.

    interval : int, optional
        The number of weeks to include in the rolling average (default is 156 weeks).

    increment : int, optional
        The number of weeks to move forward in each step (default is 4 weeks, roughly one month).

    Returns
    -------
    dict
        A dictionary where keys are month identifiers ("YYYY-MM") and values are the rolling average
        of weekly returns over the specified interval. If insufficient data exists to compute the
        average for a given month, the value will be None.
    """

    rolling_weekly_returns = {}

    sorted_keys = sorted(weekly_returns.keys(), reverse=True)  # sorts by (year, week)
    weekly_returns_list = [weekly_returns[k] for k in sorted_keys]

    current = 0
    month_start = months[current]
    week_start = month_latest_week[month_start]

    # Get index of the latest week of the month to start which is the previous one since in paper states "prior to month end."
    start = sorted_keys.index(week_start)

    while current < len(months):

        if start + interval < len(weekly_returns_list):
            # Rolling average of the 156 weekly returns
            rolling_weekly_returns[months[current]] = (
                sum(weekly_returns_list[start : start + interval]) / interval
            )
        else:
            # Not enough data
            rolling_weekly_returns[months[current]] = None

        start += increment
        current += 1

    return rolling_weekly_returns


def get_rolling_market_returns(stocks):
    """
    Calculate the monthly rolling market return as the average of individual stock-level
    3-year rolling weekly returns across all stocks.

    For each month, this function aggregates the 3-year rolling average of weekly returns
    (typically 156 weeks) from all stocks and computes the mean value to represent the
    market-level return for that month.

    Parameters
    stocks
        A list of dictionaries, each representing a stock. Each dictionary must contain a key
        "rolling_avg_3y_weekly_returns_by_month", which maps month strings (in "YYYY-MM" format)
        to the rolling average of weekly returns for that stock.

    Returns
    dict
        A dictionary where keys are months (str in "YYYY-MM" format) and values are the
        average of 3-year rolling weekly returns across all stocks for that month.
    """

    rolling_market_returns = {}

    for stock in stocks:
        rolling_avg_3y_weekly_returns_by_month = stock["subfeatures"]["monthly"][
            "rolling_avg_3y_weekly_returns_by_month"
        ]

        # For every stock and every month, the rolling_avg_3y_weekly_returns_by_month value is summed to get the average across the whole market
        for month, value in rolling_avg_3y_weekly_returns_by_month.items():

            if month not in rolling_market_returns:
                rolling_market_returns[month] = {"sum": value, "count": 1}
            else:
                rolling_market_returns[month]["sum"] += value
                rolling_market_returns[month]["count"] += 1

    # Convert sum/count into average
    for month in rolling_market_returns:
        total = rolling_market_returns[month]["sum"]
        count = rolling_market_returns[month]["count"]
        rolling_market_returns[month] = total / count

    return rolling_market_returns


def get_market_returns_weekly(stocks):
    """
    Calculate the average weekly market returns across multiple stocks.

    This function computes the average return for each week by aggregating
    the weekly returns of all stocks and then calculating the mean return
    for each week.

    Parameters
    ----------
    stocks : list of dict
        A list of dictionaries where each dictionary represents a stock's
        data. Each stock dictionary should contain:
        - "returnsWeekly" (dict): A dictionary where keys are week identifiers
          (e.g., (year, week number)) and values are the return for that week
          (float).

    Returns
    -------
    dict
        A dictionary where keys are week identifiers (e.g., (2022, 1)) and values
        are the average weekly return for that week across all stocks (float).
    """
    market_return_details = {}
    for stock in stocks:
        for week, returns in stock["weekly_returns"].items():
            if week not in market_return_details:
                market_return_details[week] = {"sum": returns, "count": 1}
            else:
                market_return_details[week]["sum"] += returns
                market_return_details[week]["count"] += 1

    market_returns = {}
    for week, returns in market_return_details.items():
        market_returns[week] = returns["sum"] / returns["count"]

    return market_returns


def handle_market_returns_weekly(stock, market_return_details):
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
            - ["subfeatures"]["weekly_returns"]: A dictionary where keys are weeks
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
    weekly_returns = stock["subfeatures"]["weekly"]["weekly_returns"].items()
    for week, returns in weekly_returns:
        if returns is None:
            continue
        if week not in market_return_details:
            market_return_details[week] = {"sum": returns, "count": 1}
        else:
            market_return_details[week]["sum"] += returns
            market_return_details[week]["count"] += 1
    return market_return_details
