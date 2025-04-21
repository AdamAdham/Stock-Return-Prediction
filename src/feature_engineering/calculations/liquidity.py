from datetime import datetime
from collections import defaultdict
import numpy as np

from src.feature_engineering.utils import calculate_return

from src.config.settings import RETURN_ROUND_TO


def calculate_turn(months_sorted, vol_monthly, shares_monthly):
    """
    Calculate the 3-month rolling share turnover ratio for each month.

    Share turnover is defined as the average trading volume over a period
    divided by the number of outstanding shares. This function computes
    the turnover over a 3-month rolling window, starting from each month in `months_sorted`.

    Parameters
    ----------
    months_sorted : list of str
        A list of month identifiers in the format "YYYY-MM", ordered from the most recent to the oldest.

    vol_monthly : dict
        A dictionary where keys are month identifiers (str) in the format "YYYY-MM" and values are the
        total trading volume for each respective month.

    shares_monthly : dict
        A dictionary where keys are month identifiers (str) in the format "YYYY-MM" and values are the
        number of outstanding shares at the last trading day of each respective month.

    Returns
    -------
    dict
        A dictionary where keys are month identifiers (str) and values are the 3-month rolling share turnover
        ratio (float) for each month. If the number of outstanding shares for a month is zero, the result will be
        `None` to avoid division by zero.
    """

    # Calculate the share turnover for each month
    result = {}
    for i in range(len(months_sorted) - 2):
        avg_volume = (
            vol_monthly[months_sorted[i]]
            + vol_monthly[months_sorted[i + 1]]
            + vol_monthly[months_sorted[i + 2]]
        ) / 3
        share = shares_monthly[months_sorted[i]]
        result[months_sorted[i]] = (
            avg_volume / share if share != 0 else None
        )  # Prevent division by zero

    # Assign None to remaining months_sorted
    result[months_sorted[-2]] = None
    result[months_sorted[-1]] = None
    return result


def calculate_std_turn(prices_daily, shares):
    """
    Calculate the monthly standard deviation of daily share turnover (turnover volatility).

    Share turnover is defined as the ratio of daily trading volume to the number
    of outstanding shares on that day. This function computes the standard deviation
    of this value for each month.

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries containing daily trading data with keys:
        - "date" (str): Date in the format "YYYY-MM-DD".
        - "volume" (int): Number of shares traded on that day.

    shares : dict
        Key is "date" (str): Date in the format "YYYY-MM-DD" and value is outstanding shares of that day.

    Returns
    -------
    dict
        A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
        and values are the standard deviation (float) of daily share turnover for that month.
        If no valid turnover data is available for a month, the value will be `None`.
    """
    # Collect daily turnover per month
    monthly_turns = defaultdict(list)

    for entry in prices_daily:
        date_str = entry["date"]
        volume = entry["volume"]
        share = shares.get(date_str, None)

        if share is not None and share != 0:
            turnover = volume / share
            date = datetime.strptime(date_str, "%Y-%m-%d")
            month = f"{date.year}-{date.month:02d}"

            monthly_turns[month].append(turnover)

    # Calculate standard deviation for each month
    std_turn_per_month = {}
    for month, turns in monthly_turns.items():
        if len(turns) > 1:
            std_turn_per_month[month] = float(np.std(turns, ddof=1))  # Sample std dev
        else:
            std_turn_per_month[month] = None  # Not enough data to compute std dev

    return std_turn_per_month


def calculate_mve(months_sorted, market_caps_monthly, current=False):
    """
    Calculate the monthly Market Value of Equity (mve) using the natural logarithm
    of the market capitalization from the previous month.

    For each month in the input list, this function retrieves the market capitalization
    of the prior month and computes its natural logarithm as the mve.

    Parameters
    ----------
    months_sorted : list of str
        A list of month identifiers in the format "YYYY-MM", ordered from most recent to oldest.
    market_caps_monthly : dict
        A dictionary where keys are month identifiers ("YYYY-MM") and values are
        the market capitalization (float or int) for that month.
    current : bool, optional
        - If True, uses current month's last trading days market cap.
        - If False (default), uses previous month's last trading days market cap.

    Returns
    -------
    dict
        A dictionary where keys are month identifiers (str) in the format "YYYY-MM",
        and values are the natural logarithm of the market capitalization (float)
        from the previous month.
    """

    if current == True:
        return {
            month: np.log(market_caps_monthly[month]) for month in market_caps_monthly
        }

    mve_per_month = {}
    for i in range(len(months_sorted) - 1):
        month = months_sorted[i]  # Get current month
        market_cap = market_caps_monthly[
            months_sorted[i + 1]
        ]  # Get the market cap of the previous month

        if month not in mve_per_month:
            # Store the natural log of the market cap
            mve_per_month[month] = np.log(market_cap)

    # Assign None to remaining months_sorted
    mve_per_month[months_sorted[-1]] = None

    return mve_per_month


def calculate_dolvol(months_sorted, dollar_volume_monthly, current=False):
    """
    Calculates the Dollar Volume (dolvol) factor for each month using pre-aggregated monthly dollar volume data.

    The dolvol factor is defined as the natural logarithm of the average daily dollar volume
    (price x volume) from two months_sorted prior to the current month.

    Parameters
    ----------
    months_sorted : list of str
        A list of month identifiers in "YYYY-MM" format, sorted from latest to oldest.
    dollar_volume_monthly : dict
        A dictionary where each key is a month ("YYYY-MM") and each value is another dictionary with:
            - "sum" (float): Total dollar volume for the month.
            - "count" (int): Number of trading days in that month.
    current : bool, optional
        - If True, uses current month's last trading days market cap.
        - If False (default), uses previous month's last trading days market cap.
    Returns
    -------
    dict
        A dictionary where each key is a month ("YYYY-MM") and the value is the calculated
        dolvol (log of average daily dollar volume from two months_sorted earlier). If data is
        insufficient, the value is set to None.

    Notes
    -----
    - Assumes `months_sorted` is sorted from most recent to oldest.
    - Requires at least two months_sorted of lookback data to compute dolvol for a given month.
    """

    if current:
        # For each month return the log of the average
        # No need for None handling since dollar_volume_monthly[month] cannot be None due to implementation at src.feature_engineering.utils.get_weekly_monthly_summary
        return {
            month: (
                np.log(
                    dollar_volume_monthly[month]["sum"]
                    / dollar_volume_monthly[month]["count"]
                )
                if dollar_volume_monthly[month]["sum"] != 0
                else None
            )  # Can be that the only volumes present are 0
            for month in months_sorted
        }

    # For each month t get the sum and count for month t+2 which is the 2 months_sorted prior current month t
    dolvol_monthly = {}
    for i in range(len(months_sorted) - 2):
        curr_month = months_sorted[i]
        month_2 = months_sorted[i + 2]
        avg_dv = (
            dollar_volume_monthly[month_2]["sum"]
            / dollar_volume_monthly[month_2]["count"]
        )
        dolvol_monthly[curr_month] = (
            np.log(avg_dv) if avg_dv != 0 else None
        )  # Can be that the only volumes present are 0

    # Make all months_sorted that cannot be calculated to None
    dolvol_monthly[months_sorted[-2]] = None
    dolvol_monthly[months_sorted[-1]] = None

    return dolvol_monthly


def calculate_ill(prices_daily):
    """
    Calculates the Amihud illiquidity measure for each month based on daily price and volume data.

    The Amihud illiquidity measure is defined as the average ratio of the absolute daily return
    to the daily dollar trading volume.

    Parameters
    ----------
    prices_daily : list of dict
        A list of dictionaries, each representing a day of trading with the following keys:
            - "date" (str): The date in "YYYY-MM-DD" format.
            - "price" (float): The closing price for the day.
            - "volume" (float): The trading volume for the day.

    Returns
    -------
    dict
        A dictionary where each key is a month in "YYYY-MM" format and the value is the
        average Amihud illiquidity for that month. If no valid data is available for a
        month (e.g., zero volume every day), the value is set to None.

    Notes
    -----
    - Skips days with zero dollar volume to avoid division by zero.
    - Uses price from day i and day i+1 to calculate daily return, so the last day in the list is ignored.
    """

    ill_monthly = {}
    prev_month = None
    ill_sum = 0
    ill_count = 0

    for i in range(len(prices_daily) - 1):
        today = prices_daily[i]
        yesterday = prices_daily[i + 1]
        date_obj = datetime.strptime(today["date"], "%Y-%m-%d")
        current_month = f"{date_obj.year}-{date_obj.month:02d}"

        # Calculate absolute return
        abs_ret = abs(
            calculate_return(
                today["price"],
                yesterday["price"],
                round_to=RETURN_ROUND_TO,
            )
        )

        dollar_volume = today["price"] * today["volume"]

        if dollar_volume == 0:
            continue  # Avoid division by zero

        illiquidity = abs_ret / dollar_volume

        # If first month or still in the same month
        if prev_month is None or current_month == prev_month:
            ill_sum += illiquidity
            ill_count += 1
        else:
            # Store previous month’s average
            ill_monthly[prev_month] = ill_sum / ill_count if ill_count != 0 else None
            # Reset counters for the new month
            ill_sum = illiquidity
            ill_count = 1

        prev_month = current_month

    # Handle last month’s average
    if prev_month is not None:
        ill_monthly[prev_month] = (
            ill_sum / ill_count if ill_count > 0 else None
        )  # Prevent division by zero

    # Handle last entry in the list separately to handle the case where it's the only entry in the month
    last_entry = prices_daily[-1]
    date_last = datetime.strptime(last_entry["date"], "%Y-%m-%d")
    month_last = f"{date_last.year}-{date_last.month:02d}"

    # If the last entry is the only one in the month, set its ill to none since the month has no returns since no previous day
    if month_last not in ill_monthly:
        ill_monthly[month_last] = None

    return ill_monthly


def calculate_zerotrade(
    months_sorted,
    vol_sum_monthly,
    shares_monthly,
    zero_trading_days,
    trading_days_count,
):
    """
    Calculate the turnover-adjusted number of zero trading days for each month.

    This function computes a measure called "zerotrade" for each month in the input list,
    excluding the last month. The measure adjusts the count of zero trading days based
    on the turnover (volume / shares) for that month and normalizes by the typical number
    of trading days in a month (assumed to be 21).

    "Deflator" is not used as in the original equation

    The formula used is:
        - If turnover is zero:       zerotrade = zero_days
        - If turnover is non-zero:   zerotrade = (zero_days + 1 / turnover) * (21 / trading_days)

    This metric is typically used to assess stock liquidity or trading inactivity.

    Parameters
    ----------
    months_sorted : list of str
        An ordered list of months_sorted (e.g., ["2024-01", "2024-02", ...]).
    vol_sum_monthly : dict
        Dictionary mapping each month (str) to the total trading volume.
    shares_monthly : dict
        Dictionary mapping each month (str) to the number of outstanding shares at month-end.
    zero_trading_days : dict
        Dictionary mapping each month (str) to the number of days with zero trading volume.
    trading_days_count : dict
        Dictionary mapping each month (str) to the total number of trading days.

    Returns
    -------
    dict
        A dictionary mapping each month (excluding the last one) to its zerotrade value.
    """

    zerotrade = {}

    for i in range(len(months_sorted) - 1):
        current_month = months_sorted[i]
        zero_days = zero_trading_days[current_month]

        # Get turnover in the prior month
        vol = vol_sum_monthly[current_month]
        shares = shares_monthly[current_month]
        turnover = vol / shares if shares != 0 else None  # Prevent division by zero

        # Get trading days in that month
        trading_days = trading_days_count[current_month]

        # Compute zerotrade
        if turnover == 0:
            lm = zero_days  # no turnover => skip adjustment term
        else:
            lm = (zero_days + (1 / turnover)) * (21 / trading_days)

        zerotrade[current_month] = lm

    zerotrade[months_sorted[-1]] = None  # Assign None to remaining month
    return zerotrade
