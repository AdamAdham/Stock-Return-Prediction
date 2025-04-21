from src.feature_engineering.utils import calculate_return

from src.config.settings import RETURN_ROUND_TO


def calculate_momentum(months_sorted, prices_monthly, offset_start, offset_end):
    """
    Calculates the momentum (rate of return) for each month over a custom time window.

    This function computes the momentum for each month by calculating the percentage change
    between the starting and ending months_sorted within a specified window (from `offset_start`
    to `offset_end` relative to the current month).

    Parameters
    ----------
    months_sorted : list of str
        A sorted list of months_sorted in "YYYY-MM" format, from latest to oldest.
    prices_monthly : dict
        A dictionary where keys are month identifiers (in "YYYY-MM" format) and values
        are the closing prices for those months_sorted.
    offset_start : int
        The number of months_sorted before the current month to start the momentum calculation.
    offset_end : int
        The number of months_sorted after the current month to end the momentum calculation.
        For example, for a 12-month momentum: `offset_start=1`, `offset_end=12`.

    Returns
    -------
    dict
        A dictionary where the keys are month identifiers (in "YYYY-MM" format), and the values
        are the calculated momentum values (rate of return) for each month. If there is insufficient
        data to calculate the momentum for a particular month, the value will be `None`.

    Notes
    -----
    - The function assumes that `prices_monthly` contains the necessary data for each month in `months_sorted`.
    - The momentum is calculated as the percentage change in closing prices between the months_sorted defined by `offset_start` and `offset_end`.
    - If there are not enough months_sorted in the list to calculate the momentum, the function will return `None` for those months_sorted.
    """
    mom = {}

    for i in range(len(months_sorted) - offset_end):
        curr_month = months_sorted[i]
        month_start = months_sorted[i + offset_start]
        month_end = months_sorted[i + offset_end]
        mom[curr_month] = calculate_return(
            prices_monthly[month_start],
            prices_monthly[month_end],
            round_to=RETURN_ROUND_TO,
        )

    # Fill in None for remaining months_sorted
    for i in range(len(months_sorted) - offset_end, len(months_sorted)):
        if len(months_sorted) - offset_end >= 0:
            mom[months_sorted[i]] = None

    return mom


def calculate_mom1m(months_sorted, prices_monthly):
    # Same as returns_monthly
    return calculate_momentum(months_sorted, prices_monthly, 0, 1)


def calculate_mom12m(months_sorted, prices_monthly, current=False):
    if current:
        # 11 not 12 such that returns are calculated from 2023-09 to 2022-10 (both inclusive) so it will be 12 months since current is included
        return calculate_momentum(months_sorted, prices_monthly, 0, 11)
    else:
        return calculate_momentum(months_sorted, prices_monthly, 1, 12)


def calculate_mom36m(months_sorted, prices_monthly):
    return calculate_momentum(months_sorted, prices_monthly, 13, 36)


def calculate_chmom(months_sorted, prices_monthly, current=False):
    """
    Calculates the change in 6-month momentum (CHMOM) for each month.

    CHMOM is defined as the difference between:
        - The 6-month momentum from t-6 to t-1 (or t-6 to t if `current=True`)
        - The 6-month momentum from t-12 to t-7

    The function loops over months in `months_sorted` (from latest to oldest) and for each,
    computes the change in momentum using the provided monthly closing prices.

    Parameters
    ----------
    months_sorted : list of str
        A list of months in "YYYY-MM" format, sorted from latest to oldest.
    prices_monthly : dict
        A dictionary mapping each month ("YYYY-MM") to its closing price.
    current : bool, optional
        - If True, includes the current month's price in the first return window (t to t-6).
        - If False (default), uses the month before the current one (t-1 to t-6).

    Returns
    -------
    dict
        A dictionary where:
            - key (str): A month in "YYYY-MM" format.
            - value (float or None): The computed change in 6-month momentum for that month.
              Returns None if prices are missing or division by zero occurs.
    """

    chmom_per_month = {}

    for i in range(len(months_sorted) - 12):  # Ensure at least 12 months_sorted of data
        curr_month = months_sorted[i]

        # Start from current month or next month
        if current:
            start = curr_month
        else:
            start = months_sorted[i + 1]  # Previous month

        t_6 = months_sorted[i + 6]  # Month t-6
        t_7 = months_sorted[i + 7]  # Month t-7
        t_12 = months_sorted[i + 12]  # Month t-12

        if (
            prices_monthly[t_6] == 0 or prices_monthly[t_12] == 0
        ):  # Prevent division by zero
            chmom_per_month[curr_month] = None
        else:
            mom1_6 = calculate_return(
                prices_monthly[start],
                prices_monthly[t_6],
                round_to=RETURN_ROUND_TO,
            )
            mom7_12 = calculate_return(
                prices_monthly[t_7],
                prices_monthly[t_12],
                round_to=RETURN_ROUND_TO,
            )

            chmom_per_month[curr_month] = mom1_6 - mom7_12

    # Make all months_sorted that cannot be calculated to None
    for i in range(len(months_sorted) - 12, len(months_sorted)):
        curr_month = months_sorted[i]
        chmom_per_month[curr_month] = None

    return chmom_per_month


def calculate_maxret(months_sorted, max_daily_returns_monthly):
    """
    Calculates the maximum daily return (maxret) for each month in the given list.

    For each month in the `months_sorted` list (except the last), this function retrieves the maximum
    daily return from the previous month (i.e., the next month in the list), and assigns it
    to the current month.

    Parameters
    ----------
    months_sorted : list of str
        List of months_sorted in "YYYY-MM" format, ordered from most recent to oldest.

    max_daily_returns_monthly : dict
        A dictionary mapping "YYYY-MM" month strings to their corresponding maximum daily return values.

    Returns
    -------
    dict
        A dictionary where each key is a month from `months_sorted`, and the value is the maximum daily return
        from the following month in the list.

    Notes
    -----
    - This function assumes that `months_sorted` is ordered in descending order (latest month first).
    - Sometimes not only the last month is set to None it is the last 2, since if there is only 1
    price in the last month so cannot have returns for that month
    """

    maxret_monthly = {}
    for i in range(len(months_sorted) - 1):
        month = months_sorted[i]  # Get current month
        maxret = max_daily_returns_monthly[
            months_sorted[i + 1]
        ]  # Get the max returns of the previous month

        if month not in maxret_monthly:
            # Store the natural log of the market cap
            maxret_monthly[month] = maxret

    maxret_monthly[months_sorted[-1]] = None  # Assign None to remaining month

    return maxret_monthly


def handle_indmom(stock, indmom):
    """
    Updates the industry momentum (indmom) data for a given stock by accumulating
    (mom12m) values for each month within its industry group.

    This function iterates over the `mom12m` values of a stock, which represent the
    12-month momentum for the stock, and updates the `indmom` dictionary by adding
    the values to the corresponding industry's monthly totals. The industry is identified
    by the stock's SIC code (2-digit code).

    Parameters
    ----------
    stock : dict
        A dictionary representing a stock, containing:
            - 'mom12m': A dictionary where keys are months_sorted (in "YYYY-MM" format) and
              values are momentum values for the respective month.
            - 'sicCode_2': A string representing the 2-digit SIC code of the stock,
              used to categorize the industry group.
    indmom : dict
        A dictionary representing the industry momentum, where keys are SIC codes and
        values are another dictionary with months_sorted as keys and dictionaries containing
        'total' (sum of momentum values) and 'count' (number of stocks contributing to that month).

    Returns
    -------
    dict
        A dictionary representing the industry momentum, where keys are SIC codes and
        values are another dictionary with months_sorted as keys and dictionaries containing
        'total' (sum of momentum values) and 'count' (number of stocks contributing to that month).
    """
    mom12m = stock["features"]["monthly"]["mom12m"]
    sic = stock["sicCode_2"]
    # For every stock and every month, the mom12m value is summed to get the average across the whole industry major group which have a common first 2 digits of the SIC code
    for month, value in mom12m.items():
        if value is None:
            continue
        if month not in indmom[sic]:
            indmom[sic][month] = {"total": value, "count": 1}
        else:
            indmom[sic][month]["total"] += value
            indmom[sic][month]["count"] += 1

    return indmom
