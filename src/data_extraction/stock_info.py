import json
from datetime import datetime, timedelta
import time
import gc

from src.data_extraction.api_client import APIClient
from src.data_extraction.utils import remove_duplicates_and_sort_by_date
from src.utils.disk_io import write_json, load_all_stocks
from src.config.settings import RAW_DIR

api_client = APIClient()


def get_stock_info(stock):
    """
    Retrieves and aggregates financial and market data for a given stock.

    This function gathers income statement data, balance sheet data, market capitalization data,
    and end-of-day (EOD) stock price data for a specific stock symbol. The user can specify the interval
    for financial statements (e.g., annual or quarterly) and optional date ranges for market cap and EOD data.

    Parameters
    ----------
    stock : dict
        A dictionary containing at least a 'symbol' key representing the stock ticker.
    interval : str, optional
        Interval for financial data, either 'annual' or 'quarterly'. Default is 'annual'.
    market_cap_start : str, optional
        The start date (YYYY-MM-DD) for retrieving market cap data. Default is None.
    market_cap_end : str, optional
        The end date (YYYY-MM-DD) for retrieving market cap data. Default is None.
    eod_start : str, optional
        The start date (YYYY-MM-DD) for retrieving EOD (end-of-day) price data. Default is None.
    eod_end : str, optional
        The end date (YYYY-MM-DD) for retrieving EOD (end-of-day) price data. Default is None.

    Returns
    -------
    dict
        A copy of the input `stock` dictionary, extended with the following keys:
            - 'financial_<interval>': A dictionary containing:
                - 'income_statement': Financial income statement data.
                - 'balance_sheet': Financial balance sheet data.
            - 'market_cap': Market capitalization data within the specified date range.
            - 'eod': End-of-day stock price data within the specified date range.

    Notes
    -----
    - If no date ranges are provided, full available data for market cap and EOD will be retrieved.
    """
    start_date = "1700-12-12"  # A random old date
    end_date = datetime.today().date()

    symbol = stock["symbol"]
    stock_final = stock.copy()

    # earnings = get_earnings(symbol)
    income_statement_annual = api_client.get_income_statement(symbol, interval="annual")
    balance_sheet_annual = api_client.get_balance_sheet(symbol, interval="annual")
    income_statement_quarterly = api_client.get_income_statement(
        symbol, interval="quarterly"
    )
    balance_sheet_quarterly = api_client.get_balance_sheet(symbol, interval="quarterly")
    eod = api_client.get_eod(symbol, start_date=start_date, end_date=end_date)
    earnings = api_client.get_earnings(symbol)

    stock_final[f"financials_annual"] = {
        "income_statement": income_statement_annual,
        "balance_sheet": balance_sheet_annual,
    }
    stock_final[f"financials_quarterly"] = {
        "income_statement": income_statement_quarterly,
        "balance_sheet": balance_sheet_quarterly,
    }
    stock_final["earnings"] = earnings

    # print(f"{symbol} EOD Duplicates") TODO
    stock_final["eod"] = remove_duplicates_and_sort_by_date(eod)

    # Handling max requests restriction by FMP (market_cap)

    start_date = end_date - timedelta(
        days=19 * 365
    )  # Getting the date 19 years ago (since 19*262<5000) where 262: max weekdays in year, 5000: max number of values in a request (FMP)

    market_cap = api_client.get_market_cap(
        symbol, start_date=start_date + timedelta(days=1), end_date=end_date
    )

    # If market cap data not available, assign market cap and outstanding shares None since outstanding shares dependant on market cap
    if len(market_cap) == 0:
        stock_final["market_cap"] = None
        stock_final["outstanding_shares"] = None
        return stock_final

    window = timedelta(days=10)
    # Loop till all available market caps are retrieved
    while True:
        last_date_in_market_cap = datetime.strptime(
            market_cap[-1]["date"], "%Y-%m-%d"
        ).date()

        # If the earliest date in market_cap is within the window of start_date, request more data. Holidays and weekends did not allow direct equation
        if start_date <= last_date_in_market_cap <= start_date + window:

            # End date to be the earliest date of the previous response + 1 day to prevent dublicates
            end_date = datetime.strptime(
                market_cap[-1]["date"], "%Y-%m-%d"
            ).date() + timedelta(days=1)
            start_date = end_date - timedelta(days=19 * 365)

            market_cap_extended = api_client.get_market_cap(
                symbol, start_date=start_date, end_date=end_date
            )

            if market_cap_extended:  # Ensure there is new data
                market_cap += market_cap_extended
            else:
                break  # Exit loop if no new data is returned
        else:
            break  # Exit loop if there is no match with the last date in market_cap, then no more data to retrieve

    # print(f"{symbol} Market Cap Duplicates") TODO
    stock_final["market_cap"] = remove_duplicates_and_sort_by_date(market_cap)

    # EOD sometimes has one more later date, so we just remove it
    if stock_final["market_cap"][0]["date"] != stock_final["eod"][0]["date"]:
        stock_final["eod"] = stock_final["eod"][1:]

    # Calculate outstanding shares since FMP only supplies till 2021
    # using daily market cap / closing price
    # Round to nearest integer
    stock_final["outstanding_shares"] = {
        cap["date"]: int(round(cap["marketCap"] / price["price"], 0))
        for cap, price in zip(stock_final["market_cap"], stock_final["eod"])
    }

    return stock_final


def get_all_stock_info(
    stock_profiles,
    start_index=0,
    calls_per_minute=750,
    calls_per_stock=10,
    sleep_buffer=5,
):
    """
    Retrieves extended stock information for a list of stocks while respecting API rate limits.

    This function collects financial statements, market capitalization, and end-of-day (EOD) stock price data
    for each stock in the input list using the `get_stock_info` function. It enforces an API call rate limit
    to ensure no more than `calls_per_minute` API calls are made within any given minute, with a configurable
    sleep buffer to add extra safety against rate limit breaches.

    Parameters
    ----------
    stock_profiles : list of dict
        A list of dictionaries where each dictionary represents a stock and must include at least a 'symbol' key.
    start_index : int, optional
        The index to start from when iterating the stocks.
    calls_per_minute : int, optional
        The maximum number of API calls allowed per 60 seconds. Default is 200.
    calls_per_stock : int, optional
        Estimated number of API calls made per stock. Default is 10 (7 guarenteed and 3 incase market cap is requested multiple times)
    sleep_buffer : int, optional
        Additional seconds to sleep after reaching the rate limit to provide a safe margin. Default is 5.

    Returns
    -------
    dict
        A dictionary with the keys:
        - 'success': list of symbols successfully processed.
        - 'failed': list of symbols for which data fetching failed.

    Notes
    -----
    - If the estimated API calls exceed the allowed `calls_per_minute`, the function will pause execution
      for 60 seconds (plus an optional buffer) before continuing.
    - Any errors encountered while fetching a specific stock's data will be logged, and the function
      will continue with the remaining stocks.
    - The `calls_per_stock` value should reflect the number of API calls made inside `get_stock_info`.
    """

    calls_used = 0
    start_time = time.time()

    success = []
    failed = []

    for i in range(start_index, len(stock_profiles)):
        stock = stock_profiles[i]
        # Track API usage — assuming 4 API calls per stock
        calls_used += calls_per_stock

        # Number of calls exceeded maximum number of call per minute
        if calls_used >= calls_per_minute:
            elapsed = time.time() - start_time
            # Check if the first call was
            if elapsed < 60:
                # time_to_sleep = 60 - elapsed + sleep_buffer # Assumes all api calls take the same ammount of time (but can be that 150 api calls happen in the last 20 seconds)
                time_to_sleep = 60
                print(
                    f"Sleeping for {time_to_sleep:.2f} seconds to respect API limit..."
                )
                time.sleep(time_to_sleep)
            calls_used = 0
            start_time = time.time()

        try:
            stock_info = get_stock_info(stock)

            path = RAW_DIR / f"{stock["symbol"]}.json"
            write_json(path, stock_info)

            print(f"Stock {stock["symbol"]} , Index {i} saved")
            success.append(stock["symbol"])

            # Ensure efficient memory usage
            del stock_info
            gc.collect()

        except Exception as e:
            print(f"Error fetching data for {stock['symbol']}: {e}")
            failed.append(stock["symbol"])
            continue

    return {"success": success, "failed": failed}


def add_info(
    path,
    api_req_func,
    info_name,
    stock_params,
    const_params=None,
    sort_key=None,
    reverse=True,
    start_index=0,
    end_index=None,
):
    """
    Enriches stock data JSON files with information retrieved from a specified API function.

    This function loads all stock JSON files from a directory, applies an API request function
    to each stock using both dynamic (stock-specific) and optional constant parameters, and stores
    the API response under a specified key in each stock's data. The result can optionally be
    sorted before saving. The updated stock data is saved back to the file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the directory containing individual stock JSON files.
    api_req_func : callable
        Function to call the API. It should accept keyword arguments based on the keys in `stock_params` and `const_params`.
    info_name : str
        The key under which the API response will be stored in each stock's dictionary.
    stock_params : dict
        Dictionary mapping argument names (for the API function) to keys in each stock's dictionary.
    const_params : dict, optional
        Dictionary of constant parameters to pass to the API function (default is None).
    sort_key : str, optional
        If provided, the API response (assumed to be a list of dicts) is sorted by this key (default is None).
    reverse : bool, optional
        Whether to reverse the sort order (default is True).
    start_index : int, optional
        Index of the first stock to process (default is 0).
    end_index : int, optional
        Index at which to stop processing stocks (exclusive). If None, processes all stocks to the end (default is None).

    Returns
    -------
    None

    Notes
    -----
    - Each stock's updated data is saved back to its corresponding JSON file.
    - Stocks that encounter exceptions during processing are skipped and logged to the console.
    """

    stocks = load_all_stocks(path)

    success = []
    failed = []
    for i, stock in enumerate(stocks):
        # Check if within the limits
        if i < start_index:
            continue
        if end_index is not None and i >= end_index:
            break

        try:
            print(f"Stock {stock['symbol']} , Index {i} started")

            # Dynamically build the dictionary of arguments to pass
            call_params = {}

            # Extract stock-specific parameters
            for key, stock_key in stock_params.items():
                call_params[key] = stock[stock_key]

            # Add constant parameters if any
            if const_params:
                call_params.update(const_params)

            # Call the API function with all params as kwargs
            response = api_req_func(**call_params)

            if sort_key is not None:
                stock[info_name] = sorted(
                    response, key=lambda split: split[sort_key], reverse=reverse
                )
            else:
                stock[info_name] = response

            path_stock = path / f"{stock["symbol"]}.json"
            write_json(path_stock, stock)

            print(f"Stock {stock["symbol"]} , Index {i} saved")
            success.append(stock["symbol"])

        except Exception as e:
            print(f"Error fetching data for {stock['symbol']}: {e}")
            failed.append(stock["symbol"])
            continue
