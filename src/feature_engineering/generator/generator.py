import traceback

from src.utils.disk_io import load_all_stocks
from src.utils.metrics import time_call
from src.feature_engineering.calculations.momentum import (
    calculate_mom1m,
    calculate_mom12m,
    calculate_mom36m,
    calculate_chmom,
    calculate_maxret,
    handle_indmom,
)

from src.feature_engineering.calculations.liquidity import (
    calculate_turn,
    calculate_std_turn,
    calculate_mve,
    calculate_dolvol,
    calculate_ill,
    calculate_zerotrade,
)

from src.feature_engineering.calculations.risk import (
    calculate_retvol_std,
    calculate_idiovol,
    calculate_beta_betasq,
)

from src.feature_engineering.calculations.ratios import (
    calculate_ep_sp,
    calculate_agr,
)

from src.feature_engineering.utils import (
    get_market_cap_monthly,
    get_rolling_returns_weekly,
    handle_market_returns_weekly,
    get_weekly_monthly_summary,
    get_returns_weekly,
    get_shares_monthly,
)

from src.feature_engineering.filtering import check_stock_validity, filter_stock

from src.utils.disk_io import write_json
from src.utils.information import get_sic_industry_names

from src.config.settings import RAW_DIR, PROCESSED_DIR


def get_features(stock: dict) -> dict:
    """
    Computes and attaches a set of financial and market-related features for a given stock.

    This function extracts time-series and financial data from a stock dictionary and
    computes various statistical and fundamental indicators. These include momentum metrics,
    liquidity measures, valuation ratios, and volatility. The computed features are stored
    in a new 'stats' dictionary and appended to the stock data.

    Parameters
    ----------
    stock : dict
        A dictionary representing a stock, expected to include the following keys:
            - 'eod': List of daily end-of-day prices with 'date' and 'price'.
            - 'market_cap': Market capitalization data over time.
            - 'financials_annual': A dictionary with:
                - 'income_statement': Annual income statement data.
                - 'balance_sheet': Annual balance sheet data.

    Returns
    -------
    dict
        A copy of the input `stock` dictionary with an added 'stats' key, containing:
            - 'mom1m': 1-month momentum.
            - 'mom12m': 12-month momentum.
            - 'mom36m': 36-month momentum.
            - 'chmom': Change in momentum.
            - 'maxret': Maximum daily return per month.
            - 'mve': Market value of equity (average or latest market cap).
            - 'dolvol': Monthly dollar volume.
            - 'amihud': Amihud illiquidity measure.
            - 'retvol': Return volatility.
            - 'ep': Earnings-to-price ratio.
            - 'sp': Sales-to-price ratio.
            - 'agr': Asset growth rate.

    Notes
    -----
    - All assumptions are in each individual function
    """
    stock = stock.copy()

    # Get all raw data needed for calculating the features
    prices_daily = stock["eod"]
    market_caps = stock["market_cap"]
    income_statements_annual = stock["financials_annual"]["income_statement"]
    income_statements_quarterly = stock["financials_quarterly"]["income_statement"]
    balance_sheet_annual = stock["financials_annual"]["balance_sheet"]
    balance_sheet_quarterly = stock["financials_quarterly"]["balance_sheet"]
    outstanding_shares = stock["outstanding_shares"]

    (
        weeks_sorted,
        prices_weekly,
        months_sorted,
        month_latest_week,
        prices_monthly,
        dollar_volume_monthly,
        vol_sum_monthly,
        zero_trading_days_count_monthly,
        trading_days_count_monthly,
        max_ret_current,
        daily_returns_monthly,
    ) = time_call(get_weekly_monthly_summary, prices_daily)
    returns_weekly = time_call(get_returns_weekly, weeks_sorted, prices_weekly)

    # To store variables that were used in feature calculation, but can be helpful in the future
    subfeatures = {
        "weekly": {},
        "monthly": {},
        "quarterly": {},
        "annual": {},
        "lists": {},
    }
    subfeatures["monthly"]["rolling_avg_3y_returns_weekly_by_month"] = time_call(
        get_rolling_returns_weekly,
        weeks_sorted,
        months_sorted,
        month_latest_week,
        returns_weekly,
        interval=156,
        increment=4,
    )
    subfeatures["lists"]["months_sorted"] = months_sorted
    subfeatures["lists"]["weeks_sorted"] = weeks_sorted
    subfeatures["monthly"]["prices_monthly"] = prices_monthly
    subfeatures["monthly"]["month_latest_week"] = month_latest_week
    subfeatures["weekly"]["returns_weekly"] = returns_weekly

    # Feature Engineering
    features = {"weekly": {}, "monthly": {}, "quarterly": {}, "annual": {}}

    features["monthly"]["mom1m"] = time_call(
        calculate_mom1m, months_sorted, prices_monthly
    )
    features["monthly"]["mom12m"] = time_call(
        calculate_mom12m, months_sorted, prices_monthly
    )
    features["monthly"]["mom12m_current"] = time_call(
        calculate_mom12m, months_sorted, prices_monthly, current=True
    )
    features["monthly"]["mom36m"] = time_call(
        calculate_mom36m, months_sorted, prices_monthly
    )
    features["monthly"]["chmom"] = time_call(
        calculate_chmom, months_sorted, prices_monthly
    )
    features["monthly"]["chmom_current"] = time_call(
        calculate_chmom, months_sorted, prices_monthly, current=True
    )
    features["monthly"]["maxret"] = time_call(
        calculate_maxret, months_sorted, max_ret_current
    )
    features["monthly"]["maxret_current"] = max_ret_current

    # Calculate all features that depend on the availability of outstanding shares
    if outstanding_shares:

        shares_monthly = time_call(get_shares_monthly, outstanding_shares)

        features["monthly"]["zerotrade"] = time_call(
            calculate_zerotrade,
            months_sorted,
            vol_sum_monthly,
            shares_monthly,
            zero_trading_days_count_monthly,
            trading_days_count_monthly,
        )
        features["monthly"]["turn"] = time_call(
            calculate_turn, months_sorted, vol_sum_monthly, shares_monthly
        )
        features["monthly"]["std_turn"] = time_call(
            calculate_std_turn, prices_daily, outstanding_shares
        )

        # Populate intermediate variables
        subfeatures["monthly"]["vol_sum_monthly"] = vol_sum_monthly
        subfeatures["monthly"]["shares_monthly"] = shares_monthly
        subfeatures["monthly"][
            "zero_trading_days_count_monthly"
        ] = zero_trading_days_count_monthly
        subfeatures["monthly"][
            "trading_days_count_monthly"
        ] = trading_days_count_monthly
    else:
        features["monthly"]["turn"] = None
        features["monthly"]["std_turn"] = None
        features["monthly"]["zerotrade"] = None
        subfeatures["monthly"]["vol_sum_monthly"] = None
        subfeatures["monthly"]["shares_monthly"] = None
        subfeatures["monthly"]["zero_trading_days_count_monthly"] = None
        subfeatures["monthly"]["trading_days_count_monthly"] = None

    # Calculate all features that depend on the availability of market cap
    if market_caps:
        market_cap_monthly = time_call(get_market_cap_monthly, market_caps)
        features["monthly"]["mve"] = time_call(
            calculate_mve, months_sorted, market_cap_monthly
        )
        features["monthly"]["mve_current"] = time_call(
            calculate_mve, months_sorted, market_cap_monthly, current=True
        )

        ep_annual, sp_annual = time_call(
            calculate_ep_sp, income_statements_annual, market_caps
        )
        ep_quarterly, sp_quarterly = time_call(
            calculate_ep_sp, income_statements_quarterly, market_caps
        )

        features["annual"]["ep_annual"] = ep_annual
        features["annual"]["sp_annual"] = sp_annual
        features["quarterly"]["ep_quarterly"] = ep_quarterly
        features["quarterly"]["sp_quarterly"] = sp_quarterly

        subfeatures["monthly"]["market_cap"] = market_cap_monthly
    else:
        features["monthly"]["mve"] = None
        features["monthly"]["mve_current"] = None
        features["annual"]["ep_annual"] = None
        features["annual"]["sp_annual"] = None
        features["quarterly"]["ep_quarterly"] = None
        features["quarterly"]["sp_quarterly"] = None
        subfeatures["monthly"]["market_cap"] = None

    features["monthly"]["dolvol"] = time_call(
        calculate_dolvol, months_sorted, dollar_volume_monthly
    )
    features["monthly"]["dolvol_current"] = time_call(
        calculate_dolvol, months_sorted, dollar_volume_monthly, current=True
    )
    features["monthly"]["ill"] = time_call(calculate_ill, prices_daily)

    features["monthly"]["retvol"] = time_call(
        calculate_retvol_std, daily_returns_monthly
    )

    features["annual"]["agr_annual"] = time_call(calculate_agr, balance_sheet_annual)
    features["quarterly"]["agr_quarterly"] = time_call(
        calculate_agr, balance_sheet_quarterly
    )

    stock["features"] = features
    stock["subfeatures"] = subfeatures

    return stock


def enrich_stocks_with_features(
    handle_market_return_values: bool = False,
    handle_indmom_values: bool = False,
    start_index: int = 0,
    end_index: int | None = None,
    input_directory: str = RAW_DIR,
    output_directory: str = PROCESSED_DIR,
) -> dict[str, dict]:
    """
    Enriches stock data with statistical features, industry momentum, and market returns.

    This function processes raw stock JSON files from the input directory by computing
    statistical features (via `get_features`), then saves the enriched stock data to disk.
    It also aggregates data to compute:
    - Industry momentum (indmom) for each SIC code and month.
    - Weekly average market returns across all stocks.

    Parameters
    ----------
    handle_market_return_values : bool, optional
        Flag to let the function handle market_returns_weekly.

    handle_indmom_values : bool, optional
        Flag to let the function handle indmom.

    start_index : int, optional
        Index to start processing stocks from. Defaults to 0.

    end_index : int, optional
        Index to stop processing stocks at (exclusive). Defaults to None (process all remaining stocks).

    input_directory : Path or str, optional
        Directory containing raw stock data JSON files. Defaults to `RAW_DIR`.

    output_directory : Path or str, optional
        Directory to save enriched stock data. Defaults to `PROCESSED_DIR`.

    Returns
    -------
    tuple
        A tuple containing:

        aggregate_stats: dict
            - 'indmom' : dict
                A nested dictionary structured as {sic_code: {month: average_industry_momentum}}.
            - 'market_returns_weekly' : dict
                A dictionary structured as {week: average_weekly_market_return}.

        status: dict
            - 'success' : list
                List of stock symbols that were successfully processed.
            - 'failed' : list
                List of stock symbols that encountered errors during processing.

    Notes
    -----
    - Each stock is expected to be stored as a JSON file containing at least a 'symbol' field.
    - Enriched data is saved under the same filename in the specified output directory.
    - Stocks that fail during processing are skipped, and detailed error messages (including stack trace) are printed.
    - Industry momentum and market returns are averaged after all stocks are processed.
    """

    stocks = load_all_stocks(input_directory)

    # If handler flags are True -> initialize appropriately else -> assign None
    if handle_indmom_values:
        sic_codes_names = get_sic_industry_names()
        indmom = {sic_code: {} for sic_code in sic_codes_names}
    else:
        indmom = None
    market_return_details = {} if handle_market_return_values else None

    success = []
    failed = []
    invalid = {}
    filtered = {}

    for i, stock in enumerate(stocks):
        # Check if within the limits
        if i < start_index:
            continue
        if end_index is not None and i >= end_index:
            break

        try:
            print(f"Stock {stock['symbol']} , Index {i} started")

            # Check validity of stock according to a criteria
            valid = check_stock_validity(stock, invalid)
            if not valid:
                print(f"Skipped stock {stock['symbol']} because was invalid")
                continue

            # Calculate features for the stock
            enriched_stock = get_features(stock)

            if handle_indmom_values:
                # Update indmom and market returns sum and count to be averaged once done
                indmom = handle_indmom(enriched_stock, indmom)

            if handle_market_return_values:
                market_return_details = handle_market_returns_weekly(
                    enriched_stock, market_return_details
                )

            # Save stock to disk
            output_path = output_directory / f"{stock['symbol']}.json"
            write_json(output_path, enriched_stock)

            print(f"Stock {stock['symbol']} , Index {i} saved")
            success.append(stock["symbol"])

        except Exception as e:
            # Print detailed error information
            error_message = (
                f"Error processing stock {stock.get('symbol', 'N/A')} at index {i}."
            )
            error_message += (
                f"\nError Type: {type(e).__name__}\nError Message: {str(e)}"
            )
            error_message += f"\nStack Trace:\n{traceback.format_exc()}"
            print(error_message)
            print("-" * 100, "\n \n \n")

            failed.append(stock["symbol"])
            continue

    # Get average of all months_sorted for each SIC code
    for sic, months_sorted in indmom.items():
        for month, data in months_sorted.items():
            indmom[sic][month] = data["total"] / data["count"]

    # Get average of weekly market returns
    market_returns_weekly = {}
    for week, returns in market_return_details.items():

        market_returns_weekly[week] = (
            returns["sum"] / returns["count"] if returns["count"] != 0 else None
        )  # Prevent division by zero

    return {"indmom": indmom, "market_returns_weekly": market_returns_weekly}, {
        "success": success,
        "failed": failed,
        "invalid": invalid,
        "filtered": filtered,
    }


def enrich_stocks_with_aggregate_features(
    indmom: dict[str, dict[str, float]],
    market_returns_weekly: dict[str, float | None],
    start_index: int = 0,
    end_index: int | None = None,
    input_directory: str = PROCESSED_DIR,
    output_directory: str = PROCESSED_DIR,
) -> dict[str, list[str]]:
    """
    Enhances stock data with aggregate statistical metrics using precomputed market and industry information.

    This function loads stock data with intermediate variables from the specified input directory, computes:
    - Beta (stock's sensitivity to the market)
    - Beta squared (used in some factor models)
    - Idiosyncratic volatility (volatility unexplained by the market)
    - Industry momentum (based on SIC code)

    It then appends these metrics under the 'features' field of each stock and saves the enriched version.

    Parameters
    ----------
    indmom : dict
        Dictionary containing industry momentum data of the form {sic_code: {month: avg_industry_momentum}}.

    market_returns_weekly : dict
        Dictionary containing average weekly market returns of the form {week: return}.

    start_index : int, optional
        Index to begin processing stocks from. Defaults to 0.

    end_index : int, optional
        Index to stop processing stocks at (inclusive). Defaults to None (process all).

    input_directory : Path or str, optional
        Directory containing raw stock data JSON files. Defaults to `PROCESSED_DIR`.

    output_directory : Path or str, optional
        Directory to save enriched stock data. Defaults to `PROCESSED_DIR`. (so will overwrite the data)

    Returns
    -------
    dict
        A dictionary with the keys:
        - 'success': list of symbols successfully processed.
        - 'failed': list of symbols for which data fetching failed.

    Notes
    -----
    - Assumes each stock file contains 'subfeatures' and 'features' fields.
    - Assumes 'sicCode_2' is available for mapping industry momentum.
    - Errors during stock processing are logged and the stock is skipped.
    """

    stocks = load_all_stocks(input_directory)
    success = []
    failed = []

    for i, stock in enumerate(stocks):
        if i < start_index:
            continue  # Skip until we reach start_index
        if end_index is not None and i >= end_index:
            break  # Stop if index exceeded specified end_index

        try:
            print(f"Stock {stock['symbol']} , Index {i} started")

            # Extracting variables
            subfeatures = stock["subfeatures"]
            weeks_sorted = subfeatures["lists"]["weeks_sorted"]
            months_sorted = subfeatures["lists"]["months_sorted"]
            month_latest_week = subfeatures["monthly"]["month_latest_week"]
            returns_weekly = subfeatures["weekly"]["returns_weekly"]

            # Calculate beta and betasq
            beta, betasq = calculate_beta_betasq(
                weeks_sorted,
                months_sorted,
                month_latest_week,
                returns_weekly,
                market_returns_weekly,
            )

            beta_current, betasq_current = calculate_beta_betasq(
                weeks_sorted,
                months_sorted,
                month_latest_week,
                returns_weekly,
                market_returns_weekly,
                current=True,
            )

            # Calculate idiovol
            idiovol = calculate_idiovol(
                weeks_sorted,
                months_sorted,
                month_latest_week,
                returns_weekly,
                market_returns_weekly,
            )

            idiovol_current = calculate_idiovol(
                weeks_sorted,
                months_sorted,
                month_latest_week,
                returns_weekly,
                market_returns_weekly,
                current=True,
            )

            # Add beta, betasq, and idiovol to stock
            stock["features"]["monthly"]["beta"] = beta
            stock["features"]["monthly"]["beta_current"] = beta_current
            stock["features"]["monthly"]["betasq"] = betasq
            stock["features"]["monthly"]["betasq_current"] = betasq_current
            stock["features"]["monthly"]["idiovol"] = idiovol
            stock["features"]["monthly"]["idiovol_current"] = idiovol_current

            sic_2 = stock["sicCode_2"]
            stock["features"]["monthly"]["indmom"] = {
                month: indmom[sic_2][month] for month in months_sorted
            }

            # Save stock to disk
            output_path = output_directory / f"{stock['symbol']}.json"
            write_json(output_path, stock)
            success.append(stock["symbol"])
            print(f"Stock {stock['symbol']} , Index {i} saved")

        except Exception as e:
            print("here")
            # Print detailed error information
            error_message = (
                f"Error processing stock {stock.get('symbol', 'N/A')} at index {i}."
            )
            error_message += (
                f"\nError Type: {type(e).__name__}\nError Message: {str(e)}"
            )
            error_message += f"\nStack Trace:\n{traceback.format_exc()}"
            print(error_message)
            print("-" * 100, "\n \n \n")

            failed.append(stock["symbol"])
            continue

    return {"success": success, "failed": failed}
