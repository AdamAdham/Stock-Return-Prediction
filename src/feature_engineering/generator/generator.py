import traceback

from src.utils.json_io import load_all_stocks
from src.feature_engineering.calculations.momentum import (
    calculate_mom1m,
    calculate_mom12m,
    calculate_mom12m_current,
    calculate_mom36m,
    calculate_chmom,
    calculate_chmom_current,
    calculate_maxret,
    calculate_maxret_current,
    handle_indmom,
)

from src.feature_engineering.calculations.liquidity import (
    calculate_turn,
    calculate_std_turn,
    calculate_mve,
    calculate_mve_current,
    calculate_dolvol,
    calculate_dolvol_current,
    calculate_ill,
    calculate_zerotrade,
    calculate_zerotrade_current,
)

from src.feature_engineering.calculations.risk import (
    calculate_retvol,
    calculate_idiovol,
    calculate_beta_betasq,
)

from src.feature_engineering.calculations.ratios import (
    calculate_ep_sp_annual,
    calculate_ep_sp_quarterly,
    calculate_agr_annual,
    calculate_agr_quarterly,
)

from src.feature_engineering.utils import (
    get_dollar_volume_monthly,
    get_market_cap_monthly,
    get_monthly_price,
    get_rolling_weekly_returns,
    get_stock_returns_weekly,
    get_volume_shares_statistics,
    handle_market_returns_weekly,
)

from src.utils.json_io import write_json
from src.utils.information import get_sic_industry_names

from src.config.settings import RAW_DIR, PROCESSED_DIR


def get_features(stock):
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

    months_sorted, prices_monthly = get_monthly_price(prices_daily)
    dollar_volume_monthly = get_dollar_volume_monthly(prices_daily)
    weeks_sorted, month_latest_week, weekly_returns = get_stock_returns_weekly(
        prices_daily
    )

    max_ret_current = calculate_maxret_current(prices_daily)

    # To store variables that were used in feature calculation, but can be helpful in the future
    # TODO Remove any that are not useful
    subfeatures = {
        "weekly": {},
        "monthly": {},
        "quarterly": {},
        "annual": {},
        "lists": {},
    }
    subfeatures["monthly"]["rolling_avg_3y_weekly_returns_by_month"] = (
        get_rolling_weekly_returns(
            weeks_sorted,
            months_sorted,
            month_latest_week,
            weekly_returns,
            interval=156,
            increment=4,
        )
    )
    subfeatures["lists"]["months_sorted"] = months_sorted
    subfeatures["lists"]["weeks_sorted"] = weeks_sorted
    subfeatures["monthly"]["prices_monthly"] = prices_monthly
    subfeatures["monthly"]["dollar_volume_monthly"] = dollar_volume_monthly
    subfeatures["monthly"]["month_latest_week"] = month_latest_week
    subfeatures["weekly"]["weekly_returns"] = weekly_returns

    # Feature Engineering
    features = {"weekly": {}, "monthly": {}, "quarterly": {}, "annual": {}}

    features["monthly"]["mom1m"] = calculate_mom1m(months_sorted, prices_monthly)
    features["monthly"]["mom12m"] = calculate_mom12m(months_sorted, prices_monthly)
    features["monthly"]["mom12m_current"] = calculate_mom12m_current(
        months_sorted, prices_monthly
    )
    features["monthly"]["mom36m"] = calculate_mom36m(months_sorted, prices_monthly)
    features["monthly"]["chmom"] = calculate_chmom(months_sorted, prices_monthly)
    features["monthly"]["chmom_current"] = calculate_chmom_current(
        months_sorted, prices_monthly
    )
    features["monthly"]["maxret"] = calculate_maxret(months_sorted, max_ret_current)
    features["monthly"]["maxret_current"] = max_ret_current

    # Calculate all features that depend on the availability of outstanding shares
    if outstanding_shares:
        (
            vol_sum,
            shares_monthly,
            zero_trading_days_count,
            trading_days_count,
            _,
        ) = get_volume_shares_statistics(prices_daily, outstanding_shares)
        features["monthly"]["zerotrade"] = calculate_zerotrade(
            months_sorted,
            vol_sum,
            shares_monthly,
            zero_trading_days_count,
            trading_days_count,
        )
        features["monthly"]["zerotrade_current"] = calculate_zerotrade_current(
            months_sorted,
            vol_sum,
            shares_monthly,
            zero_trading_days_count,
            trading_days_count,
        )

        features["monthly"]["turn"] = calculate_turn(
            months_sorted, vol_sum, shares_monthly
        )
        features["monthly"]["std_turn"] = calculate_std_turn(
            prices_daily, outstanding_shares
        )

        # Populate intermediate variables
        subfeatures["monthly"]["vol_sum"] = vol_sum
        subfeatures["monthly"]["shares_monthly"] = shares_monthly
        subfeatures["monthly"]["zero_trading_days_count"] = zero_trading_days_count
        subfeatures["monthly"]["trading_days_count"] = trading_days_count

    else:
        features["monthly"]["turn"] = None
        features["monthly"]["std_turn"] = None
        features["monthly"]["zerotrade"] = None

        subfeatures["monthly"]["vol_sum"] = None
        subfeatures["monthly"]["shares_monthly"] = None
        subfeatures["monthly"]["zero_trading_days_count"] = None
        subfeatures["monthly"]["trading_days_count"] = None

    # Calculate all features that depend on the availability of market cap
    if market_caps:
        market_cap_monthly = get_market_cap_monthly(market_caps)
        features["monthly"]["mve"] = calculate_mve(months_sorted, market_cap_monthly)
        features["monthly"]["mve_current"] = calculate_mve_current(market_cap_monthly)
        ep_annual, sp_annual = calculate_ep_sp_annual(
            income_statements_annual, market_caps
        )

        ep_quarterly, sp_quarterly = calculate_ep_sp_quarterly(
            income_statements_quarterly, market_caps
        )
        features["annual"]["ep"] = ep_annual
        features["annual"]["sp"] = sp_annual
        features["quarterly"]["ep"] = ep_quarterly
        features["quarterly"]["sp"] = sp_quarterly

        subfeatures["monthly"]["market_cap"] = market_cap_monthly
    else:
        features["monthly"]["mve"] = None
        features["annual"]["ep"] = None
        features["annual"]["sp"] = None

        subfeatures["monthly"]["market_cap"] = None

    features["monthly"]["dolvol"] = calculate_dolvol(
        months_sorted, dollar_volume_monthly
    )
    features["monthly"]["dolvol_current"] = calculate_dolvol_current(
        months_sorted, dollar_volume_monthly
    )
    features["monthly"]["ill"] = calculate_ill(prices_daily)
    features["monthly"]["retvol"] = calculate_retvol(prices_daily)

    features["annual"]["agr"] = calculate_agr_annual(balance_sheet_annual)
    features["quarterly"]["agr"] = calculate_agr_quarterly(balance_sheet_quarterly)

    stock["features"] = features
    stock["subfeatures"] = subfeatures

    return stock


def enrich_stocks_with_features(
    start_index=0,
    end_index=None,
    input_directory=RAW_DIR,
    output_directory=PROCESSED_DIR,
):
    """
    Enriches stock data with statistical features, industry momentum, and market returns.

    This function processes raw stock JSON files from the input directory by computing
    statistical features (via `get_features`), then saves the enriched stock data to disk.
    It also aggregates data to compute:
    - Industry momentum (indmom) for each SIC code and month.
    - Weekly average market returns across all stocks.

    Parameters
    ----------
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
        - aggregate_stats : dict
            A dictionary with:
            - 'indmom' : dict
                A nested dictionary structured as {sic_code: {month: average_industry_momentum}}.
            - 'market_returns' : dict
                A dictionary structured as {week: average_weekly_market_return}.
        - status : dict
            A dictionary with:
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
    sic_codes_names = get_sic_industry_names()

    indmom = {sic_code: {} for sic_code in sic_codes_names}
    market_return_details = {}
    success = []
    failed = []
    for i, stock in enumerate(stocks):
        if i < start_index:
            continue  # Skip until we reach start_index
        if end_index is not None and i >= end_index:
            break
        try:
            print(f"Stock {stock['symbol']} , Index {i} started")

            enriched_stock = get_features(stock)

            # Update indmom and market returns sum and count to be averaged once done
            indmom = handle_indmom(enriched_stock, indmom)
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
            print("-" * 100)
            print("\n \n \n")

            failed.append(stock["symbol"])
            continue

    # Get average of all months_sorted for each SIC code
    for sic, months_sorted in indmom.items():
        for month, data in months_sorted.items():
            indmom[sic][month] = data["total"] / data["count"]

    # Get average of weekly market returns
    market_returns = {}
    for week, returns in market_return_details.items():
        market_returns[week] = returns["sum"] / returns["count"]

    return {"indmom": indmom, "market_returns": market_returns}, {
        "success": success,
        "failed": failed,
    }


def enrich_stocks_with_aggregate_features(
    indmom,
    market_returns_weekly,
    start_index=0,
    end_index=None,
    input_directory=PROCESSED_DIR,
    output_directory=PROCESSED_DIR,
):
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
            subfeatures = stock["subfeatures"]

            # Calculate beta and betasq
            beta, betasq = calculate_beta_betasq(
                subfeatures["lists"]["weeks_sorted"],
                subfeatures["monthly"]["months_sorted"],
                subfeatures["monthly"]["month_latest_week"],
                subfeatures["weekly"]["weekly_returns"],
                market_returns_weekly,
                interval=156,
                increment=4,
            )

            # Calculate idiovol
            idiovol = calculate_idiovol(
                subfeatures["lists"]["weeks_sorted"],
                subfeatures["monthly"]["months_sorted"],
                subfeatures["monthly"]["month_latest_week"],
                subfeatures["weekly"]["returns"],
                market_returns_weekly,
                interval=156,
                increment=4,
            )

            # Add beta, betasq, and idiovol to stock
            stock["features"][
                "beta"
            ] = beta  # Assuming calculate_beta_betasq returns a tuple (beta, betasq)
            stock["features"]["betasq"] = betasq
            stock["features"]["idiovol"] = idiovol

            sic_2 = stock["sicCode_2"]
            stock["features"]["indmom"] = indmom[sic_2]

            # Save stock to disk
            output_path = output_directory / f"{stock['symbol']}.json"
            write_json(output_path, stock)
            success.append(stock["symbol"])

        except Exception as e:
            print(f"Error processing stock {stock.get('ticker', 'N/A')}: {e}")
            failed.append(stock["symbol"])
            continue

    return {"success": success, "failed": failed}
