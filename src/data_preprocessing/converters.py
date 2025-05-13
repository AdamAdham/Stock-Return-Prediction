import traceback
import pandas as pd

from src.config.settings import PROCESSED_DIR, DATAFRAMES_DIR
from src.utils.disk_io import load_all_stocks


def json_dataframe_all(
    start_index: int = 0,
    end_index: int = None,
    input_directory: str = PROCESSED_DIR,
    output_directory: str = DATAFRAMES_DIR,
) -> dict:
    stocks = load_all_stocks(input_directory)
    success = []
    failed = []

    for i, stock in enumerate(stocks):
        # Check if within the limits
        if i < start_index:
            continue
        if end_index is not None and i >= end_index:
            break

        try:
            # Convert json to dataframe and save it to csv
            df = json_to_dataframe(stock)
            path_output = output_directory / f"{stock['symbol']}.csv"
            df.to_csv(path_output)
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

    return {
        "success": success,
        "failed": failed,
    }


def json_to_dataframe(stock):
    """
    Convert a stock's JSON data into pandas DataFrame.

    Parameters
    ----------
    stock : dict
        A dictionary containing the stock's features and subfeatures, typically loaded from a JSON file.
        It should include:
            - stock["features"]["monthly"]: Monthly features as a dict.
            - stock["features"]["quarterly"]: Quarterly features as a dict.
            - stock["features"]["annual"]: Annual features as a dict.
            - stock["subfeatures"]["monthly"]: Monthly subfeatures as a dict.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame combining the stock's features, subfeatures.
        The result is aligned and sorted (earliest to latest) by date index and ready for further processing or model input.
    """
    df_features = pd.DataFrame(stock["features"]["monthly"])
    df_quarterly = pd.DataFrame(stock["features"]["quarterly"])
    df_annual = pd.DataFrame(stock["features"]["annual"])

    # Include all monthly subfeatures except "month_latest_week"
    df_sub = pd.DataFrame(
        {
            k: v
            for k, v in stock["subfeatures"]["monthly"].items()
            if k != "month_latest_week"
        }
    )
    df_temp = pd.concat([df_features, df_sub, df_quarterly, df_annual], axis=1)

    # Sort by index/date
    df_sorted = df_temp.sort_index(ascending=True)

    static_metadata = {
        "symbol": stock["symbol"],
        "sic_code_2": stock["sicCode_2"],
        "sic_industry": stock["sicIndustry"],
        "exchange_short_name": stock["exchangeShortName"],
        "exchange": stock["exchange"],
    }

    for key, value in static_metadata.items():
        df_sorted[key] = value

    return df_sorted
