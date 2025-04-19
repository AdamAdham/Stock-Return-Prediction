import traceback
import pandas as pd

from src.config.settings import MACRO_DATA, PROCESSED_DIR, DATAFRAMES_DIR
from src.utils.json_io import load_all_stocks


def format_macro_data():
    # Read csv
    macro = pd.read_csv(MACRO_DATA)

    # Format df index to be YYYY-MM
    macro["yyyymm"] = (
        macro["yyyymm"].astype(str).str[:4] + "-" + macro["yyyymm"].astype(str).str[4:]
    )
    macro.set_index("yyyymm", inplace=True)
    macro.index.name = None

    # Remove commas from column Index
    macro["Index"] = macro["Index"].str.replace(",", "")

    # Remove dates that will never be used
    # "1962-01-02" is the earliest date in our dataset
    macro = macro[macro.index >= "1962-01-02"]

    return macro


def json_dataframe_all(
    start_index=0,
    end_index=None,
    input_directory=PROCESSED_DIR,
    output_directory=DATAFRAMES_DIR,
):
    stocks = load_all_stocks(input_directory)
    macro_data = format_macro_data()
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
            df = json_to_dataframe(stock, macro_data)
            path_output = output_directory / f"{stock["symbol"]}.csv"
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


def json_to_dataframe(stock, macro_data):
    """
    Convert a stock's JSON data and macroeconomic data into a combined pandas DataFrame.

    Parameters
    ----------
    stock : dict
        A dictionary containing the stock's features and subfeatures, typically loaded from a JSON file.
        It should include:
            - stock["features"]["monthly"]: Monthly features as a dict.
            - stock["features"]["quarterly"]: Quarterly features as a dict.
            - stock["features"]["annual"]: Annual features as a dict.
            - stock["subfeatures"]["monthly"]: Monthly subfeatures as a dict.

    macro_data : pandas.DataFrame
        A DataFrame containing macroeconomic indicators indexed by date (must be datetime index).
        Only rows with dates later than or equal to the earliest date in the stock's features are included.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame combining the stock's features, subfeatures, and filtered macroeconomic data.
        The result is aligned by date index and ready for further processing or model input.
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
    df_sorted

    # Remove all dates that are before the earliest date and after the latest date for that stock
    earliest_date = df_sorted.index[0]
    latest_date = df_sorted.index[-1]
    macro_filtered = macro_data[
        (macro_data.index >= earliest_date) & (macro_data.index <= latest_date)
    ]
    df_conc = pd.concat([df_sorted, macro_filtered], axis=1)
    df_conc

    return df_conc


def fill_quarterly_annual(df, inplace=True):
    # Prevent inplace changes
    if inplace:
        df_temp = df
    else:
        df_temp = df.copy()

    # Fill the quarterly and annual columns with the most recent non NaN entry
    columns = [
        "ep_quarterly",
        "sp_quarterly",
        "agr_quarterly",
        "ep_annual",
        "sp_annual",
        "agr_annual",
    ]
    df_temp[columns] = df_temp[columns].ffill()

    return df_temp
