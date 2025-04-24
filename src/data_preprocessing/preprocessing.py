import traceback
import pandas as pd

from src.config.settings import MACRO_DATA, PROCESSED_DIR, DATAFRAMES_DIR
from src.utils.disk_io import load_all_stocks
from sklearn.preprocessing import StandardScaler


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
    macro["Index"] = macro["Index"].str.replace(",", "").astype("float64")

    # Remove dates that will never be used
    # "1962-01-02" is the earliest date in our dataset
    macro = macro[macro.index >= "1962-01-02"]

    return macro


def add_macro_data(df, macro_data):
    """
    Appends macroeconomic data to a stock-specific DataFrame by aligning on the months "YYYY-MM" index.

    This function filters the `macro_data` DataFrame to match the months range of the `df` DataFrame
    (i.e., the stock data), and then concatenates the filtered macroeconomic data to the stock data
    along the columns.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing stock-specific features with a month index.

    macro_data : pd.DataFrame
        A DataFrame containing macroeconomic indicators, also indexed by month.

    Returns
    -------
    pd.DataFrame
        A new DataFrame resulting from the horizontal concatenation of the original stock data and
        the filtered macroeconomic data, aligned by index.

    Notes
    -----
    - It is assumed that both `df` and `macro_data` have a month index.
    """

    # Remove all months that are before the earliest months and after the latest months for that stock
    earliest_date = df.index[0]
    latest_date = df.index[-1]
    macro_filtered = macro_data[
        (macro_data.index >= earliest_date) & (macro_data.index <= latest_date)
    ]
    df_conc = pd.concat([df, macro_filtered], axis=1)

    return df_conc


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
            df = json_to_dataframe(stock)
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
        "sicCode_2": stock["sicCode_2"],
        "exchangeShortName": stock["exchangeShortName"],
        "exchange": stock["exchange"],
    }

    for key, value in static_metadata.items():
        df_sorted[key] = value

    return df_sorted


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


def remove_stock_columns(
    columns=None,
    remove_current=False,
    remove_lagged=True,
    remove_subfeatures=True,
    additional_columns=None,
):
    """
    Removes specified columns from a stock data DataFrame based on user-defined criteria.

    This function allows for the selective removal of columns from a stock DataFrame.
    It provides options to remove 'current' columns, 'lagged' columns, 'subfeatures' columns,
    and any additional columns specified by the user. If no columns are specified, the function will
    remove columns according to the flags `remove_current`, `remove_lagged`, `remove_subfeatures`,
    and `additional_columns`.

    Parameters
    ----------
    columns : list, optional
        List of specific column names to remove. If not provided, columns will be removed based
        on the flags `remove_current`, `remove_lagged`, `remove_subfeatures`, and `additional_columns`.
    remove_current : bool, optional
        Whether to remove 'current' columns (default is False).
    remove_lagged : bool, optional
        Whether to remove 'lagged' columns (default is True).
    remove_subfeatures : bool, optional
        Whether to remove 'subfeatures' columns (default is True).
    additional_columns : list, optional
        List of additional specific columns to remove, regardless of the other flags.

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified columns removed.

    Notes
    -----
    The columns that are removed are from the following categories:
    - 'current' columns: e.g., "mom12m_current", "chmom_current", etc.
    - 'lagged' columns: e.g., "mom12m", "chmom", etc.
    - 'subfeatures' columns: e.g., "rolling_avg_3y_returns_weekly_by_month", "prices_monthly", etc.

    If `columns` is provided, only those specific columns will be removed regardless of the flags.
    The `additional_columns` parameter allows for removing extra columns beyond the predefined categories.
    """

    current_cols = [
        "mom12m_current",
        "chmom_current",
        "maxret_current",
        "mve_current",
        "dolvol_current",
        "beta_current",
        "betasq_current",
        "idiovol_current",
    ]
    lagged_cols = [
        "mom12m",
        "chmom",
        "maxret",
        "mve",
        "dolvol",
        "beta",
        "betasq",
        "idiovol",
    ]
    subfeatures_cols = [
        "rolling_avg_3y_returns_weekly_by_month",
        "prices_monthly",
        "month_latest_week",
        "vol_sum_monthly",
        "shares_monthly",
        "zero_trading_days_count_monthly",
        "trading_days_count_monthly",
        "market_cap",
    ]

    if columns is None:
        columns = []
        columns += current_cols if remove_current else []
        columns += lagged_cols if remove_lagged else []
        columns += subfeatures_cols if remove_subfeatures else []
        columns += additional_columns if additional_columns is not None else []

    if columns:
        df = df.drop(columns=columns)
    return df


def scale_df(df, target_col, scaler_class=StandardScaler, ratio=0.7):
    """
    Scales the features and target column of a DataFrame using a specified scaler,
    fitting the scalers only on a portion of the data (typically the training set).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and the target column.

    target_col : str
        The name of the target column to be scaled separately.

    scaler_class : scikit-learn scaler instance, optional
        An instance of a scikit-learn scaler (e.g., StandardScaler(), MinMaxScaler()).
        Defaults to StandardScaler().

    ratio : float, optional
        The ratio of the dataset to use for fitting the scaler (typically corresponds to the training portion).
        Defaults to 0.7.

    Returns
    -------
    df_scaled : pd.DataFrame
        A new DataFrame with scaled features and target column, preserving the original index.

    scaler_x : scikit-learn Scaler instance
        The fitted scaler used to transform the feature columns.

    scaler_y : scikit-learn Scaler instance
        The fitted scaler used to transform the target column.

    """

    # Determine the index for the split
    split_idx = int(len(df) * ratio)

    # Identify features (all except the target)
    columns_x = df.columns.difference([target_col]).tolist()

    # Fit scalers only on the training portion
    scaler_x = scaler_class()
    scaler_x.fit(df.iloc[:split_idx][columns_x])

    scaler_y = scaler_class()
    scaler_y.fit(df.iloc[:split_idx][[target_col]])

    # Transform the full dataset using the fitted scalers
    df_scaled = pd.DataFrame(
        scaler_x.transform(df[columns_x]), columns=columns_x, index=df.index
    )
    df_scaled[target_col] = scaler_y.transform(df[[target_col]])

    return df_scaled, scaler_x, scaler_y


def sequential_cross_sectional_split(x, y, train_ratio=0.7, val_ratio=0.15):
    total_len = len(x)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test
