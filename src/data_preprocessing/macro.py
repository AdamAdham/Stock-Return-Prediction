import pandas as pd

from src.config.settings import MACRO_DATA


def get_macro_data() -> pd.DataFrame:
    # Read csv
    macro = pd.read_csv(MACRO_DATA)

    # Format df index to be YYYY-MM
    macro["yyyymm"] = (
        macro["yyyymm"].astype(str).str[:4] + "-" + macro["yyyymm"].astype(str).str[4:]
    )
    macro.set_index("yyyymm", inplace=True)
    macro.index.name = "month"

    # Remove commas from column Index
    macro["Index"] = macro["Index"].str.replace(",", "").astype("float64")

    # Remove dates that will never be used
    # "1962-01-02" is the earliest date in our dataset
    macro = macro[macro.index >= "1962-01-02"]

    # Fill NaN values from "2003-01" with the average
    macro["csp"] = macro["csp"].fillna(macro["csp"].mean())

    return macro


def add_macro_data(df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
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


def add_macro_data(
    df: pd.DataFrame, macro_data: pd.DataFrame, nan_gaps: bool = False
) -> pd.DataFrame:
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

    nan_gaps : bool, optional
        If True, fills missing months with NaN gaps rather than removing them. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A new DataFrame resulting from the horizontal concatenation of the original stock data and
        the filtered macroeconomic data, aligned by index.

    Notes
    -----
    - It is assumed that both `df` and `macro_data` have an index in "YYYY-MM" format.
    """

    if nan_gaps:
        # Remove all months that are before the earliest months and after the latest months for that stock
        # So leaving an gaps that are filled with nan rather than removed row entirely
        earliest_date = df.index[0]
        latest_date = df.index[-1]
        macro_filtered = macro_data[
            (macro_data.index >= earliest_date) & (macro_data.index <= latest_date)
        ]
    else:
        # Ensure both dataframes share the same index range
        common_index = df.index.intersection(macro_data.index)
        # Filter macro_data to only include rows with the common index to not add nan gaps to stock data
        macro_filtered = macro_data.loc[common_index]

    # Concatenate along columns, only where indices match
    df_conc = pd.concat([df, macro_filtered], axis=1)

    return df_conc
