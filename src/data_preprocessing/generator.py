from src.data_preprocessing.preprocessing import (
    fill_quarterly_annual,
    fill_nan,
)
from src.data_preprocessing.macro import add_macro_data
from src.data_preprocessing.feature_selection import (
    remove_stock_columns,
)
import pandas as pd


def format_stock_df(
    df: pd.DataFrame, macro: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Preprocesses a stock-level DataFrame by applying a series of cleaning and enrichment steps.

    Steps performed:
    1. Fills in missing quarterly and annual data.
    2. Separates and retains static company information.
    3. Drops rows with missing 1-month momentum ('mom1m').
    4. Merges macroeconomic data into the stock DataFrame.
    5. Fills remaining missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The raw stock-level DataFrame. Must contain at least the 'mom1m' column.

    macro : pd.DataFrame
        A DataFrame containing macroeconomic indicators to be merged with the stock data.

    Returns
    -------
    tuple
        - pd.DataFrame: The cleaned and enriched stock-level DataFrame.
        - pd.DataFrame: A DataFrame containing the static company information that was separated.
    """
    df_temp = df.copy()
    df_temp = fill_quarterly_annual(df_temp)
    df_temp, static_info = remove_stock_columns(df_temp, remove_static=False)
    df_temp = df_temp.dropna(subset=["mom1m"])
    df_temp = add_macro_data(df_temp, macro)
    df_temp = fill_nan(df_temp)
    return df_temp, static_info


def combine_dfs(dfs: list[pd.DataFrame]):
    """
    Combines a list of stock-level DataFrames into a single DataFrame with consistent formatting.

    Steps performed:
    1. Concatenates the input DataFrames row-wise.
    2. Fixes incorrect symbol values where boolean `True` is used instead of the string "TRUE".
    3. Sets the index name to "month".
    4. Sorts the resulting DataFrame by 'symbol' and index ('month').

    Parameters
    ----------
    dfs : list of pd.DataFrame
        A list of DataFrames containing stock data. Each DataFrame must include a 'symbol' column
        and have a datetime index representing months.

    Returns
    -------
    pd.DataFrame
        The combined and sorted DataFrame.
    """

    # Combine the stock dataframes row by row
    df_combined = pd.concat(dfs)

    # Revert TRUE Symbol
    df_combined.loc[df_combined["symbol"] == True, "symbol"] = "TRUE"
    df_combined.index.name = "month"

    # Sort by symbol then month/index
    df_combined = df_combined.sort_values(by=["symbol", df_combined.index.name])
    return df_combined
