import pandas as pd
import numpy as np
from src.utils.helpers import month_gap_diff


def check_nans_only_at_top(df: pd.DataFrame) -> dict[str, bool]:
    """
    Checks whether each column in a DataFrame has NaN values only at the top (i.e., before the first non-NaN value).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to check.

    Returns
    -------
    dict
        A dictionary where keys are column names and values are booleans:
        - True if all NaN values in the column are located only before the first non-NaN value (or if the column is entirely NaN).
        - False if there are any NaNs after the first non-NaN value.
    """

    result = {}
    for col in df.columns:
        first_valid_label = df[col].first_valid_index()
        if first_valid_label is None:
            result[col] = True  # All NaNs
            continue
        # Convert label to positional index
        pos = df.index.get_loc(first_valid_label)
        non_nan_after = df[col].iloc[pos:].isna().any()
        result[col] = not non_nan_after
    return result


def get_month_gaps(df: pd.DataFrame) -> list:
    """
    Gets the gaps in the month index of a DataFrame and returns a list of missing months.

    The function assumes that the index of `df` is of type `datetime` or can be interpreted as months in the "YYYY-MM" format.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a datetime-like index, where the index represents months (e.g., "YYYY-MM").

    Returns
    -------
    list
        A list of missing months in the "YYYY-MM" format.

    Notes
    -----
    - The function assumes the DataFrame has a monthly frequency.
    - The function will return an empty list if there are no gaps.
    """

    # Generate a range of months between the first and last month
    first_month = df.index.min()
    last_month = df.index.max()

    all_months = pd.date_range(start=first_month, end=last_month, freq="MS").strftime(
        "%Y-%m"
    )

    # Find missing months by comparing the full range with the actual months in the index
    missing_months = sorted(set(all_months) - set(df.index))

    return missing_months, all_months


def get_symbol_month_gaps(
    df: pd.DataFrame,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int | list[int]]]]:
    """
    For each symbol, calculates the gaps in months between consecutive dates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'symbol' column and a datetime index (e.g., "YYYY-MM").

    Returns
    -------
    tuple
        A tuple containing:
        - dict[str, list[dict]]: For each symbol, a list of dictionaries with keys
          'previous_date', 'next_date', and 'gap' indicating non-consecutive month gaps.
        - dict[str, dict[str, int | list[int]]]: For each symbol, a summary dictionary with:
            - 'count': total number of dates
            - 'gaps': list of month gaps greater than 0
    """
    df = df.copy()
    result = {}
    equivalent = {}

    for symbol, group in df.groupby("symbol"):
        dates = group.index.to_list()
        result[symbol] = []
        equivalent[symbol] = {"gaps": [], "count": len(dates)}

        for i in range(len(dates) - 1):
            gap = month_gap_diff(dates[i], dates[i + 1])
            if gap > 0:
                result[symbol].append(
                    {"previous_date": dates[i], "next_date": dates[i + 1], "gap": gap}
                )
                equivalent[symbol]["gaps"].append(gap)

    return result, equivalent


def parse_column_quantile_intervals(df: pd.DataFrame, bins: int = 10) -> dict:
    """
    Computes quantile-based interval counts for each numeric column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns to be binned.
    bins : int, optional
        Number of quantile bins to compute (default is 10).

    Returns
    -------
    dict
        A dictionary where keys are column names and values are:
            - A dictionary mapping interval strings to value counts if binning succeeds.
            - A string message if the column lacks sufficient variation to compute quantiles.

    Notes
    -----
    - Only numeric columns are processed; others are ignored.
    - NaN values are excluded from the binning operation.
    - Uses unique quantile edges to prevent duplicate bin boundaries.
    """

    result = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Drop NaN values
            non_na_values = df[col].dropna()

            # Create quantile-based bins
            try:
                bin_edges = np.unique(
                    np.quantile(non_na_values, np.linspace(0, 1, bins + 1))
                )

                # Cut based on bin edges
                bin_counts = (
                    pd.cut(
                        non_na_values,
                        bins=bin_edges,
                        include_lowest=True,
                        duplicates="drop",
                    )
                    .value_counts()
                    .sort_index()
                )

                # Store results
                result[col] = {
                    str(interval): count for interval, count in bin_counts.items()
                }
            except ValueError:
                # If all values are the same, quantile splitting will fail
                result[col] = "Not enough variation to split"
        else:
            continue

    return result


def parse_list_quantile_intervals(values: list[float], bins: int = 10) -> dict:
    values = pd.Series(values).dropna()

    try:
        # Create quantile-based bin edges
        bin_edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))

        # Cut values into bins
        bin_counts = (
            pd.cut(values, bins=bin_edges, include_lowest=True, duplicates="drop")
            .value_counts()
            .sort_index()
        )

        return {str(interval): count for interval, count in bin_counts.items()}

    except ValueError:
        # If not enough unique values
        return {"Error": "Not enough variation to split into bins."}
