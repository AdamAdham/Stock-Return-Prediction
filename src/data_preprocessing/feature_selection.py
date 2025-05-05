import numpy as np
import pandas as pd


def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "pearson",
    inplace: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Identifies and removes columns with correlation above the given threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing numerical features.
    threshold : float
        The correlation threshold above which to remove one of the correlated features.
    method : str
        Correlation method: 'pearson', 'kendall', or 'spearman'.
    inplace : bool
        If True, drops columns from the original DataFrame. Otherwise, returns a new DataFrame.

    Returns:
    --------
    reduced_df : pd.DataFrame
        A DataFrame with highly correlated columns removed (if inplace=False).
    dropped_cols : list
        A list of column names that were removed.
    """
    corr_matrix = df.corr(method=method).abs()

    # Get upper triangle of covariance matrix since it is mirrored to prevent duplicates
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    if inplace:
        df.drop(columns=to_drop, inplace=True)
        return df, to_drop
    else:
        reduced_df = df.drop(columns=to_drop)
        return reduced_df, to_drop


def remove_stock_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    remove_current: bool = False,
    remove_lagged: bool = True,
    remove_aggr: bool = False,
    remove_subfeatures: bool = True,
    remove_static: bool = True,
    additional_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
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
    remove_aggr : bool, optional
        Whether to remove aggregate columns (idiovol, beta, betasq with their current variants) columns (default is True).
    remove_subfeatures : bool, optional
        Whether to remove 'subfeatures' columns (default is True).
    remove_static : bool, optional
        Whether to remove static nontemporal columns (default is True).
    additional_columns : list, optional
        List of additional specific columns to remove, regardless of the other flags.

    Returns
    -------
    tuple
        A tuple containing:
            pd.DataFrame
                DataFrame with the specified columns removed.
            dict[str,str]:
                Dictionary containing all the static nontemporal values.

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
    ]
    lagged_cols = [
        "mom12m",
        "chmom",
        "maxret",
        "mve",
        "dolvol",
    ]

    aggregate_cols = [
        "beta_current",
        "betasq_current",
        "idiovol_current",
        "beta",
        "betasq",
        "idiovol",
    ]

    subfeatures_cols = [
        "rolling_avg_3y_returns_weekly_by_month",
        "prices_monthly",
        "vol_sum_monthly",
        "shares_monthly",
        "zero_trading_days_count_monthly",
        "trading_days_count_monthly",
        "market_cap",
    ]
    static_columns = [
        "symbol",
        "sic_code_2",
        "sic_industry",
        "exchange_short_name",
        "exchange",
    ]

    # Get the static values as a single dict
    static_info = df[static_columns].iloc[0].to_dict()

    if columns is None:
        columns = []
        columns += current_cols if remove_current else []
        columns += lagged_cols if remove_lagged else []
        columns += subfeatures_cols if remove_subfeatures else []
        columns += static_columns if remove_static else []
        columns += aggregate_cols if remove_aggr else []
        columns += additional_columns if additional_columns is not None else []

    if columns:
        df = df.drop(columns=columns)

    return df, static_info
