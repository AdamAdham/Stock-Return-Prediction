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
