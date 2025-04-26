import traceback
import pandas as pd

from src.config.settings import MACRO_DATA, PROCESSED_DIR, DATAFRAMES_DIR
from src.utils.disk_io import load_all_stocks
from sklearn.preprocessing import StandardScaler


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


def fill_nan(df):
    for col in df.columns:
        # Fill the initial NaN values with zeros, since no information can be put that will not be data leakage
        first_valid_index = df[col].first_valid_index()
        integer_index = df.index.get_loc(first_valid_index)

        # If there are nans at the start
        if integer_index > 0:
            df[col] = df[col].fillna(0, limit=integer_index)

        # For remaining NaN values
        df[col] = df[col].fillna(df[col].ewm(halflife=5, adjust=False).mean())
    return df
