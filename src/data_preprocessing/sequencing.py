from src.config.settings import PROCESSED_DIR, MACRO_DATA
from src.utils.disk_io import load_all_stocks
from src.utils.helpers import month_gap_diff
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


def create_sequences(
    df: pd.DataFrame,
    timesteps: int,
    target_column: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms a time-indexed DataFrame into sequences suitable for training sequence models (e.g., RNNs, LSTMs).

    For each sample, this function creates an input sequence of `timesteps` consecutive rows and assigns
    the target value as the `target_column` value of the row immediately following the sequence if passed.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame where rows represent time steps and columns represent features.
        The index is assumed to be time-based and will be sorted in ascending order.

    timesteps : int
        The number of time steps to include in each input sequence.

    target_column : str, optional
        The name of the column to be used as the prediction target. If None, only sequences are returned without targets.

    Returns
    -------
    tuple of np.ndarray
        - time : ndarray of shape (num_samples,)
            The timestamps corresponding to each target value `y`.

        - x : ndarray of shape (num_samples, timesteps, num_features)
            The input sequences.

        - y : ndarray of shape (num_samples,)
            The corresponding target values for each input sequence. If `target_column` is None, this will be None.

    Raises
    ------
    IndexError
        If the number of rows in `df` is less than or equal to `timesteps`, so that no valid sequence can be formed.

    Notes
    -----
    - This function converts the entire DataFrame to a NumPy array for speed.
    - Efficient memory allocation is used to handle large datasets.
    - The function assumes the DataFrame is sorted in time order (ascending), and will sort it if not.
    - Each `x[i]` corresponds to the sequence `df[i:i+timesteps]`, and `y[i]` is `df[i+timesteps][target_column]`.
    """

    # Sort by index assuming it is time-based
    df_sorted = df.sort_index(ascending=True)

    # Convert to NumPy for speed
    data = df_sorted.values

    if target_column is not None:
        target_idx = df_sorted.columns.get_loc(target_column)

    # Number of sequences we can generate
    num_samples = data.shape[0] - timesteps
    if num_samples <= 0:
        print("Number of samples less than timesteps")
        return None

    # Allocate memory for inputs and targets
    x = np.zeros((num_samples, timesteps, data.shape[1]))
    y = np.zeros((num_samples,)) if target_column is not None else None

    # Populate x and y
    for i in range(num_samples):
        x[i] = data[i : i + timesteps]
        if target_column is not None:
            y[i] = data[i + timesteps, target_idx]

    # Get time values corresponding to the target points
    time = np.array(df_sorted.index[timesteps:])

    return time, x, y


def create_sequences_all(
    macro_df, target_column, timesteps, input_directory=PROCESSED_DIR
):
    # TODO RESET this function
    """
    Generates sequences of features and corresponding target values for all stocks, including macroeconomic data.

    This function:
    - Loads preprocessed stock data from a specified directory.
    - Concatenates monthly, quarterly, annual stock features and subfeatures (excluding "month_latest_week"),
      along with provided macroeconomic features.
    - Constructs input sequences of shape (timesteps, num_features) and associated target values using
      a sliding window approach, for each stock independently.
    - Returns combined training data from all stocks.

    Parameters
    ----------
    macro_df : pandas.DataFrame
        A DataFrame containing macroeconomic features aligned by time index (e.g., month).

    target_column : str
        The name of the target column used for prediction.

    timesteps : int
        The number of past time steps to use as input features in each sequence.

    input_directory : str, optional
        The directory path where processed stock files are stored (default is `PROCESSED_DIR`).

    Returns
    -------
    tuple of np.ndarray
        - x_all : ndarray of shape (total_samples, timesteps, num_features)
            The complete array of input sequences from all stocks.

        - y_all : ndarray of shape (total_samples,)
            The corresponding target values for each input sequence.

    Notes
    -----
    - This function avoids pre-allocating NumPy arrays due to varying data availability across stocks;
      it uses Python lists and converts them to arrays at the end.
    - Each stock's features are processed independently to allow for differences in available data.
    - Feature concatenation includes: monthly, quarterly, annual stock features, selected subfeatures,
      and the same macroeconomic data applied to all stocks.
    - The function uses `create_sequences()` internally to perform sequence construction per stock.
    """

    stocks = load_all_stocks(input_directory)

    # Why not np.array? -> Because concatenation is computationally and memory expensive due to not knowing the shape of the array
    x_all = []
    y_all = []

    for stock in stocks:

        # Extract relevant features
        features = pd.DataFrame(stock["features"]["monthly"])
        features_quarterly = pd.DataFrame(stock["features"]["quarterly"])
        features_annual = pd.DataFrame(stock["features"]["annual"])

        # Include all monthly subfeatures except "month_latest_week"
        subfeatures = pd.DataFrame(
            {
                k: v
                for k, v in stock["subfeatures"]["monthly"].items()
                if k != "month_latest_week"
            }
        )

        # Concatenate the dfs
        features_conc = pd.concat(
            [features, features_quarterly, features_annual, subfeatures, macro_df],
            axis=1,
        )

        # Create array with shape (batch, timesteps, features) and add to the rest of the other stock sequences
        # to still have the shape (batch, timesteps, features)
        time, x, y = create_sequences(features_conc, target_column, timesteps)
        x_all.extend(x)
        y_all.extend(y)

    return np.array(x_all), np.array(y_all)


def split_sequences(
    time: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Splits time series data into training, validation, and test sets.

    This function divides the input sequences (`x`, `y`) and their corresponding time indices into
    training, validation, and testing subsets according to the specified proportions. The split is
    done sequentially, preserving the temporal order of the data.

    Parameters
    ----------
    time : array-like
        An array or list of time indices corresponding to each sequence sample.

    x : np.ndarray
        A NumPy array of input sequences with shape (num_samples, timesteps, num_features).

    y : np.ndarray
        A NumPy array of target values with shape (num_samples,).

    train_size : float, optional
        Proportion of the data to be used for training (default is 0.7).

    val_size : float, optional
        Proportion of the data to be used for validation (default is 0.15).
        The test set will use the remaining data.

    Returns
    -------
    tuple
        A tuple containing the following 9 elements:
        - time_train : array-like
        - time_val : array-like
        - time_test : array-like
        - x_train : np.ndarray
        - x_val : np.ndarray
        - x_test : np.ndarray
        - y_train : np.ndarray
        - y_val : np.ndarray
        - y_test : np.ndarray

    Notes
    -----
    - The remaining data after allocating the train and validation sets will be used for the test set.
    - The function assumes that `x`, `y`, and `time` are sorted in chronological order.
    - Copy this `time_train, time_val, time_test, x_train, x_val, x_test, y_train, y_val, y_test` to use.
    """

    train_size = int(len(x) * train_size)
    val_size = int(len(x) * val_size)

    x_train = x[:train_size]
    y_train = y[:train_size]
    time_train = time[:train_size]

    x_val = x[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]
    time_val = time[train_size : train_size + val_size]

    x_test = x[train_size + val_size :]
    y_test = y[train_size + val_size :]
    time_test = time[train_size + val_size :]

    return (
        time_train,
        time_val,
        time_test,
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    )


def scale_sequences(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scaler_class: type[TransformerMixin] = StandardScaler,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TransformerMixin,
    TransformerMixin,
]:
    """
    Scales input (X) and output (Y) sequences for training, validation, and test sets using the specified scaler.
    Fitting the scaler only on the training split to prevent data leakage.

    This function reshapes the input sequences to apply scaling across all timesteps and features,
    then reshapes them back to their original form. It also scales the output targets separately.

    Parameters
    ----------
    x_train : array-like of shape (n_samples, n_timesteps, n_features)
        Training input sequences.
    x_val : array-like of shape (n_samples, n_timesteps, n_features)
        Validation input sequences.
    x_test : array-like of shape (n_samples, n_timesteps, n_features)
        Test input sequences.
    y_train : array-like of shape (n_samples, 1) or (n_samples,)
        Training target values.
    y_val : array-like of shape (n_samples, 1) or (n_samples,)
        Validation target values.
    y_test : array-like of shape (n_samples, 1) or (n_samples,)
        Test target values.
    scaler_class : class, optional
        A scikit-learn scaler class (e.g., StandardScaler or MinMaxScaler). Default is StandardScaler.

    Returns
    -------
    tuple
        A tuple containing:
            - x_train_scaled, x_val_scaled, x_test_scaled : np.ndarray
                Scaled input sequences.
            - y_train_scaled, y_val_scaled, y_test_scaled : np.ndarray
                Scaled target values.
            - x_scaler : sklearn Scaler instance
                Fitted scaler used for the input sequences.
            - y_scaler : sklearn Scaler instance
                Fitted scaler used for the output targets.

    Notes
    ------
        - Copy the parameters `x_train, x_val, x_test, y_train, y_val, y_test`
        - Copy the returns `x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_scaler, y_scaler`
    """

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Flatten X to fit scaler on (samples * timesteps, features)
    x_scaler = scaler_class()
    x_train_reshaped = x_train.reshape(
        -1, x_train.shape[-1]
    )  # By having no limit on 1 dimension and number of features as 2nd dimension
    x_scaler.fit(x_train_reshaped)

    # Transform the reshaped x data and then revert to initial shape
    x_train_scaled = x_scaler.transform(x_train_reshaped).reshape(x_train.shape)
    x_val_scaled = x_scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(
        x_val.shape
    )
    x_test_scaled = x_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(
        x_test.shape
    )

    # Reshaped y data to ensure it is only 1 output
    y_scaler = scaler_class()
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Transform y data
    y_scaler.fit(y_train)
    y_train_scaled = y_scaler.transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    return (
        x_train_scaled,
        x_val_scaled,
        x_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
    )


def create_sequences_with_gaps(
    df: pd.DataFrame, target_column: str, timesteps: int, max_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create time sequences per symbol, skipping sequences where the gap between months exceeds max_gap.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a datetime index and a 'symbol' column.
    target_column : str
        The column to predict.
    timesteps : int
        Number of steps in each sequence.
    max_gap : int
        Maximum allowed gap (in months) between consecutive timestamps in a sequence.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        time, x, y arrays
    """
    sequences = []
    targets = []
    times = []

    for symbol, group in df.groupby("symbol"):
        group = group.sort_index(ascending=True)
        data = group.values
        target_idx = group.columns.get_loc(target_column)
        dates = group.index.to_list()

        for i in range(len(group) - timesteps):
            date_seq = dates[i : i + timesteps + 1]

            # Check all consecutive gaps in the sequence
            valid = True
            for j in range(timesteps):
                gap = month_gap_diff(date_seq[j], date_seq[j + 1])
                if gap > max_gap:
                    valid = False
                    break

            if not valid:
                continue

            sequences.append(data[i : i + timesteps])
            targets.append(data[i + timesteps, target_idx])
            times.append(date_seq[-1])

    return (
        np.array(times),
        np.array(sequences),
        np.array(targets),
    )
