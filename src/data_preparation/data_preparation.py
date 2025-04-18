from src.config.settings import PROCESSED_DIR, MACRO_DATA
from src.utils.json_io import load_all_stocks
import numpy as np
import pandas as pd


def format_macro():
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

    return macro


def create_sequences(df, target_column, timesteps):
    """
    Transforms a time-indexed DataFrame into sequences suitable for training sequence models (e.g., RNNs, LSTMs).

    For each sample, this function creates an input sequence of `timesteps` consecutive rows and assigns
    the target value as the `target_column` value of the row immediately following the sequence.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame where rows represent time steps and columns represent features.
        The index is assumed to be time-based and will be sorted in ascending order.

    target_column : str
        The name of the column to be used as the prediction target.

    timesteps : int
        The number of time steps to include in each input sequence.

    Returns
    -------
    tuple of np.ndarray
        - x : ndarray of shape (num_samples, timesteps, num_features)
            The input sequences.

        - y : ndarray of shape (num_samples,)
            The corresponding target values for each input sequence.

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

    # Sort by index assuming it is time based
    df_sorted = df.sort_index(ascending=True)

    # Convert to NumPy for speed
    data = df_sorted.values  # Convert entire DataFrame to NumPy array
    target_idx = df_sorted.columns.get_loc(target_column)

    # Getting number of samples
    num_samples = data.shape[0] - timesteps
    if num_samples <= 0:
        raise IndexError(
            f"Cannot create a sample with timestep={timesteps} because length of df={len(df)}"
        )

    # Preallocates memory for the x and y arrays — crucial for performance on large datasets.
    x = np.zeros(
        (num_samples, timesteps, data.shape[1])
    )  # shape = (num of samples, num of timesteps, num of features)
    y = np.zeros((num_samples,))

    # Populates the x and y arrays
    for i in range(num_samples):
        x[i] = data[i : i + timesteps]
        y[i] = data[i + timesteps, target_idx]

    return x, y


def create_sequences_all(
    macro_df, target_column, timesteps, input_directory=PROCESSED_DIR
):
    stocks = load_all_stocks(input_directory)

    # Why not np.array? -> Because concatenation is computationally and memory expensive due to not knowing the shape of the array
    x_all = []
    y_all = []

    for stock in stocks:
        # TODO get feature ID, SIC_2 and any other important thing

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
        x, y = create_sequences(features_conc, target_column, timesteps)
        x_all.extend(x)
        y_all.extend(y)

    return np.array(x_all), np.array(y_all)
