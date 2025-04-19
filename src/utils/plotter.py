import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_model_history(history):
    """
    Plots the training and validation loss (Mean Squared Error) and Mean Absolute Error (MAE)
    from the model's training history.

    Args:
        history (tensorflow.python.keras.callbacks.History): The history object returned from model training.

    Displays:
        - A plot of training vs. validation loss (MSE).
        - A plot of training vs. validation Mean Absolute Error (MAE).
    """

    # Plot loss (Mean Squared Error)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    # Plot Mean Absolute Error (MAE)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["mean_absolute_error"], label="Training MAE")
    plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()
    plt.title("Training vs Validation MAE")
    plt.show()


def plot_pred_real_timeseries_train_val_test(
    time,
    time_train,
    time_val,
    time_test,
    y_train,
    y_val,
    y_test,
    y_pred_train,
    y_pred_val,
    y_pred_test,
    inverse_scaler=None,
    fig_size=(16, 8),
    x_ticks=20,
):
    """
    Plots actual and predicted time series values for the training, validation, and test sets.

    This function visualizes how well the model's predictions align with actual values across different
    dataset splits. It allows optional inverse scaling of the values for interpretation in original scale.

    Parameters
    ----------
    time : array-like
        Full sequence of time indices used to generate the x-axis tick labels.

    time_train : array-like
        Time indices corresponding to the training set.

    time_val : array-like
        Time indices corresponding to the validation set.

    time_test : array-like
        Time indices corresponding to the test set.

    y_train : np.ndarray
        Actual target values for the training set.

    y_val : np.ndarray
        Actual target values for the validation set.

    y_test : np.ndarray
        Actual target values for the test set.

    y_pred_train : np.ndarray
        Predicted target values for the training set.

    y_pred_val : np.ndarray
        Predicted target values for the validation set.

    y_pred_test : np.ndarray
        Predicted target values for the test set.

    inverse_scaler : callable, optional
        Function to inverse-transform predicted and actual values (e.g., from a scaler).
        If None, no transformation is applied.

    fig_size : tuple, optional
        Figure size for the plot (default is (16, 8)).

    x_ticks : int, optional
        Number of x-axis ticks to display (default is 20).

    Returns
    -------
    None
        Displays a matplotlib plot comparing actual vs predicted values over time.
    """

    if inverse_scaler != None:
        y_train = inverse_scaler(y_train)
        y_val = inverse_scaler(y_val)
        y_test = inverse_scaler(y_test)
        y_pred_train = inverse_scaler(y_pred_train)
        y_pred_val = inverse_scaler(y_pred_val)
        y_pred_test = inverse_scaler(y_pred_test)

    plt.figure(figsize=fig_size)

    # Plot actual values
    plt.plot(time_train, y_train, label="Train (Actual)", color="blue")
    plt.plot(time_val, y_val, label="Validation (Actual)", color="orange")
    plt.plot(time_test, y_test, label="Test (Actual)", color="green")

    # Plot predicted values
    plt.plot(
        time_train,
        y_pred_train,
        "--",
        label="Train (Predicted)",
        color="blue",
        alpha=0.6,
    )
    plt.plot(
        time_val,
        y_pred_val,
        "--",
        label="Validation (Predicted)",
        color="orange",
        alpha=0.6,
    )
    plt.plot(
        time_test, y_pred_test, "--", label="Test (Predicted)", color="green", alpha=0.6
    )

    tick_positions = np.linspace(
        0, len(time) - 1, x_ticks, dtype=int
    )  # Create array of indices to get values that are evenly spaces
    tick_labels = time[tick_positions]  # Get corresponding labels

    plt.xticks(tick_positions, tick_labels, rotation=90)
    plt.legend()
    plt.show()


def plot_pred_real_timeseries(
    time,
    y,
    y_pred,
    color,
    color_pred,
):
    """
    Plots the actual vs. predicted time series values.

    Args:
        time (array-like): Time indices for the data.
        y (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        color (str): Color for the actual values line.
        color_pred (str): Color for the predicted values line.

    Displays:
        - A time series plot comparing actual vs. predicted values.
    """

    # Plot actual values
    plt.plot(time, y, label="Train (Actual)", color=color)

    # Plot predicted values
    plt.plot(
        time,
        y_pred,
        "--",
        label="Train (Predicted)",
        color=color_pred,
        alpha=0.6,
    )

    tick_positions = np.linspace(
        0, len(time) - 1, 10, dtype=int
    )  # Create array of indices to get values that are evenly spaces
    tick_labels = time[tick_positions]  # Get corresponding labels

    plt.xticks(tick_labels, rotation=90)
    plt.legend()
    plt.show()


def basic_eda(
    df, categorical_features=[], hist_bins=30, correlation=True, distribution=True
):
    """
    Perform basic exploratory data analysis (EDA) on a given DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    categorical_features (array[string]): names of the categorical features in the dataframe
    """
    print("\n--- Dataset Overview ---")

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print()
    print(f"Types: \n{df.dtypes}")

    print("\n--- Summary Statistics ---")
    print(df.describe(include="all"))

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    if correlation:
        print("\n--- Correlation Matrix ---")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.show()

    if distribution:
        print("\n--- Feature Distributions ---")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        n_cols = 3
        n_rows = (
            len(numeric_cols) + n_cols - 1
        ) // n_cols  # ceiling of len(numeric_cols) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, bins=hist_bins, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Density")

        # Remove unused subplots if there are any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    if len(categorical_features) > 0:
        print("\n--- Categorical Feature Counts ---")
        for col in categorical_features:
            plt.figure(figsize=(8, 4))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.show()
