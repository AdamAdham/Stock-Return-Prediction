import matplotlib.pyplot as plt
import numpy as np


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
    model,
    time,
    time_train,
    X_train,
    y_train,
    time_val,
    X_val,
    y_val,
    time_test,
    X_test,
    y_test,
    inverse_scaler=None,
    fig_size=(16, 8),
    x_ticks=20,
):
    """
    Plots the actual and predicted time series values for training, validation, and test datasets.

    Args:
        model (tensorflow.keras.Model): The trained model used for predictions.
        time (array-like): The complete time index for reference.
        time_train (array-like): Time indices for the training set.
        X_train (array-like): Input features for the training set.
        y_train (array-like): Actual target values for the training set.
        time_val (array-like): Time indices for the validation set.
        X_val (array-like): Input features for the validation set.
        y_val (array-like): Actual target values for the validation set.
        time_test (array-like): Time indices for the test set.
        X_test (array-like): Input features for the test set.
        y_test (array-like): Actual target values for the test set.
        inverse_scaler (function, optional): Function to inverse transform the predicted values.

    Displays:
        - A time series plot comparing actual vs. predicted values for train, validation, and test sets.
    """

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

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


def plot_real_pred(y, y_pred, graph_title="Actual vs Predicted"):
    """
    Plots the actual vs. predicted values in a scatter plot.

    Args:
        y (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        graph_title (str, optional): Title for the plot. Default is "Actual vs Predicted".

    Displays:
        - A scatter plot comparing actual vs. predicted values.
        - A 45-degree reference line for perfect predictions.
    """

    # Plot actual vs. predicted values
    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_pred, alpha=0.5, label="Predictions")
    plt.plot(
        [np.nanmin(y), np.nanmax(y)],
        [np.nanmin(y), np.nanmax(y)],
        "r",
        linestyle="--",
        label="Perfect Fit",
    )  # 45-degree line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(graph_title)
    plt.legend()
    plt.show()
