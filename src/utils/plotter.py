import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl


def plot_real_vs_pred_all_sets(
    y_train,
    y_train_pred,
    y_val,
    y_val_pred,
    y_test,
    y_test_pred,
    save_path="real_vs_pred.pdf",
):
    # Force disable LaTeX text rendering to avoid errors
    mpl.rcParams["text.usetex"] = False

    # Aesthetic settings
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    sns.set_style("whitegrid")

    # Create figure
    plt.figure(figsize=(6.5, 4.5))

    # Plot: Predicted vs Real
    plt.scatter(
        y_train,
        y_train_pred,
        color="#1f77b4",
        alpha=0.5,
        label="Train",
        edgecolor="k",
        s=30,
    )
    plt.scatter(
        y_val,
        y_val_pred,
        color="#2ca02c",
        alpha=0.5,
        label="Validation",
        edgecolor="k",
        s=30,
    )
    plt.scatter(
        y_test,
        y_test_pred,
        color="#ff7f0e",
        alpha=0.6,
        label="Test",
        edgecolor="k",
        s=30,
    )

    # Plot reference line y = x
    min_val = min(np.min(y_train), np.min(y_val), np.min(y_test))
    max_val = max(np.max(y_train), np.max(y_val), np.max(y_test))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="Ideal")

    # Axis labels and formatting
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("Predicted vs. Actual Values Across Sets")
    plt.legend(loc="upper left", frameon=False)

    # Save
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.show()


def plot_model_history(
    history, metrics=["loss", "mean_absolute_error"], style="ggplot"
):
    """
    Plots the training and validation metrics from the model's training history.

    Args:
        history (tensorflow.python.keras.callbacks.History): The history object returned from model training.
        metrics (list of str): List of metrics to plot (default includes "loss" and "mean_absolute_error").

    Displays:
        - A plot for each metric specified in the metrics list.
    """
    plt.style.use(style)
    print(history)
    for metric in metrics:
        if metric in history.history:  # Check if the metric is available in history
            plt.figure(figsize=(8, 5))
            plt.plot(history.history[metric], label=f"Training {metric.capitalize()}")
            plt.plot(
                history.history[f"val_{metric}"],
                label=f"Validation {metric.capitalize()}",
            )
            plt.xlabel("Epochs")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.title(f"Training vs Validation {metric.capitalize()}")
            plt.show()
        else:
            print(f"Warning: Metric '{metric}' not found in the history object.")


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
    extra_plots=None,
    inverse_transform=None,
    fig_size=(16, 8),
    x_ticks=20,
    style="ggplot",
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

    inverse_transform : callable, optional
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

    Notes:
        - Copy `time, time_train, time_val, time_test, y_train, y_val, y_test, y_pred_train, y_pred_val, y_pred_test` for your parameters
    """
    plt.style.use(style)

    if inverse_transform is not None:
        # Since inverse_transform expects shape (n_samples, n_features), while current is (n_samples,)
        y_train = inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_val = inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_test = inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_train = inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
        y_pred_val = inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
        y_pred_test = inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

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

    # Extra plots to add eg: mean
    if extra_plots is not None:
        for name, plot_dict in extra_plots.items():
            x = plot_dict["x"]
            y = plot_dict["y"]
            line_style = plot_dict["line_style"]
            color = plot_dict["color"]
            alpha = plot_dict["alpha"]

            plt.plot(x, y, line_style, label=name, color=color, alpha=alpha)

    tick_positions = np.linspace(
        0, len(time) - 1, x_ticks, dtype=int
    )  # Create array of indices to get values that are evenly spaces
    tick_labels = time[tick_positions]  # Get corresponding labels

    plt.xticks(tick_positions, tick_labels, rotation=90)
    plt.legend()
    plt.show()


def plot_pred_real_timeseries(time, y, y_pred, color, color_pred, style="ggplot"):
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
    plt.style.use(style)

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
    df,
    categorical_features=[],
    hist_bins=30,
    correlation=True,
    correlation_method="pearson",
    correlation_mean=False,
    distribution=True,
    distribution_cols: int = 3,
    distribution_min_max=False,
    style="ggplot",
):
    """
    Perform basic exploratory data analysis (EDA) on a given DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to analyze.
    categorical_features (array[string]): names of the categorical features in the dataframe
    """
    plt.style.use(style)

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
        if correlation_mean:
            corr_matrix = mean_correlation(df)
        else:
            corr_matrix = df.corr(method=correlation_method, numeric_only=True)

        # Adjust figure size based on number of columns
        num_cols = len(corr_matrix.columns)
        fig_size = max(10, num_cols * 0.6)  # Auto scale width

        print("\n--- Correlation Matrix ---")
        plt.figure(figsize=(fig_size, fig_size * 0.75))  # Wider for more columns
        heatmap = sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.5},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title("Feature Correlation Matrix", fontsize=14)
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9
        )
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.show()

    if distribution:
        print("\n--- Feature Distributions ---")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        n_cols = distribution_cols
        n_rows = (
            len(numeric_cols) + n_cols - 1
        ) // n_cols  # ceiling of len(numeric_cols) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, bins=hist_bins, ax=axes[i])
            if distribution_min_max:
                plt.xlim(df[col].min(), df[col].max())
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


def plot_real_pred(y, y_pred, graph_title="Actual vs Predicted"):
    # Plot actual vs. predicted values
    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_pred, alpha=0.5, label="Predictions")
    plt.plot(
        [y.min(), y.max()], [y.min(), y.max()], "r", linestyle="--", label="Perfect Fit"
    )  # 45-degree line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(graph_title)
    plt.legend()
    plt.show()


def mean_correlation(df, correlation_methods=["pearson", "kendall", "spearman"]):
    """
    Calculate the mean correlation matrix across multiple methods (Pearson, Kendall, Spearman).

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe for which correlations are calculated.

    correlation_methods : list of str
        The list of correlation methods to use (default: ["pearson", "kendall", "spearman"]).

    Returns
    -------
    pandas.DataFrame
        A DataFrame representing the average correlation values across methods.
    """
    correlation_matrices = []
    correlation_methods = ["pearson", "kendall", "spearman"]

    # Calculate correlation matrices for each method
    for method in correlation_methods:
        corr_matrix = df.corr(method=method, numeric_only=True)
        correlation_matrices.append(corr_matrix)

    final_dic = {}
    for col in correlation_matrices[0].columns:
        series0 = correlation_matrices[0][col]
        series1 = correlation_matrices[1][col]
        series2 = correlation_matrices[2][col]

        # Calculate the mean across the three series
        final_dic[col] = (series0 + series1 + series2) / 3

    # Convert the dictionary into a DataFrame for easier visualization
    final_df = pd.DataFrame(final_dic)

    return final_df


def plot_r2_broken_axis(model_results, save_path="r2_scores_broken_axis.pdf"):
    """
    Plots train and validation R² scores with a broken y-axis for better visibility,
    and shows x-axis labels between the two subplots.
    """

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 18,
            "font.size": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    models = [m["model"] for m in model_results]
    train_pos = [m["train_pos"] for m in model_results]
    val_pos = [m["val_pos"] for m in model_results]
    train_scores = [m["train_r2"] for m in model_results]
    val_scores = [m["val_r2"] for m in model_results]

    x = np.arange(len(models))
    bar_width = 0.35

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
    )

    ax1.bar(
        train_pos, train_scores, width=0.4, color="#ff7f0e"
    )  # You can change width if needed

    tick_offset = 0.2  # adjust this based on bar width
    adjusted_ticks = [x + tick_offset for x in train_pos]

    ax1.set_xticks(adjusted_ticks)
    ax1.set_xticklabels(models, rotation=0, ha="center")

    ax1.set_ylabel("Training $R^2$")
    ax2.set_ylabel("Validation $R^2$")

    ax2.bar(
        val_pos, val_scores, width=0.4, color="#1f77b4"
    )  # You can change width if needed

    ax2.set_xticks([])

    # Limits and spines
    ax1.set_xlim(0.5, 5.9)
    ax2.set_xlim(0.5, 5.9)

    ax1.set_ylim(0.75, 1)
    ax2.set_ylim(-2.3, 0)

    # Diagonal break lines
    kwargs = dict(
        marker=[(-1, -1), (1, 1)],
        markersize=13,
        linestyle="none",
        color="#707070",
        mec="#707070",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # Title and legend
    # fig.suptitle("R² Scores by Model (Broken Y-Axis)", fontsize=14)
    # fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.savefig(save_path, dpi=300, format="pdf")
    plt.show()
