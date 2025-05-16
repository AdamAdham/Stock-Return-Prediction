import pandas as pd

from src.config.settings import STOCKS_DIR
from src.data_preprocessing.preprocessing import (
    scale_df,
    split_data_dates,
    scale_train_val_test,
)
from src.data_preprocessing.sequencing import create_sequences_train_val_test

from src.data_preprocessing.macro import get_macro_data
from src.data_preprocessing.feature_selection import remove_highly_correlated_features
from src.utils.information import get_sic_mapping

from sklearn.preprocessing import StandardScaler


def get_final_dataset(
    train_end_date="2019-06",
    val_end_date="2022-07",
    test_end_date="2025-01",
    sample_size: int | None = None,
    scaler_class=StandardScaler,
    stock_timesteps: int = 12,
    macro_timesteps: int = 12,
):
    raw_df = pd.read_csv(
        STOCKS_DIR / "stock_final.csv",
        index_col=0,
    )
    raw_df.drop(["sp_annual", "agr_annual"], inplace=True, axis=1)

    raw_df = raw_df.rename(
        columns={
            "mom12m_current": "mom12m",
            "chmom_current": "chmom",
            "maxret_current": "maxret",
            "mve_current": "mve",
            "dolvol_current": "dolvol",
        }
    )

    macro = get_macro_data()
    macro_corr, remove_cols = remove_highly_correlated_features(macro)

    numeric_stock_cols = [
        "mom1m",
        "mom12m",
        "mom36m",
        "chmom",
        "maxret",
        "turn",
        "std_turn",
        "mve",
        "dolvol",
        "ill",
        "retvol",
        "ep_quarterly",
        "sp_quarterly",
        "agr_quarterly",
        "ep_annual",
        "beta",
        "betasq",
        "idiovol",
        "indmom",
    ]

    total_numeric_cols = numeric_stock_cols + list(macro_corr.columns)
    sic_map = get_sic_mapping(raw_df["sic_code_2"])

    raw_df["sic_code_2_mapped"] = raw_df["sic_code_2"].map(sic_map)

    ## Formatting Macro
    macro_final, _, _ = scale_df(
        macro_corr, "Index", train_end_date=train_end_date, scaler_class=scaler_class
    )

    ## Formatting Stock
    df = raw_df.merge(macro_final, on="month", how="left")

    if sample_size is not None:
        df_sample_symbols = (
            df["symbol"].drop_duplicates().sample(n=sample_size, random_state=42)
        )
        df_sample = df[df["symbol"].isin(df_sample_symbols)]
    else:
        df_sample = df
    train_time, train_data, val_time, val_data, test_time, test_data = split_data_dates(
        df_sample,
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        test_end_date=test_end_date,
    )

    train_scaled, val_scaled, test_scaled, scaler_x, scaler_y = scale_train_val_test(
        train_data[numeric_stock_cols],
        val_data[numeric_stock_cols],
        test_data[numeric_stock_cols],
        scaler_class=StandardScaler,
        target_col="mom1m",
    )

    train_data.loc[:, numeric_stock_cols] = train_scaled[numeric_stock_cols].values
    val_data.loc[:, numeric_stock_cols] = val_scaled[numeric_stock_cols].values
    test_data.loc[:, numeric_stock_cols] = test_scaled[numeric_stock_cols].values
    (
        x_train_stock,
        train_sic_mapped_stock,
        x_train_macro,
        y_train_stock,
        x_val_stock,
        val_sic_mapped_stock,
        x_val_macro,
        y_val_stock,
        x_test_stock,
        test_sic_mapped_stock,
        x_test_macro,
        y_test_stock,
    ) = create_sequences_train_val_test(
        train_data,
        val_data,
        test_data,
        stock_timesteps=stock_timesteps,
        macro_timesteps=macro_timesteps,
    )

    return (
        train_time,
        x_train_stock,
        train_sic_mapped_stock,
        x_train_macro,
        y_train_stock,
        val_time,
        x_val_stock,
        val_sic_mapped_stock,
        x_val_macro,
        y_val_stock,
        test_time,
        x_test_stock,
        test_sic_mapped_stock,
        x_test_macro,
        y_test_stock,
    )
