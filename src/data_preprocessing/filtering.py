import pandas as pd
import numpy as np

from src.utils.helpers import month_gap_diff


def remove_short_isolated_sequences(
    df: pd.DataFrame, max_gap: int = 6, min_seq: int = 12
) -> pd.DataFrame:
    """
    Remove sequences of dates that are isolated by gaps > max_gap on both sides and shorter than min_seq.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index (monthly) and a 'symbol' column.
    max_gap : int
        Max allowed gap in months before/after a sequence.
    min_seq : int
        Minimum length of sequence to keep, if isolated by large gaps.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with short isolated sequences removed.
    """
    df = df.copy()
    keep_mask = [True] * len(df)
    print("min seq", min_seq, "max", max_gap)
    grouped = df.groupby("symbol")
    keep_index = 0
    for symbol, group in grouped:
        print("-" * 100)
        print(symbol)
        dates = group.index.to_list()
        n = len(dates)

        i = 0
        while i < n:
            seq_start = i
            while i + 1 < n and (month_gap_diff(dates[i], dates[i + 1]) <= max_gap):
                i += 1
            seq_end = i

            # Check gaps before and after
            # If they are at the start or end of the list then assign 0 since there is no gap
            gap_before = (
                month_gap_diff(dates[seq_start - 1], dates[seq_start])
                if seq_start > 0
                else 0
            )
            gap_after = (
                month_gap_diff(dates[seq_end], dates[seq_end + 1])
                if seq_end + 1 < n
                else 0
            )

            seq_length = seq_end - seq_start + 1

            print(
                "seq_start",
                seq_start,
                "    ",
                "seq_end",
                seq_end,
                "   ",
                "gap_before",
                gap_before,
                "    ",
                "gap_after",
                gap_after,
                "     ",
                "seq_length",
                seq_length,
                "  ",
                "total length",
                n,
            )

            if seq_length < min_seq:
                print("seq<min")
                print("make local false", seq_start, "to", seq_end)
                print(
                    "make global false",
                    seq_start + keep_index,
                    "to",
                    seq_end + keep_index,
                )
                for j in range(seq_start, seq_end + 1):
                    keep_mask[j + keep_index] = False

            i = seq_end + 1

        keep_index += n

    return keep_mask


def invalidate_weeks_by_valid_months(
    weekly_returns_df, month_to_weeks, symbol_to_valid_months
):
    """
    Invalidate weekly returns for each symbol in weekly_returns_df based on whether the week
    belongs to a valid month for that symbol.

    Parameters:
    - weekly_returns_df: pd.DataFrame
        Rows: week keys in "YYYY-WW" format
        Columns: stock symbols
        Values: weekly returns

    - month_to_weeks: dict
        Format: { "YYYY-MM": ["YYYY-WW", ...] }

    - symbol_to_valid_months: dict
        Format: { "SYMBOL": ["YYYY-MM", ...] }

    Returns:
    - pd.DataFrame: same shape as weekly_returns_df but with invalid weeks set to NaN
    """
    # Build reverse mapping: symbol -> set of valid weeks
    symbol_to_valid_weeks = {}
    for symbol, valid_months in symbol_to_valid_months.items():
        valid_weeks = set()
        for month in valid_months:
            valid_weeks.update(month_to_weeks.get(month, []))
        symbol_to_valid_weeks[symbol] = valid_weeks

    # Create a copy to avoid modifying original
    result_df = weekly_returns_df.copy()

    # Invalidate weeks
    for symbol in result_df.columns:
        if symbol not in symbol_to_valid_weeks:
            # Invalidate all weeks if no valid months
            result_df[symbol] = np.nan
        else:
            valid_weeks = symbol_to_valid_weeks[symbol]
            mask = ~result_df.index.isin(valid_weeks)
            result_df.loc[mask, symbol] = np.nan

    return result_df
