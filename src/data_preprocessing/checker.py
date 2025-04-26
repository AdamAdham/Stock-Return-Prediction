def check_nans_only_at_top(df):
    result = {}
    for col in df.columns:
        first_valid_label = df[col].first_valid_index()
        if first_valid_label is None:
            result[col] = True  # All NaNs
            continue
        # Convert label to positional index
        pos = df.index.get_loc(first_valid_label)
        non_nan_after = df[col].iloc[pos:].isna().any()
        result[col] = not non_nan_after
    return result
