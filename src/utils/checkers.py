import numpy as np


def compare_string_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    only_in_list1 = list(set1 - set2)
    only_in_list2 = list(set2 - set1)
    in_both = list(set1 & set2)

    return only_in_list1, only_in_list2, in_both


def compare_series(series1, series2):
    mask = ~np.isclose(series1["cov"], series2["cov_manual"], equal_nan=True)
    return mask


def arrays_equal_with_tolerance(a, b, atol=1e-8, rtol=1e-5):
    if a.shape != b.shape:
        return False

    # Compare NaNs at the same positions
    nan_mask = np.isnan(a) & np.isnan(b)

    # Compare the rest using allclose
    not_nan_mask = ~np.isnan(a) & ~np.isnan(b)

    # If one has NaN and the other doesn't, it's not equal
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False

    return np.allclose(a[not_nan_mask], b[not_nan_mask], atol=atol, rtol=rtol)
