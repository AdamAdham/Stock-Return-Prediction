def compare_feature(feature_name, feature1, feature2):
    for month1, month2 in zip(feature1, feature2):
        value1 = feature1[month1]
        value2 = feature2[month2]
        if value1 != value2:
            print(
                f"feature {feature_name}, month: {month1}, value: {value1}, value2: {value2}"
            )


def compare_features(stock1, stock2, debug=False):
    """
    Compare the features of two stock dictionaries across multiple time intervals.

    Parameters:
        stock1 (dict): First stock dictionary with nested feature structure.
        stock2 (dict): Second stock dictionary with the same structure.
        debug (bool): If True, prints detailed comparison steps.

    Returns:
        None. Prints mismatches and missing features.
    """

    # Checking 2 stocks have identical features
    intervals = ["weekly", "monthly", "quarterly", "annual"]
    for interval in intervals:
        if debug:
            print(f"Checking {interval} interval")

        # Union of keys to ensure all possible features are checked
        keys1 = stock1["features"].get(interval, {})
        keys2 = stock2["features"].get(interval, {})
        all_keys = set(keys1.keys()).union(keys2.keys())

        # Loop through all the keys and intervals checking each dict
        for feature_key in all_keys:
            if debug:
                print(f"Comparing {feature_key}")

            feature1 = stock1["features"][interval].get(feature_key, None)
            feature2 = stock2["features"][interval].get(feature_key, None)

            # Check if both keys are not present so identical
            if feature1 is None and feature2 is None:
                continue
            # Check if a key is in a dict and absent in the other
            if feature1 is None:
                print(f"{feature_key} is not in stock1")
                continue
            if feature2 is None:
                print(f"{feature_key} is not in stock2")
                continue

            compare_feature(feature_key, feature1, feature2)

        if debug:
            print("-" * 100)


def compare_stocks(stocks1, stocks2, debug=False):

    for stock1, stock2 in zip(stocks1, stocks2):
        if debug:
            print(f"Comparing {stock1["symbol"]}")

        compare_features(stock1, stock2)

        if debug:
            print("-" * 100)


def returns(new, old):
    return (new - old) / old * 100
