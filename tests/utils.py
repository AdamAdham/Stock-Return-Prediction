def compare_feature(feature_name, feature1, feature2):
    for month1, month2 in zip(feature1, feature2):
        value1 = feature1[month1]
        value2 = feature2[month2]
        if value1 != value2:
            print(
                f"feature {feature_name}, month: {month1}, value: {value1}, value2: {value2}"
            )


def compare_features(stock1, stock2):
    # Checking 2 stocks yield identical result
    for feature_key in stock1["features"]["monthly"]:
        feature = stock1["features"]["monthly"][feature_key]
        feature_perf = stock2["features"]["monthly"][feature_key]
        compare_feature(feature_key, feature, feature_perf)


def returns(new, old):
    return (new - old) / old * 100
