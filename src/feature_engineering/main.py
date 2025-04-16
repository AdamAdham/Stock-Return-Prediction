from src.feature_engineering.generator.generator import (
    enrich_stocks_with_features,
    enrich_stocks_with_aggregate_features,
)


def main():
    # Populate raw stocks with features
    aggregate_stats, status = enrich_stocks_with_features()

    # Populate processed stock (containing features) with aggregate stats, which are
    # indmom, beta, betasq and their current variants
    enrich_stocks_with_aggregate_features(
        aggregate_stats["indmom"], aggregate_stats["market_returns_weekly"]
    )


if __name__ == "__main__":
    main()
