from pathlib import Path


# Paths
# Dynamically resolve the project root, assuming this file is src/config/settings.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "stocks" / "raw"
PROCESSED_DIR = DATA_DIR / "stocks" / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

STOCKS_STATS_TEST = DATA_DIR / "stocks_stats_test"
STOCKS_STATS_TEST = DATA_DIR / "stocks_test"

# Constant Values
RETURN_ROUND_TO = None
