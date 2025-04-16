from pathlib import Path


# Paths
# Dynamically resolve the project root, assuming this file is src/config/settings.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

# Stocks
STOCKS_DIR = DATA_DIR / "stocks"
RAW_DIR = STOCKS_DIR / "raw"
PROCESSED_DIR = STOCKS_DIR / "processed"

# Tests
TEST_DIR = DATA_DIR / "tests"
AGGREGATE_DIR_TEST = TEST_DIR / "aggregate"
RAW_DIR_TEST = TEST_DIR / "raw"
PROCESSED_DIR_TEST = TEST_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

# Metadata
METADATA_DIR = DATA_DIR / "metadata"

# Constant Values
ENABLE_TIMING = False
RETURN_ROUND_TO = None
