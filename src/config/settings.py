from pathlib import Path


# Paths
# Dynamically resolve the project root, assuming this file is src/config/settings.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

# Stocks
STOCKS_DIR = DATA_DIR / "stocks"
RAW_DIR = STOCKS_DIR / "raw"
RAW_NEW_DIR = STOCKS_DIR / "raw_new"
PROCESSED_DIR = STOCKS_DIR / "processed"
PROCESSED_NEW_DIR = STOCKS_DIR / "processed_new"
DATAFRAMES_DIR = STOCKS_DIR / "dataframes"
DATAFRAMES_NEW_DIR = STOCKS_DIR / "dataframes_new"

# Tests
TEST_DIR = DATA_DIR / "tests"
AGGREGATE_DIR_TEST = TEST_DIR / "aggregate"
RAW_DIR_TEST = TEST_DIR / "raw"
PROCESSED_DIR_TEST = TEST_DIR / "processed"
DATAFRAMES_DIR_TEST = TEST_DIR / "dataframes"
REFERENCE_DIR = DATA_DIR / "reference"
EDA_DIR = DATA_DIR / "eda"

# Metadata
METADATA_DIR = DATA_DIR / "metadata"

# Model Results
MODEL_RESULTS_DIR = PROJECT_ROOT / "model_results"

# Macroeconomic data
MACRO_DIR = DATA_DIR / "macroeconomics"
MACRO_DATA = MACRO_DIR / "macroeconomics_data.csv"

# Constant Values
ENABLE_TIMING = False
RETURN_ROUND_TO = None
