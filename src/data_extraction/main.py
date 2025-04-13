import json
from src.data_extraction.utils import get_stock_profiles
from src.data_extraction.stock_info import get_all_stock_info
from src.utils.json_io import read_json
from src.config.settings import REFERENCE_DIR


def main():
    # Get all stocks available at FMP
    # Use https://financialmodelingprep.com/api/v3/stock/list?apikey=API_KEY
    stock_list_path = REFERENCE_DIR / "stock_list.json"
    stock_list = read_json(stock_list_path)

    # Get all stocks that FMP provides an  SIC code for
    # Use https://financialmodelingprep.com/stable/all-industry-classification?apikey=API_KEY
    stocks_sic_codes_path = REFERENCE_DIR / "stocks_sic_codes.json"
    stocks_sic_codes = read_json(stocks_sic_codes_path)

    # Filtering to get all stocks that fit the criteria (given in the function's docstring)
    stock_profiles, _ = get_stock_profiles(stocks_sic_codes, stock_list)

    # Populate the stocks with eod, financial statements, market caps, and shares
    # These stocks are saved on disk since it takes large storage so cannot be stored in memory
    success_fails_stocks = get_all_stock_info(stock_profiles)


if __name__ == "__main__":
    main()
