from datetime import datetime
from src.utils.information import get_sic_industry_names


def remove_duplicates_and_sort_by_date(market_cap_data):
    """
    Removes duplicate dates from a list of market cap dictionaries and sorts them by date.

    Parameters
    ----------
    market_cap_data : list of dict
        A list of dictionaries where each dictionary contains a 'date' key with a string date value.

    Returns
    -------
    list of dict
        A sorted list of dictionaries with unique dates.
    """
    seen_dates = set()  # To track unique dates
    unique_market_cap_data = []

    # Loop through the list and filter out duplicate dates
    for data in market_cap_data:
        date_str = data["date"]
        date = datetime.strptime(
            date_str, "%Y-%m-%d"
        ).date()  # Convert date string to date object

        # If the date hasn't been seen before, add it to the list and mark it as seen
        if date not in seen_dates:
            unique_market_cap_data.append(data)
            seen_dates.add(date)
        # else:
        #     print(f"Duplicate date found and removed: {date_str}") TODO

    # Sort the list of dictionaries by the 'date' key from latest to oldest
    unique_market_cap_data.sort(
        key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"), reverse=True
    )

    return unique_market_cap_data


def get_stock_profiles(stocks_sic_codes, stock_list):
    """
    Matches stocks in sic_codes with additional details from stock_list using the symbol.

    This function updates each stock in the `stocks_sic_codes` list by adding relevant information
    from `stock_list`, such as exchange details, SIC code industry, and stock type.

    The function also filters by:
        Excluding stocks not in both lists
        Excluding stocks having SIC code not in available SIC codes
        Only including stock types
        Excluding same stock symbols on different exchanges (using .)

    Parameters
    ----------
    stocks_sic_codes : list of dict
        A list of dictionaries containing stock information, each having at least a 'symbol' key
        and a 'sicCode' key.

    stock_list : list of dict
        A list of dictionaries containing detailed information about stocks, including 'symbol',
        'exchange', 'exchangeShortName', and 'type'.

    Returns
    -------
    tuple
        A tuple containing:
        - `updated_stocks_sic_codes` (list of dict): A list of dictionaries with updated stock information,
          including SIC code industry, exchange details, and stock type.
        - `not_in_stock_symbols` (list of dict): A list of stocks from `stocks_sic_codes` that could
          not be matched with the `stock_list` based on the symbol or other criteria.

    Notes
    -----
    - The SIC code is expected to be a string, and the first 2 characters are used to match the
      industry name from the predefined SIC industries.
    - The function ensures that stocks with duplicate symbols are removed from the result.
    """

    # Create a lookup dictionary for quick symbol lookup with {symbol: stock_list}
    stock_lookup = {stock["symbol"]: stock for stock in stock_list}
    sic_industries = get_sic_industry_names()

    updated_stocks_sic_codes = []
    not_in_stock_symbols = []

    for stock in stocks_sic_codes:
        symbol = stock.get("symbol")
        sic_2 = stock["sicCode"][0:2]
        if (
            symbol in stock_lookup
            and sic_2 in sic_industries.keys()
            and stock_lookup[symbol]["type"] == "stock"
            and "." not in symbol
        ):
            # There exists the symbol in the stock details (so can get its information) and
            # SIC code is in the classified SIC codes by the US government and
            # The investment type is a "stock" and
            # The symbol does not have . since this indicated it is just the same symbol but in a different exchange
            stock_details = stock_lookup[symbol]
            # Merge the stock stock_list with exchange details
            stock.update(
                {
                    "sicCode_2": sic_2,
                    "sicIndustry": sic_industries.get(sic_2, "Unknown"),
                    "sicDivision": get_sic_division(sic_2),
                    "exchange": stock_details["exchange"],
                    "exchangeShortName": stock_details["exchangeShortName"],
                    "type": stock_details["type"],
                }
            )
            updated_stocks_sic_codes.append(stock)
        else:
            not_in_stock_symbols.append(stock)

    # Identify duplicate symbols by counting their occurrences
    symbol_counts = {}
    for stock in updated_stocks_sic_codes:
        symbol = stock["symbol"]  # Assuming 'symbol' is the key for stock symbol
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    # Filter out stocks with duplicate symbols
    updated_stocks_sic_codes = [
        stock
        for stock in updated_stocks_sic_codes
        if symbol_counts[stock["symbol"]] == 1
    ]

    return updated_stocks_sic_codes, not_in_stock_symbols


def get_sic_division(sic_2):
    """
    Maps a 2-digit Standard Industry Classification (SIC) code to its corresponding industry division.

    This function takes a 2-digit SIC code and returns the corresponding industry division as per the
    mapping defined in the function. The SIC code is used to classify industries based on the type of
    economic activity.

    Data from https://www.osha.gov/data/sic-manual

    Parameters
    ----------
    sic_2 : str
        A 2-digit string representing the SIC code (e.g., "01", "10", "20").

    Returns
    -------
    str
        A string describing the corresponding industry division for the given SIC code.
    """

    industry_mapping = {
        "01": "Division A: Agriculture, Forestry, And Fishing",
        "02": "Division A: Agriculture, Forestry, And Fishing",
        "07": "Division A: Agriculture, Forestry, And Fishing",
        "08": "Division A: Agriculture, Forestry, And Fishing",
        "09": "Division A: Agriculture, Forestry, And Fishing",
        "10": "Division B: Mining",
        "12": "Division B: Mining",
        "13": "Division B: Mining",
        "14": "Division B: Mining",
        "15": "Division C: Construction",
        "16": "Division C: Construction",
        "17": "Division C: Construction",
        "20": "Division D: Manufacturing",
        "21": "Division D: Manufacturing",
        "22": "Division D: Manufacturing",
        "23": "Division D: Manufacturing",
        "24": "Division D: Manufacturing",
        "25": "Division D: Manufacturing",
        "26": "Division D: Manufacturing",
        "27": "Division D: Manufacturing",
        "28": "Division D: Manufacturing",
        "29": "Division D: Manufacturing",
        "30": "Division D: Manufacturing",
        "31": "Division D: Manufacturing",
        "32": "Division D: Manufacturing",
        "33": "Division D: Manufacturing",
        "34": "Division D: Manufacturing",
        "35": "Division D: Manufacturing",
        "36": "Division D: Manufacturing",
        "37": "Division D: Manufacturing",
        "38": "Division D: Manufacturing",
        "39": "Division D: Manufacturing",
        "40": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "41": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "42": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "43": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "44": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "45": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "46": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "47": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "48": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "49": "Division E: Transportation, Communications, Electric, Gas, And Sanitary Services",
        "50": "Division F: Wholesale Trade",
        "51": "Division F: Wholesale Trade",
        "52": "Division G: Retail Trade",
        "53": "Division G: Retail Trade",
        "54": "Division G: Retail Trade",
        "55": "Division G: Retail Trade",
        "56": "Division G: Retail Trade",
        "57": "Division G: Retail Trade",
        "58": "Division G: Retail Trade",
        "59": "Division G: Retail Trade",
        "60": "Division H: Finance, Insurance, And Real Estate",
        "61": "Division H: Finance, Insurance, And Real Estate",
        "62": "Division H: Finance, Insurance, And Real Estate",
        "63": "Division H: Finance, Insurance, And Real Estate",
        "64": "Division H: Finance, Insurance, And Real Estate",
        "65": "Division H: Finance, Insurance, And Real Estate",
        "67": "Division H: Finance, Insurance, And Real Estate",
        "70": "Division I: Services",
        "72": "Division I: Services",
        "73": "Division I: Services",
        "75": "Division I: Services",
        "76": "Division I: Services",
        "78": "Division I: Services",
        "79": "Division I: Services",
        "80": "Division I: Services",
        "81": "Division I: Services",
        "82": "Division I: Services",
        "83": "Division I: Services",
        "84": "Division I: Services",
        "86": "Division I: Services",
        "87": "Division I: Services",
        "88": "Division I: Services",
        "89": "Division I: Services",
        "91": "Division J: Public Administration",
        "92": "Division J: Public Administration",
        "93": "Division J: Public Administration",
        "94": "Division J: Public Administration",
        "95": "Division J: Public Administration",
        "96": "Division J: Public Administration",
        "97": "Division J: Public Administration",
        "99": "Division J: Public Administration",
    }

    return industry_mapping[sic_2]
