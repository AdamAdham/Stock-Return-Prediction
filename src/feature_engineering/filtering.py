def check_stock_validity(stock: dict, invalid: dict[str, str]) -> bool:
    """
    Validates the presence of essential financial data for a given stock.

    This function checks if the stock's essential financial data, such as its EOD data, market cap,
    income statements, and balance sheets (both annual and quarterly), are available and not None or empty.

    Parameters
    ----------
    stock : dict
        A dictionary containing stock data, with keys like "symbol", "eod", "market_cap", and nested financial data
        such as annual and quarterly income statements and balance sheets.

    invalid : dict
        A dictionary that maps the stock symbol to an error message if any required field is missing or invalid.

    Returns
    -------
    bool
        Returns True if all required fields are present and valid (not None or empty). Returns False if any
        required field is missing or invalid, and the error message is added to the `invalid` dictionary.
    """

    required_fields = {
        "EOD data": stock["eod"],
        "Market cap": stock["market_cap"],
        "Annual income statement": stock["financials_annual"]["income_statement"],
        "Quarterly income statement": stock["financials_quarterly"]["income_statement"],
        "Annual balance sheet": stock["financials_annual"]["balance_sheet"],
        "Quarterly balance sheet": stock["financials_quarterly"]["balance_sheet"],
    }

    symbol = stock["symbol"]

    for field_name, value in required_fields.items():
        if value is None or len(value) == 0:
            invalid[symbol] = f"{field_name} is missing"
            return False

    return True


def filter_stock(stock: dict, filtered: dict[str, str]):
    # TODO use splits to actually see if the price is low or high and remove penny stocks and see
    # Add other filters here
    price_lower = 0.01
    for eod in stock["eod"]:
        if eod["price"] < price_lower:
            print(f"{stock['symbol']} had a daily price less that {price_lower}")
            filtered[stock["symbol"]] = f"daily price less that {price_lower}"
            return False

    return True
