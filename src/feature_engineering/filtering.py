def check_stock_validity(stock, invalid):
    """
    Validates the presence of essential financial data for a given stock.

    Returns:
        bool: True if all required fields are present and not None, False otherwise.
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


def filter_stock(stock, filtered):
    # TODO use splits to actually see if the price is low or high and remove penny stocks and see
    # Add other filters here
    price_lower = 0.01
    for eod in stock["eod"]:
        if eod["price"] < price_lower:
            print(f"{stock['symbol']} had a daily price less that {price_lower}")
            filtered[stock["symbol"]] = f"daily price less that {price_lower}"
            return False

    return True
