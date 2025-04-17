from collections import defaultdict
from src.feature_engineering.utils import calculate_return

from src.config.settings import RETURN_ROUND_TO


def calculate_ep_sp_annual(income_statement_annual, market_caps):
    """
    Calculate the Earnings to Market Capitalization (EP) and Sales to Market Capitalization (SP) ratios for each fiscal year.

    The EP ratio is calculated as the net income divided by the market capitalization for a given fiscal year.
    The SP ratio is calculated as the revenue divided by the market capitalization for a given fiscal year.

    An important distinction is the difference between "date" (date of financial statement itself) and "acceptedData" the
    date that financial statement is publicly available. So the ratio will use the market cap at "date" ; however, the data
    will be available after the "acceptedDate"

    Parameters
    ----------
    income_statement_annual : list of dict
        A list of dictionaries, where each dictionary contains financial data for a fiscal year, including:
            - "date" (str): The date of the income statement (e.g., "2024-09-28").
            - "calendarYear" (int): The fiscal year (e.g., 2024).
            - "netIncome" (float): The net income for the fiscal year.
            - "revenue" (float): The revenue for the fiscal year.

    market_caps : list of dict
        A list of dictionaries, where each dictionary contains market capitalization data for a specific date, including:
            - "date" (str): The date of the market capitalization (e.g., "2024-09-28").
            - "marketCap" (float): The market capitalization on the given date.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - ep (dict): A dictionary where the key is the fiscal year (e.g., "2024"), and the value is the Earnings to Market Cap ratio (EP) for that year.
        - sp (dict): A dictionary where the key is the fiscal year (e.g., "2024"), and the value is the Sales to Market Cap ratio (SP) for that year.

    Notes
    -----
    - The function assumes that `income_statement_annual` and `market_caps` are sorted latest to earliest.
    - The function matches each fiscal year's data with the market capitalization on the corresponding fiscal year end date.
    """

    market_cap_index = 0
    ep = {}
    sp = {}

    for income_statement in income_statement_annual:

        date_fiscal_year = income_statement["date"]
        fiscal_year = income_statement["calendarYear"]

        net_income = income_statement["netIncome"]
        revenue = income_statement["revenue"]
        market_cap = None

        # Get market cap at fiscal year date
        while market_cap_index < len(market_caps):
            market_date = market_caps[market_cap_index]["date"]

            if date_fiscal_year == market_date:
                market_cap = market_caps[market_cap_index]["marketCap"]
                break
            elif date_fiscal_year > market_date:
                # Average market cap if the fiscal year date is between two market dates
                if (
                    market_cap_index > 0
                ):  # Make sure you're not accessing market_cap_index-1 when it's 0
                    market_cap = (
                        market_caps[market_cap_index - 1]["marketCap"]
                        + market_caps[market_cap_index]["marketCap"]
                    ) / 2
                else:
                    # If market_cap_index is 0, just use the current market cap (you could handle this differently if you want)
                    market_cap = market_caps[market_cap_index]["marketCap"]
                break

            market_cap_index += 1

        # The financial statements go further than the earliest marketcap (explained in most cases by company going public after the dates of financial statements releases)
        if date_fiscal_year < market_caps[-1]["date"]:
            ep[fiscal_year] = None
            sp[fiscal_year] = None

        # Prevent division by zero
        elif market_cap is None or market_cap == 0:
            print(
                f"While calculating quarterly ep/sp, the market_cap is None or 0 for {income_statement["symbol"]}"
            )
            print(
                f"income_statement_date:{date_fiscal_year}, market_date: {market_date}, market_cap: {market_cap}, market_cap_index: {market_cap_index}"
            )
            ep[fiscal_year] = None
            sp[fiscal_year] = None
        else:
            ep[fiscal_year] = net_income / market_cap
            sp[fiscal_year] = revenue / market_cap

    return ep, sp


def calculate_ep_sp_quarterly(income_statement_quarterly, market_caps):
    """
    Calculate the Earnings to Market Capitalization (EP) and Sales to Market Capitalization (SP) ratios for each fiscal year.

    The EP ratio is calculated as the net income divided by the market capitalization for a given fiscal year.
    The SP ratio is calculated as the revenue divided by the market capitalization for a given fiscal year.

    An important distinction is the difference between "date" (date of financial statement itself) and "acceptedData" the
    date that financial statement is publicly available. So the ratio will use the market cap at "date" ; however, the data
    will be available after the "acceptedDate"

    Parameters
    ----------
    income_statement_quarterly : list of dict
        A list of dictionaries, where each dictionary contains financial data for a fiscal year, including:
            - "date" (str): The date of the income statement (e.g., "2024-09-28").
            - "calendarYear" (int): The fiscal year (e.g., 2024).
            - "netIncome" (float): The net income for the fiscal year.
            - "revenue" (float): The revenue for the fiscal year.

    market_caps : list of dict
        A list of dictionaries, where each dictionary contains market capitalization data for a specific date, including:
            - "date" (str): The date of the market capitalization (e.g., "2024-09-28").
            - "marketCap" (float): The market capitalization on the given date.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - ep (dict): A dictionary where the key is the fiscal year (e.g., "2024"), and the value is the Earnings to Market Cap ratio (EP) for that year.
        - sp (dict): A dictionary where the key is the fiscal year (e.g., "2024"), and the value is the Sales to Market Cap ratio (SP) for that year.

    Notes
    -----
    - The function assumes that `income_statement_quarterly` and `market_caps` are sorted latest to earliest.
    - The function matches each fiscal year's data with the market capitalization on the corresponding fiscal year end date.
    """

    market_cap_index = 0
    ep = {}
    sp = {}
    date_of_quarter = defaultdict(list)

    for income_statement in income_statement_quarterly:

        date = income_statement["date"]
        year = income_statement["calendarYear"]
        quarter = income_statement["period"]

        year_quarter = f"{year}-{quarter}"  # In this form '2024-Q1'

        net_income = income_statement["netIncome"]
        revenue = income_statement["revenue"]
        market_cap = None

        # Get market cap at quarter date
        while market_cap_index < len(market_caps):
            market_date = market_caps[market_cap_index]["date"]

            # If there exists a market cap date with same date as income statement use that
            if date == market_date:
                market_cap = market_caps[market_cap_index]["marketCap"]
                break
            # If not just average the 2 market cap values closest to the income statement date
            elif date > market_date:
                # Average market cap if the quarter date is between two market dates
                if (
                    market_cap_index > 0
                ):  # Make sure you're not accessing market_cap_index-1 when it's 0
                    market_cap = (
                        market_caps[market_cap_index - 1]["marketCap"]
                        + market_caps[market_cap_index]["marketCap"]
                    ) / 2
                else:
                    # If market_cap_index is 0, just use the current market cap (you could handle this differently if you want)
                    market_cap = market_caps[market_cap_index]["marketCap"]
                break

            market_cap_index += 1

        # The financial statements go further than the earliest marketcap (explained in most cases by company going public after the dates of financial statements releases)
        if date < market_caps[-1]["date"]:
            ep[year_quarter] = None
            sp[year_quarter] = None

        # Prevent division by zero
        elif market_cap is None or market_cap == 0:
            print(
                f"While calculating quarterly ep/sp, the market_cap is None or 0 for {income_statement["symbol"]}"
            )
            print(
                f"income_statement_date:{date}, market_date: {market_date}, market_cap: {market_cap}, market_cap_index: {market_cap_index}"
            )
            ep[year_quarter] = None
            sp[year_quarter] = None
        else:
            ep[year_quarter] = net_income / market_cap
            sp[year_quarter] = revenue / market_cap

    return ep, sp


def calculate_agr_annual(balance_sheet_annual):
    """
    Calculate the annual percent change in total assets for each year.

    Parameters
    ----------
    balance_sheet_annual : list of dict
        A list of dictionaries, where each dictionary contains balance sheet data for a fiscal year, including:
            - "date" (str): The date of the balance sheet (e.g., "2024-09-28") assuming latest to earliest balance sheet dates.
            - "calendarYear" (int): The fiscal year (e.g., 2024).
            - "totalAssets" (float): The total assets for the fiscal year.

    Returns
    -------
    dict
        A dictionary where the key is the fiscal year (e.g., "2024") and the value is the annual percent change in total assets for that year.
    """

    percent_change = {}

    # Calculate percent change in total assets
    for i in range(len(balance_sheet_annual) - 1):
        current_year_data = balance_sheet_annual[i]
        previous_year_data = balance_sheet_annual[i + 1]

        current_year = current_year_data["calendarYear"]

        current_total_assets = current_year_data["totalAssets"]
        previous_total_assets = previous_year_data["totalAssets"]

        percent_change[current_year] = calculate_return(
            current_total_assets,
            previous_total_assets,
            round_to=RETURN_ROUND_TO,
        )

    # Assign None to the remaining year
    last_year = balance_sheet_annual[-1]["calendarYear"]
    percent_change[last_year] = None

    return percent_change


def calculate_agr_quarterly(balance_sheet_quarterly):
    """
    Calculate the quarterly percent change in total assets for each balance sheet period.

    Parameters
    ----------
    balance_sheet_quarterly : list of dict
        A list of dictionaries representing quarterly balance sheet data, **ordered from latest to earliest**.
        Each dictionary should contain:
            - "date" (str): The date of the balance sheet (e.g., "2024-09-28").
            - "calendarYear" (int): The fiscal year (e.g., 2024).
            - "period" (str): The fiscal quarter (e.g., "Q1", "Q2", "Q3", or "Q4").
            - "totalAssets" (float): The total assets reported for the period.

    Returns
    -------
    dict
        A dictionary mapping each period (e.g., "2024-Q1") to the percent change in total assets
        compared to the previous quarter. The earliest quarter in the input will have a value of None
        since there's no prior data to compare against.

    Notes
    -----
    - Assumes that the input list is sorted from most recent to oldest.
    """

    percent_change = {}

    # Calculate percent change in total assets
    for i in range(len(balance_sheet_quarterly) - 1):
        current_year_data = balance_sheet_quarterly[i]
        previous_year_data = balance_sheet_quarterly[i + 1]

        current_year = current_year_data["calendarYear"]
        quarter = current_year_data["period"]

        current_year_quarter = f"{current_year}-{quarter}"  # In this form '2024-Q1'

        current_total_assets = current_year_data["totalAssets"]
        previous_total_assets = previous_year_data["totalAssets"]

        percent_change[current_year_quarter] = calculate_return(
            current_total_assets, previous_total_assets, round_to=RETURN_ROUND_TO
        )

    # Assign None to the remaining year
    last_year = balance_sheet_quarterly[-1]["calendarYear"]
    last_quarter = balance_sheet_quarterly[-1]["period"]
    percent_change[f"{last_year}-{last_quarter}"] = None

    return percent_change
