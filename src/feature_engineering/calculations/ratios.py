from datetime import datetime

from src.feature_engineering.utils import calculate_return

from src.config.settings import RETURN_ROUND_TO


def calculate_ep_sp(
    income_statements: list[dict], market_caps: list[dict]
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """
    Calculate the Earnings to Market Capitalization (EP) and Sales to Market Capitalization (SP) ratios for each fiscal period.

    The EP ratio is calculated as the net income divided by the market capitalization for a given fiscal period.
    The SP ratio is calculated as the revenue divided by the market capitalization for a given fiscal period.

    An important distinction is the difference between "date" (date of financial statement itself) and "fillingDate" the
    date that financial statement is publicly available. So the ratio will use the market cap at "date" ; however, the data
    will be available after the "fillingDate"

    Parameters
    ----------
    income_statements : list of dict
        A list of dictionaries, where each dictionary contains financial data for a fiscal period, including:
            - "date" (str): The date the income statement was made (e.g., "2024-09-28").
            - "fillingDate" (str): The date the income statement was filed (e.g., "2024-09-28").
            - "netIncome" (float): The net income for the fiscal period.
            - "revenue" (float): The revenue for the fiscal period.

    market_caps : list of dict
        A list of dictionaries, where each dictionary contains market capitalization data for a specific date, including:
            - "date" (str): The date of the market capitalization (e.g., "2024-09-28").
            - "marketCap" (float): The market capitalization on the given date.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - ep (dict): A dictionary the keys are month identifiers (in "YYYY-MM" format), and the value is the Earnings to Market Cap ratio (EP) for that period.
        - sp (dict): A dictionary the keys are month identifiers (in "YYYY-MM" format), and the value is the Sales to Market Cap ratio (SP) for that period.
    """

    market_cap_index = 0
    ep = {}
    sp = {}

    income_statements_sorted = sorted(
        income_statements,
        key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"),
        reverse=True,
    )

    for income_statement in income_statements_sorted:
        # Parse dates
        date = income_statement["date"]  # date of income statement
        date_filling = datetime.strptime(income_statement["fillingDate"], "%Y-%m-%d")
        month_filling = f"{date_filling.year}-{date_filling.month:02d}"

        # Get values from income statement
        net_income = income_statement["netIncome"]
        revenue = income_statement["revenue"]
        market_cap = None

        # Get market cap at fiscal period date
        while market_cap_index < len(market_caps):
            market_date = market_caps[market_cap_index]["date"]

            # If market cap date is same as income statement use it directly
            if date == market_date:
                market_cap = market_caps[market_cap_index]["marketCap"]
                break
            elif date > market_date:
                # Average market cap if the fiscal period date is between two market dates
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

        # Check if not first occurrence of month
        # If so then this is the older income statement, so just ignore since it is
        # outdated data. (eg: AAPL:"fillingDate": "2006-12-29")
        if month_filling in ep:
            continue

        # The financial statements go further than the earliest marketcap (explained in most cases by company going public after the dates of financial statements releases)
        if date < market_caps[-1]["date"]:
            ep[month_filling] = None
            sp[month_filling] = None

        # Prevent division by zero
        elif market_cap is None or market_cap == 0:
            print(
                f"Calculating ep/sp, the market_cap is None or 0 for {income_statement["symbol"]}"
            )
            print(
                f"income_statement_date:{date}, market_date: {market_date}, market_cap: {market_cap}, market_cap_index: {market_cap_index}"
            )
            ep[month_filling] = None
            sp[month_filling] = None
        else:
            ep[month_filling] = net_income / market_cap
            sp[month_filling] = revenue / market_cap

    return ep, sp


def calculate_agr(balance_sheet: list[dict]) -> dict[str, float | None]:
    """
    Calculate the annual/quarterly percent change in total assets.

    Parameters
    ----------
    balance_sheet : list of dict
        A list of dictionaries, where each dictionary contains balance sheet data for a fiscal period, including:
            - "date" (str): The date of the balance sheet (e.g., "2024-09-28") assuming latest to earliest balance sheet dates.
            - "fillingDate" (str): The date the income statement was filed (e.g., "2024-09-28").
            - "totalAssets" (float): The total assets for the fiscal period.

    Returns
    -------
    dict
        A dictionary where the keys are month identifiers (in "YYYY-MM" format) and the value is the annual/quarterly percent change in total assets.
    """

    percent_change = {}

    balance_sheet_sorted = sorted(
        balance_sheet,
        key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"),
        reverse=True,
    )

    # For each period
    for i in range(len(balance_sheet_sorted) - 1):
        current_period_data = balance_sheet_sorted[i]
        previous_period_data = balance_sheet_sorted[i + 1]

        # Current filling date
        date_filling = datetime.strptime(current_period_data["fillingDate"], "%Y-%m-%d")
        month_filling = f"{date_filling.year}-{date_filling.month:02d}"

        # Getting assets
        current_total_assets = current_period_data["totalAssets"]
        previous_total_assets = previous_period_data["totalAssets"]

        # Check if first occurrence of month
        # If not then this is the older balance sheet, so just ignore since it is
        # outdated data. (eg: AAPL:"fillingDate": "2006-12-29")
        if month_filling not in percent_change:
            # Calculate percent change in total assets
            percent_change[month_filling] = calculate_return(
                current_total_assets,
                previous_total_assets,
                round_to=RETURN_ROUND_TO,
            )

    # Assign None to the remaining period
    date_filling = datetime.strptime(balance_sheet[-1]["fillingDate"], "%Y-%m-%d")
    month_filling = f"{date_filling.year}-{date_filling.month:02d}"
    percent_change[month_filling] = None

    return percent_change
