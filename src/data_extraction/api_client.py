import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
import os
from src.config.api_config import BASE_URL, LEGACY_BASE_URL, LEGACY_BASE_URL_V4

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")


class APIClient:
    def __init__(self, session=None):
        self.session = session or requests.Session()

    def get_api_call(self, extension, legacy=True, v4=False, separator=True):
        """
        Creates a GET API call to the base URL with the given extension, handling errors appropriately.

        This function constructs a complete API URL based on the provided extension and optional parameters,
        and makes a GET request to retrieve data from the API. It handles errors such as network issues
        or non-200 status codes by printing error messages and returning None if the request fails.

        Parameters
        ----------
        extension : str
            The path and query parameters to be appended to the base URL to form the complete API URL.

        legacy : bool, optional
            If True (default), the legacy FMP API base URL is used. If False, the latest API version's base URL is used.

        v4 : bool, optional
            If True, the API call uses the version 4 of the legacy FMP API. Default is False.

        separator : bool, optional
            If True (default), uses "&" as the separator for additional query parameters. If False, uses "?".

        Returns
        -------
        dict or None
            Returns the JSON response as a dictionary if the request is successful (status code 200).
            Returns None if an error occurs, such as a network issue or if the status code is not 200.

        Notes
        -----
        - If a network-related error occurs, an error message is printed, and None is returned.
        - If the API response has a non-200 status code, the error message (if available) is printed.
        """

        seper = "&" if separator else "?"
        base = BASE_URL if not legacy else LEGACY_BASE_URL_V4 if v4 else LEGACY_BASE_URL
        url = base + extension + f"{seper}apikey={API_KEY}"

        try:
            response = self.session.get(url)
            data = response.json()
            if response.status_code == 200:
                return data
            else:
                print("Error:", response.status_code)
                print(
                    f'Error Message: "{data.get('Error Message', 'No Error Message')}"'
                )
                return None
        except requests.exceptions.RequestException as e:
            # Handle any network-related errors or exceptions
            print("Error:", e)
            return None

    def get_income_statement(self, symbol, interval="annual"):
        """
        Fetches the income statement for a given stock symbol and time interval.

        Parameters
        ----------
        symbol : str
            The stock ticker symbol for which the income statement is requested.
        interval : str, optional
            The reporting period for the income statement, either "quarter" or "annual".
            Default is "quarter".

        Returns
        -------
        dict:
            The JSON response as a dictionary if the request is successful (status code 200).
        None:
            If an error occurs, such as a network issue or a non-200 status code.

        Notes
        -----
        - This function constructs the appropriate API extension and calls `get_api_call` to fetch the data.
        - If the API request fails, an error message is printed, and None is returned.

        Example
        -------
        >>> get_income_statement("AAPL", "yearly")
        [
            {
                'date': '2024-09-28',
                'symbol': 'AAPL',
                'reportedCurrency': 'USD',
                'cik': '0000320193',
                'filingDate': '2024-11-01',
                'acceptedDate': '2024-11-01 06:01:36',
                'fiscalYear': '2024',
                'period': 'FY',
                'revenue': 391035000000,
                'costOfRevenue': 210352000000,
                'grossProfit': 180683000000,
                'researchAndDevelopmentExpenses': 31370000000,
                'generalAndAdministrativeExpenses': 0,
                'sellingAndMarketingExpenses': 0,
                'sellingGeneralAndAdministrativeExpenses': 26097000000,
                'otherExpenses': 0,
                'operatingExpenses': 57467000000,
                'costAndExpenses': 267819000000,
                'netInterestIncome': 0,
                'interestIncome': 0,
                'interestExpense': 0,
                'depreciationAndAmortization': 11445000000,
                'ebitda': 134661000000,
                'ebit': 123216000000,
                'nonOperatingIncomeExcludingInterest': 0,
                'operatingIncome': 123216000000,
                'totalOtherIncomeExpensesNet': 269000000,
                'incomeBeforeTax': 123485000000,
                'incomeTaxExpense': 29749000000,
                'netIncomeFromContinuingOperations': 93736000000,
                'netIncomeFromDiscontinuedOperations': 0,
                'otherAdjustmentsToNetIncome': 0,
                'netIncome': 93736000000,
                'netIncomeDeductions': 0,
                'bottomLineNetIncome': 93736000000,
                'eps': 6.11,
                'epsDiluted': 6.08,
                'weightedAverageShsOut': 15343783000,
                'weightedAverageShsOutDil': 15408095000
            },
            {
                'date': '2023-09-30',
                'symbol': 'AAPL',
                'reportedCurrency': 'USD',
                'cik': '0000320193',
                'filingDate': '2023-11-03',
                'acceptedDate': '2023-11-02 18:08:27',
                'fiscalYear': '2023',
                'period': 'FY',
                'revenue': 383285000000,
                'costOfRevenue': 214137000000,
                'grossProfit': 169148000000,
                'researchAndDevelopmentExpenses': 29915000000,
                'generalAndAdministrativeExpenses': 0,
                'sellingAndMarketingExpenses': 0,
                'sellingGeneralAndAdministrativeExpenses': 24932000000,
                'otherExpenses': 382000000,
                'operatingExpenses': 54847000000,
                'costAndExpenses': 268984000000,
                'netInterestIncome': -183000000,
                'interestIncome': 3750000000,
                'interestExpense': 3933000000,
                'depreciationAndAmortization': 11519000000,
                'ebitda': 125820000000,
                'ebit': 114301000000,
                'nonOperatingIncomeExcludingInterest': 0,
                'operatingIncome': 114301000000,
                'totalOtherIncomeExpensesNet': -565000000,
                'incomeBeforeTax': 113736000000,
                'incomeTaxExpense': 16741000000,
                'netIncomeFromContinuingOperations': 96995000000,
                'netIncomeFromDiscontinuedOperations': 0,
                'otherAdjustmentsToNetIncome': 0,
                'netIncome': 96995000000,
                'netIncomeDeductions': 0,
                'bottomLineNetIncome': 96995000000,
                'eps': 6.16,
                'epsDiluted': 6.13,
                'weightedAverageShsOut': 15744231000,
                'weightedAverageShsOutDil': 15812547000
            }
        ]

        """

        params = {"period": interval}
        extension = f"income-statement/{symbol}?" + urlencode(params)

        return self.get_api_call(extension)

    def get_balance_sheet(self, symbol, interval="annual"):
        """
        Fetches the balance sheet for a given stock symbol and time interval.

        Parameters
        -------
        symbol : str
            The stock ticker symbol for which the balance sheet is requested.
        interval : str, optional
            The reporting period for the balance sheet, either "quarter" or "annual".
            Default is "quarter".

        Returns
        -------
        dict or None
            - The JSON response as a dictionary if the request is successful (status code 200).
            - None if an error occurs, such as a network issue or a non-200 status code.

        Notes
        -------
        - This function constructs the appropriate API extension and calls `get_api_call` to fetch the data.
        - If the API request fails, an error message is printed, and None is returned.

        TODO
        -------
        - Check the bulk version for optimized data retrieval.

        Example
        -------
        >>> get_balance_sheet("AAPL", "yearly")
        [
            {
                'date': '2024-09-28',
                'symbol': 'AAPL',
                'reportedCurrency': 'USD',
                'cik': '0000320193',
                'fillingDate': '2024-11-01',
                'acceptedDate': '2024-11-01 06:01:36',
                'calendarYear': '2024',
                'period': 'FY',
                'cashAndCashEquivalents': 29943000000,
                'shortTermInvestments': 35228000000,
                'cashAndShortTermInvestments': 65171000000,
                'netReceivables': 66243000000,
                'inventory': 7286000000,
                'otherCurrentAssets': 14287000000,
                'totalCurrentAssets': 152987000000,
                'propertyPlantEquipmentNet': 45680000000,
                'goodwill': 0,
                'intangibleAssets': 0,
                'goodwillAndIntangibleAssets': 0,
                'longTermInvestments': 91479000000,
                'taxAssets': 19499000000,
                'otherNonCurrentAssets': 55335000000,
                'totalNonCurrentAssets': 211993000000,
                'otherAssets': 0,
                'totalAssets': 364980000000,
                'accountPayables': 68960000000,
                'shortTermDebt': 20879000000,
                'taxPayables': 26601000000,
                'deferredRevenue': 8249000000,
                'otherCurrentLiabilities': 51703000000,
                'totalCurrentLiabilities': 176392000000,
                'longTermDebt': 96548000000,
                'deferredRevenueNonCurrent': 0,
                'deferredTaxLiabilitiesNonCurrent': 0,
                'otherNonCurrentLiabilities': 35090000000,
                'totalNonCurrentLiabilities': 131638000000,
                'otherLiabilities': 0,
                'capitalLeaseObligations': 10798000000,
                'totalLiabilities': 308030000000,
                'preferredStock': 0,
                'commonStock': 83276000000,
                'retainedEarnings': -19154000000,
                'accumulatedOtherComprehensiveIncomeLoss': -7172000000,
                'othertotalStockholdersEquity': 0,
                'totalStockholdersEquity': 56950000000,
                'totalEquity': 56950000000,
                'totalLiabilitiesAndStockholdersEquity': 364980000000,
                'minorityInterest': 0,
                'totalLiabilitiesAndTotalEquity': 364980000000,
                'totalInvestments': 126707000000,
                'totalDebt': 106629000000,
                'netDebt': 76686000000,
                'link': 'https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm',
                'finalLink': 'https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm'
            },
            {
                ...
            }
        ]

        """

        params = {"period": interval}
        extension = f"balance-sheet-statement/{symbol}?" + urlencode(params)

        return self.get_api_call(extension)

    def get_market_cap(self, symbol, start_date=None, end_date=None, limit=None):
        """
        Fetches the historical market capitalization for a given stock symbol within a specified date range.

        Maximum 5000 records per request

        Parameters
        ----------
        symbol : str
            The stock ticker symbol for which the market capitalization data is requested.

        start_date : str, optional
            The start date for the market capitalization data in the format 'YYYY-MM-DD'.
            If not provided, the earliest available data will be returned.

        end_date : str, optional
            The end date for the market capitalization data in the format 'YYYY-MM-DD'.
            If not provided, the latest available data will be returned.

        limit : int, optional
            The maximum number of historical records to retrieve. Default is None, which retrieves all available records.

        Returns
        -------
        list[dict]
            A list of dictionaries, where each dictionary contains:
            - 'symbol' : str -> The stock ticker symbol.
            - 'date' : str -> The date of the market capitalization record.
            - 'marketCap' : int -> The market capitalization value in USD.

        None
            If an error occurs, such as a network issue or an invalid response from the API.

        Notes
        -----
        - This function constructs the appropriate API extension and calls `get_api_call` to fetch the data.
        - The API response includes historical market capitalization records sorted by date.
        - If the API request fails, an error message is printed, and None is returned.

        Example
        -------
        >>> get_market_cap("AAPL", "2025-03-26", "2025-03-28", limit=3)
        [
            {'symbol': 'AAPL', 'date': '2025-03-28', 'marketCap': 3286307659600},
            {'symbol': 'AAPL', 'date': '2025-03-27', 'marketCap': 3376043917400},
            {'symbol': 'AAPL', 'date': '2025-03-26', 'marketCap': 3341054317720}
        ]
        """
        params = {"symbol": symbol, "from": start_date, "to": end_date, "limit": limit}

        # Remove keys with None values to avoid sending "None" in the query string
        params = {key: value for key, value in params.items() if value is not None}

        extension = f"historical-market-capitalization?" + urlencode(params)

        return self.get_api_call(extension, legacy=False)

    def get_earnings(self, symbol):
        """
        Fetches the earnings data for a given stock symbol.

        Parameters
        ----------
        symbol : str
            The stock ticker symbol for which the earnings data is requested.

        Returns
        -------
        dict:
            The JSON response as a dictionary if the request is successful (status code 200).
        None:
            If an error occurs, such as a network issue or a non-200 status code.

        Notes
        -----
        - This function constructs the appropriate API extension and calls `get_api_call` to fetch the earnings data.
        - The API response includes financial metrics such as earnings per share (EPS), reported and estimated earnings, and earnings dates.
        - If the API request fails, an error message is printed, and None is returned.

        Example
        -------
        >>> get_earnings("AAPL")
            [
                {
                    'symbol': 'AAPL',
                    'date': '2025-10-29',
                    'epsActual': None,
                    'epsEstimated': None,
                    'revenueActual': None,
                    'revenueEstimated': None,
                    'lastUpdated': '2025-03-30'
                },
                {
                    'symbol': 'AAPL',
                    'date': '2025-07-30',
                    'epsActual': None,
                    'epsEstimated': None,
                    'revenueActual': None,
                    'revenueEstimated': None,
                    'lastUpdated': '2025-03-30'
                }
            ]
        """

        extension = "historical/earning_calendar/" + symbol

        return self.get_api_call(extension, legacy=True, separator=False)

    def get_eod(self, symbol, start_date=None, end_date=None):
        """
        Fetches the end-of-day (EOD) stock price data for a given stock symbol within a specified date range.

        Parameters
        ----------
        symbol : str
            The stock ticker symbol for which the EOD price data is requested.

        start_date : str, optional
            The start date for the market capitalization data in the format 'YYYY-MM-DD'.
            If not provided, the earliest available data will be returned.

        end_date : str, optional
            The end date for the market capitalization data in the format 'YYYY-MM-DD'.
            If not provided, the latest available data will be returned.

        Returns
        -------
        dict:
            The JSON response as a dictionary if the request is successful (status code 200).
        None:
            If an error occurs, such as a network issue or a non-200 status code.

        Notes
        -----
        - This function constructs the appropriate API extension and calls `get_api_call` to fetch the EOD data.
        - If the API request fails, an error message is printed, and None is returned.

        Example
        -------
        >>> get_eod("AAPL", start_date="2025-03-28", end_date="2025-04-01")
        [
            {'symbol': 'AAPL', 'date': '2025-04-01', 'price': 223.19, 'volume': 35715179},
            {'symbol': 'AAPL', 'date': '2025-03-31', 'price': 222.13, 'volume': 65299321},
            {'symbol': 'AAPL', 'date': '2025-03-28', 'price': 217.9, 'volume': 39818617}
        ]
        """
        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
        }

        # Remove keys with None values to avoid sending "None" in the query string
        params = {key: value for key, value in params.items() if value is not None}

        extension = f"historical-price-eod/light?" + urlencode(params)

        return self.get_api_call(extension, legacy=False)

    def get_shares(self, symbol):
        """
        Retrieves historical float shares data for a given stock symbol.

        This function constructs the appropriate API endpoint and sends a request to
        fetch historical float shares data for the specified stock symbol. It filters
        out any parameters with `None` values to avoid malformed query strings.

        Only till 2021, Not in

        Parameters
        ----------
        symbol : str
            The stock ticker symbol (e.g., "AAPL" for Apple Inc.).

        Returns
        -------
        dict or list
            The response from the API containing historical float shares data.
            The format depends on the API implementation.

        Notes
        -----
        - This function uses the `get_api_call` function to make the API request.
        - The request is made with `legacy=False` and `v4=True`, indicating use of a version 4 API endpoint.
        - The endpoint used is: `historical/shares_float`.

        Example
        -------
        >>> get_shares("AAPL")
        [
            {
                "symbol": "AAPL",
                "date": "2024-01-31",
                "freeFloat": 99.9125,
                "floatShares": "15448370837",
                "outstandingShares": "15461900000",
                "source": null
            }
        ]
        """

        params = {"symbol": symbol}

        # Remove keys with None values to avoid sending "None" in the query string
        params = {key: value for key, value in params.items() if value is not None}

        extension = f"historical/shares_float?" + urlencode(params)

        return self.get_api_call(extension, v4=True)
