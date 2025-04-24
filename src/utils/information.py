import re


def get_sic_industry_names() -> dict[str, str]:
    """
    Extracts and returns a dictionary mapping SIC (Standard Industrial Classification) major group codes
    to their corresponding industry names.

    The function parses a predefined text containing SIC divisions and major groups, extracting
    the two-digit major group codes and their industry descriptions.

    Returns
    -------
    dict
        A dictionary where:
        - Keys are SIC major group codes (as strings, e.g., "01", "10").
        - Values are the corresponding industry names (e.g., "Agricultural Production Crops", "Metal Mining").

    Notes
    -----
    - The SIC industry classification data is sourced from the OSHA website:
      https://www.osha.gov/data/sic-manual
    - The function uses regular expressions to extract industry names efficiently.
    - The extracted SIC codes represent broad industry categories across multiple divisions.
    """

    sic_text = """Division A: Agriculture, Forestry, And Fishing

    Major Group 01: Agricultural Production Crops
    Major Group 02: Agriculture Production Livestock And Animal Specialties
    Major Group 07: Agricultural Services
    Major Group 08: Forestry
    Major Group 09: Fishing, Hunting, And Trapping
    

    Division B: Mining

    Major Group 10: Metal Mining
    Major Group 12: Coal Mining
    Major Group 13: Oil And Gas Extraction
    Major Group 14: Mining And Quarrying Of Nonmetallic Minerals, Except Fuels
    

    Division C: Construction

    Major Group 15: Building Construction General Contractors And Operative Builders
    Major Group 16: Heavy Construction Other Than Building Construction Contractors
    Major Group 17: Construction Special Trade Contractors
    

    Division D: Manufacturing

    Major Group 20: Food And Kindred Products
    Major Group 21: Tobacco Products
    Major Group 22: Textile Mill Products
    Major Group 23: Apparel And Other Finished Products Made From Fabrics And Similar Materials
    Major Group 24: Lumber And Wood Products, Except Furniture
    Major Group 25: Furniture And Fixtures
    Major Group 26: Paper And Allied Products
    Major Group 27: Printing, Publishing, And Allied Industries
    Major Group 28: Chemicals And Allied Products
    Major Group 29: Petroleum Refining And Related Industries
    Major Group 30: Rubber And Miscellaneous Plastics Products
    Major Group 31: Leather And Leather Products
    Major Group 32: Stone, Clay, Glass, And Concrete Products
    Major Group 33: Primary Metal Industries
    Major Group 34: Fabricated Metal Products, Except Machinery And Transportation Equipment
    Major Group 35: Industrial And Commercial Machinery And Computer Equipment
    Major Group 36: Electronic And Other Electrical Equipment And Components, Except Computer Equipment
    Major Group 37: Transportation Equipment
    Major Group 38: Measuring, Analyzing, And Controlling Instruments; Photographic, Medical And Optical Goods; Watches And Clocks
    Major Group 39: Miscellaneous Manufacturing Industries
    

    Division E: Transportation, Communications, Electric, Gas, And Sanitary Services

    Major Group 40: Railroad Transportation
    Major Group 41: Local And Suburban Transit And Interurban Highway Passenger Transportation
    Major Group 42: Motor Freight Transportation And Warehousing
    Major Group 43: United States Postal Service
    Major Group 44: Water Transportation
    Major Group 45: Transportation By Air
    Major Group 46: Pipelines, Except Natural Gas
    Major Group 47: Transportation Services
    Major Group 48: Communications
    Major Group 49: Electric, Gas, And Sanitary Services
    

    Division F: Wholesale Trade

    Major Group 50: Wholesale Trade-durable Goods
    Major Group 51: Wholesale Trade-non-durable Goods
    

    Division G: Retail Trade

    Major Group 52: Building Materials, Hardware, Garden Supply, And Mobile Home Dealers
    Major Group 53: General Merchandise Stores
    Major Group 54: Food Stores
    Major Group 55: Automotive Dealers And Gasoline Service Stations
    Major Group 56: Apparel And Accessory Stores
    Major Group 57: Home Furniture, Furnishings, And Equipment Stores
    Major Group 58: Eating And Drinking Places
    Major Group 59: Miscellaneous Retail
    

    Division H: Finance, Insurance, And Real Estate

    Major Group 60: Depository Institutions
    Major Group 61: Non-depository Credit Institutions
    Major Group 62: Security And Commodity Brokers, Dealers, Exchanges, And Services
    Major Group 63: Insurance Carriers
    Major Group 64: Insurance Agents, Brokers, And Service
    Major Group 65: Real Estate
    Major Group 67: Holding And Other Investment Offices
    

    Division I: Services

    Major Group 70: Hotels, Rooming Houses, Camps, And Other Lodging Places
    Major Group 72: Personal Services
    Major Group 73: Business Services
    Major Group 75: Automotive Repair, Services, And Parking
    Major Group 76: Miscellaneous Repair Services
    Major Group 78: Motion Pictures
    Major Group 79: Amusement And Recreation Services
    Major Group 80: Health Services
    Major Group 81: Legal Services
    Major Group 82: Educational Services
    Major Group 83: Social Services
    Major Group 84: Museums, Art Galleries, And Botanical And Zoological Gardens
    Major Group 86: Membership Organizations
    Major Group 87: Engineering, Accounting, Research, Management, And Related Services
    Major Group 88: Private Households
    Major Group 89: Miscellaneous Services
    

    Division J: Public Administration

    Major Group 91: Executive, Legislative, And General Government, Except Finance
    Major Group 92: Justice, Public Order, And Safety
    Major Group 93: Public Finance, Taxation, And Monetary Policy
    Major Group 94: Administration Of Human Resource Programs
    Major Group 95: Administration Of Environmental Quality And Housing Programs
    Major Group 96: Administration Of Economic Programs
    Major Group 97: National Security And International Affairs
    Major Group 99: Nonclassifiable Establishments
    """
    sic_dict = {}

    # Extract major groups and their descriptions
    matches = re.findall(r"Major Group (\d{2}): (.+)", sic_text)

    for code, description in matches:
        sic_dict[code] = description

    return sic_dict
