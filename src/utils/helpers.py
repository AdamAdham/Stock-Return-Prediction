def month_gap_diff(date1: str, date2: str) -> int:
    """
    Calculates the number of full months between two dates in "YYYY-MM" format, excluding the starting and end months.

    Parameters
    ----------
    date1 : str
        The earlier date in "YYYY-MM" format.
    date2 : str
        The later date in "YYYY-MM" format.

    Returns
    -------
    int
        The number of full months between date1 and date2, excluding the month of date1.

    Notes
    -----
    - Assumes that date2 is later than date1.
    - Does not validate date formatting or chronological order.
    """

    # Extract year and month from both strings
    year1, month1 = int(date1[:4]), int(date1[5:7])
    year2, month2 = int(date2[:4]), int(date2[5:7])

    # Calculate the difference in months
    return (year2 - year1) * 12 + (month2 - month1) - 1
