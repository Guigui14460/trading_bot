"""Some useful functions.

This file contains some functions often used in different indicators.

This script requires that `pandas` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:
    * wwma - calculate the Welle Wilder's Exponential Moving Average
    * roc - calculate the rate of change of a price serie
    * non_zero_range - returns the difference of two series and adds epsilon if to any zero values
"""

import sys

import pandas as pd


def wwma(values: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Welle Wilder's Exponential Moving Average.

    Parameters
    ----------
    values : pd.Series
        serie of data
    period : int
        number of period

    Returns
    -------
    pd.Series
        Wilder's Exponential Moving Average
    """
    return values.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def roc(values: pd.Series, period: int = 1) -> pd.Series:
    """
    Calculate the rate of change of a price serie.

    Parameters
    ----------
    values : pd.Series
        serie of data
    period : int
        number of period

    Returns
    -------
    pd.Series
        rate of change of the `values` paramater
    """
    return 100 * values.diff(period) / values.shift(period)


def non_zero_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Returns the difference of two series and adds epsilon if
    to any zero values. This occurs commonly in crypto data when
    high = low.

    Parameters
    ----------
    high : pd.Series
        the high prices
    low : pd.Series
        the low prices

    Returns
    -------
        the difference between `high` and `low`
    """
    diff = high - low
    if diff.eq(0).any().any():
        diff += sys.float_info.epsilon
    return diff
