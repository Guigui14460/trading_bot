"""All basic indicators classes.

This file allows developer to use any basic financial indicator contains in this file.

All basic indicator classes must inherit from the `BasicIndicator` class to standardize all used basic indicators.

This script requires that `pandas` and `numpy` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:
    * BasicIndicator - abstract class used to reprensent the base of each basic indicator that we will use
    * HighestHigh - class used to reprensent "Highest High" indicator
    * LowestLow - class used to reprensent "Lowest Low" indicator
    * MedianPrice - class used to reprensent "Median price" indicator
    * TypicalPrice - class used to reprensent "Typical price" indicator
    * AverageTrueRange - class used to reprensent "Average True Range" or "ATR" indicator
"""

import abc
import numpy as np
import pandas as pd

from trading_bot.indicators.base_indicator import Indicator
from trading_bot.indicators.utils import wwma


class BasicIndicator(Indicator, metaclass=abc.ABCMeta):
    """
    An abstract class used to reprensent the base of each basic indicator that we will use.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(serie)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, column_name: str, period: int) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        column_name : str
            indicator name to put in the pandas dataframe
        period : int
            number of ticks that indicator is based to calculate
        """
        Indicator.__init__(self, column_name, period)


class HighestHigh(BasicIndicator):
    """
    A class used to reprensent "Highest High" indicator.
    This statistical indicator makes a line with the highest "High" n-period ticks indicator.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(serie)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period: int = 14) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 14)

        Raises
        ------
        ValueError
            If the `period` is lesser than 0
        """
        if period < 0:
            raise ValueError("period cannot be lesser than 0")
        period = int(period)
        BasicIndicator.__init__(
            self, "Highest High (" + str(period) + ")", period)

    def calculate_serie(self, serie: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        serie : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        return serie.rolling(window=self.period).max(skipna=True)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df['High'])


class LowestLow(BasicIndicator):
    """
    A class used to reprensent "Lowest Low" indicator.
    This statistical indicator makes a line with the Lowest "Low" n-period ticks indicator.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(serie)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period: int = 14) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 14)

        Raises
        ------
        ValueError
            If the `period` is lesser than 0
        """
        if period < 0:
            raise ValueError("period cannot be lesser than 0")
        period = int(period)
        BasicIndicator.__init__(
            self, "Lowest Low (" + str(period) + ")", period)

    def calculate_serie(self, serie: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        serie : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        return serie.rolling(window=self.period).min(skipna=True)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df['Low'])


class MedianPrice(BasicIndicator):
    """
    A class used to reprensent "Median price" indicator.
    This statistical indicator makes a line with the average of the "High" and "Low" prices.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(high, low)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self) -> None:
        """
        Initializer and constructor of the class.
        """
        BasicIndicator.__init__(self, "Median Price", 0)

    def calculate_serie(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        high : pd.Series
            data used to calculate the indicator
        low : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        return np.mean(np.stack((high, low)), axis=0)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df['High'], df['Low'])


class TypicalPrice(BasicIndicator):
    """
    A class used to reprensent "Typical price" indicator.
    This statistical indicator makes a line with the average of the "High", "Low" and "Close" prices.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(high, low, close)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self) -> None:
        """
        Initializer and constructor of the class.
        """
        BasicIndicator.__init__(self, "Typical Price", None)

    def calculate_serie(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        high : pd.Series
            data used to calculate the indicator
        low : pd.Series
            data used to calculate the indicator
        close : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        return np.mean(np.stack((high, low, close)), axis=0)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df['High'], df['Low'], df['Close'])


class AverageTrueRange(BasicIndicator):
    """
    A class used to reprensent "Average True Range" or "ATR" indicator.
    This volatility indicator makes a line with the average of the "High", "Low" and "Close" prices.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(high, low, close)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period: int = 14) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 14)

        Raises
        ------
        ValueError
            If the `period` is lesser than 0
        """
        if period < 0:
            raise ValueError("period cannot be lesser than 0")
        period = int(period)
        BasicIndicator.__init__(
            self, "Average True Range (" + str(period) + ")", period)

    def calculate_serie(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        high : pd.Series
            data used to calculate the indicator
        low : pd.Series
            data used to calculate the indicator
        close : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        tr = np.stack((abs(high - low), abs(high - close.shift(1)),
                       abs(low - close.shift(1)))).max(axis=1)
        return wwma(tr, self.period)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df["High"], df["Low"], df["Close"])
