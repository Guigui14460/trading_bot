"""All momentum indicators classes.

This file allows developer to use any momentum financial indicator contains in this file.

All momentum indicator classes must inherit from the `MomentumIndicator` class to standardize all used momentum indicators.

This script requires that `pandas` and `numpy` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:
    * MomentumIndicator - abstract class used to reprensent the base of each market strength indicator that we will use
    * SimpleMovingAverage -  class used to reprensent "Simple Moving Average" or "SMA" indicator
"""

import abc
from trading_bot.indicators.basic_indicator import HighestHigh, LowestLow, TypicalPrice
from trading_bot.indicators.utils import roc, wwma
from trading_bot.indicators.moving_average_indicator import ExponentialMovingAverage
import numpy as np
import pandas as pd

from trading_bot.indicators.base_indicator import Indicator


class MomentumIndicator(Indicator, metaclass=abc.ABCMeta):
    """
    An abstract class used to reprensent the base of each momentum indicator that we will use.

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


class MovingAverageConvergenceDivergence(MomentumIndicator):
    """
    A class used to reprensent "Moving Average Convergence Divergence" or "MACD" indicator.
    This momentum indicator makes :
    - line representing the difference between n1- and n2-period exponential moving averages
    - signal line representing a n3-period exponential moving average of the first line
    Is computed using the closing prices.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate
    period2 : int
        number of ticks that indicator is based to calculate
    period3 : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(close)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period1: int = 12, period2: int = 26, period3: int = 15) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period1 : int
            number of ticks that indicator is based to calculate (default = 12)
        period2 : int
            number of ticks that indicator is based to calculate (default = 26)
        period3 : int
            number of ticks that indicator is based to calculate (default = 15)
        """
        MomentumIndicator.__init__(
            self, "MACD", period1)
        self.period2 = period2
        self.period3 = period3

    def calculate_serie(self, close: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        close : pd.Series
            data used to calculate the indicator

        Returns
        -------
        macd : pd.Series
            the serie containing the MACD calculated data (first line)
        histogram : pd.Series
            the serie containing the MACD histogram calculated data (histogram)
        signal_ema : pd.Series
            the serie containing the MACD difference (second line)
        """
        fast_ema = ExponentialMovingAverage(self.period).calculate_serie(close)
        slow_ema = ExponentialMovingAverage(
            self.period2).calculate_serie(close)
        macd = fast_ema - slow_ema
        signal_ema = ExponentialMovingAverage(
            self.period3).calculate_serie(macd)
        histogram = macd - signal_ema
        return macd, histogram, signal_ema

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        macd : pd.Series
            the serie containing the MACD calculated data (first line)
        histogram : pd.Series
            the serie containing the MACD histogram calculated data (histogram)
        signal_ema : pd.Series
            the serie containing the MACD difference (second line)

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df["Close"])

    def calculate_in_place(self, df: pd.DataFrame) -> None:
        """
        Calculate the indicator for the given data and put this directly in 
        the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator
        """
        df[self.__column_name], df[self.__column_name +
                                   " Histogram"], df[self.__column_name + " Signal"] = self.calculate(df)


class RelativeStrengthIndex(MomentumIndicator):
    """
    A class used to reprensent "Relative Strength Index" or "RSI" indicator.
    This momentum indicator makes a line with the ratio of n-period exponential moving averages for gains and losses rescaled to be between 0 and 100.
    Is computed using the closing prices.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(close)
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
        period : int
            number of ticks that indicator is based to calculate (default = 14)
        """
        MomentumIndicator.__init__(
            self, "RSI", period)

    def calculate_serie(self, close: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        close : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        delta = close.diff()

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        ema_gain = wwma(up, period=self.period)
        ema_loss = wwma(down.abs(), period=self.period)

        rs = ema_gain / ema_loss
        return 100 - (100 / (1 + rs))

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
        return self.calculate_serie(df["Close"])


class CommodityChannemIndex(MomentumIndicator):
    """
    A class used to reprensent "Commodity Channel Index" or "CCI" indicator.
    This momentum indicator makes a line with the comparaison the typical price to the mean over n periods.
    Is computed using the high, low and closing prices.

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

    def __init__(self, period: int = 20) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int
            number of ticks that indicator is based to calculate (default = 20)
        """
        MomentumIndicator.__init__(
            self, "CCI", period)

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
        typical_price = TypicalPrice(
            period=self.period).calculate_serie(high, low, close)
        tp_rolling = typical_price.rolling(window=self.period, min_periods=0)
        return (typical_price - tp_rolling.mean()) / tp_rolling.std()

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


class RateOfChange(MomentumIndicator):
    """
    A class used to reprensent "Rate Of Change" or "ROC" indicator.
    This momentum indicator makes a line with the percent change in close from n period ago.
    Is computed using the closing prices.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(close)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period: int = 12) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int
            number of ticks that indicator is based to calculate (default = 12)
        """
        MomentumIndicator.__init__(
            self, "ROC", period)

    def calculate_serie(self, close: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        close : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        return roc(close)

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
        return self.calculate_serie(df["Close"])


class WilliamsPercentR(MomentumIndicator):
    """
    A class used to reprensent "Williams %R" indicator.
    This momentum indicator makes a line which shows how close the current close is the highest high relative to the range between the highest high and lowest low over n periods.
    Is computed using the high, low and closing prices.

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

    def __init__(self, period: int = 10) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int
            number of ticks that indicator is based to calculate (default = 10)
        """
        MomentumIndicator.__init__(
            self, "Williams %R", period)

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
        hh = HighestHigh(self.period).calculate_serie(high)
        ll = LowestLow(self.period).calculate_serie(low)
        return ((hh - close) / (hh - ll)) * -100

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
