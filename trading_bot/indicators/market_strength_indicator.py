"""All market strength indicators classes.

This file allows developer to use any market strength financial indicator contains in this file.

All market strength indicator classes must inherit from the `MarketStrengthIndicator` class to standardize all used market strength indicators.

This script requires that `pandas` and `numpy` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:
    * MarketStrengthIndicator - abstract class used to reprensent the base of each market strength indicator that we will use
    * SimpleMovingAverage -  class used to reprensent "Simple Moving Average" or "SMA" indicator
    * ExponentialMovingAverage - class used to reprensent "Exponential Moving Average" or "EMA" indicator
    * DoubleExponentialMovingAverage - class used to reprensent "Double Exponential Moving Average" or "DEMA" indicator
    * TripleExponentialMovingAverage - class used to reprensent "Triple Exponential Moving Average" or "TEMA" indicator
    * MovingAverageEnvelopes - class used to reprensent "Moving Average Envelopes" or "MAE" indicator
    * TriangularMovingAverage - class used to reprensent "Triangular Moving Average" or "TMA" indicator
    * WildersMovingAverage - class used to reprensent "Wilders Moving Average" or "WildersMA" indicator
    * WeigthtedMovingAverage - class used to reprensent "Weighted Moving Average" or "WMA" indicator
"""

import abc

import numpy as np
import pandas as pd

from trading_bot.indicators.base_indicator import Indicator
from trading_bot.indicators.moving_average_indicator import ExponentialMovingAverage, SimpleMovingAverage
from trading_bot.indicators.utils import non_zero_range, roc


class MarketStrengthIndicator(Indicator, metaclass=abc.ABCMeta):
    """
    An abstract class used to reprensent the base of each market strength indicator that we will use.

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


class AccumulationDistribution(MarketStrengthIndicator):
    """
    A class used to reprensent "Accumulation Distribution" indicator.
    This momentum indicator makes a line with the accumulation of Close Location Value Volume.
    Is computed using the high, low, closing prices and volume.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(high, low, close, volume)
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
        MarketStrengthIndicator.__init__(
            self, "AD", 0)

    def calculate_serie(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
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
        volume : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        money_flow_multiplier = (
            close - low) - (high - close) / (high - low)
        return (money_flow_multiplier * volume).cumsum()

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
        return self.calculate_serie(df["High"], df["Low"], df["Close"], df["Volume"])


class OnBalanceVolume(MarketStrengthIndicator):
    """
    A class used to reprensent "On Balance Volume" or "OBV" indicator.
    This momentum indicator makes a line with the accumulation of signed volume.
    Is computed using the closing prices and volume.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(close, volume)
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
        MarketStrengthIndicator.__init__(
            self, "OBV", 0)

    def calculate_serie(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        close : pd.Series
            data used to calculate the indicator
        volume : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        df = pd.DataFrame({self.get_column_name(): range(0, len(close), 1)})
        df[self.get_column_name()] = np.nan

        neg_change = close < close.shift(1)
        pos_change = close > close.shift(1)
        if pos_change.any():
            df.loc[pos_change, self.get_column_name()] = volume
        if neg_change.any():
            df.loc[neg_change, self.get_column_name()] = -volume
        return df[self.get_column_name()].cumsum()

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
        return self.calculate_serie(df["Close"], df["Volume"])


class WeightedOnBalanceVolume(MarketStrengthIndicator):
    """
    A class used to reprensent "Weighted On Balance Volume" or "WOBV" indicator.
    This momentum indicator makes a line with the accumulation of signed volume.
    Is computed using the closing prices and volume.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(close, volume)
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
        MarketStrengthIndicator.__init__(
            self, "WOBV", 0)

    def calculate_serie(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        close : pd.Series
            data used to calculate the indicator
        volume : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        return (volume * close.diff()).cumsum()

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
        return self.calculate_serie(df["Close"], df["Volume"])


class PriceVolumeTrend(MarketStrengthIndicator):
    """
    A class used to reprensent "Price Volume Trend" or "PVT" indicator.
    This momentum indicator makes a line with the accumulation of volume times rate of change.
    Is computed using the closing prices and volume.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(close, volume)
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
        MarketStrengthIndicator.__init__(
            self, "PVT", 0)

    def calculate_serie(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate the indicator for the given serie.

        Parameters
        ----------
        close : pd.Series
            data used to calculate the indicator
        volume : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        return (roc(close) * volume).cumsum()

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
        return self.calculate_serie(df["Close"], df["Volume"])


class ChaikinMoneyFlow(MarketStrengthIndicator):
    """
    A class used to reprensent "Chaikin Money Flow" indicator.
    This momentum indicator makes a line with the total close location value volume over total volume for the previous n-period.
    Is computed using the high, low and closing prices and volume.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(high, low, close, volume)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period: int = 21) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int
            number of ticks that indicator is based to calculate (default = 21)
        """
        MarketStrengthIndicator.__init__(
            self, "Chaikin MF (" + str(period) + ")", period)

    def calculate_serie(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
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
        volume : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        high_low_range = non_zero_range(high, low)
        acc_dist = 2 * close - (high + low)
        acc_dist *= volume / high_low_range
        return acc_dist.rolling(self.period).sum() / volume.rolling(self.period).sum()

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
        return self.calculate_serie(df["High"], df["Low"], df["Close"], df["Volume"])


class ChaikinOscillator(MarketStrengthIndicator):
    """
    A class used to reprensent "Chaikin Oscillator" indicator.
    This market strength indicator makes a line with the difference between n1-period and n2-period exponential moving averages.
    Is computed using the high, low and closing prices and volume.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate the first accumulation distribution EMA
    period2 : int
        number of ticks that indicator is based to calculate the second accumulation distribution EMA

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate_serie(high, low, close, volume)
        Calculate the indicator for the given serie.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period: int = 3, period2: int = 10) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int
            number of ticks that indicator is based to calculate the first accumulation distribution EMA (default = 3)
        period2 : int
            number of ticks that indicator is based to calculate the second accumulation distribution EMA (default = 10)
        """
        MarketStrengthIndicator.__init__(
            self, "Chaikin Oscillator (" + str(period) + ", " + str(period2) + ")", period)
        self.period2 = period2

    def calculate_serie(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
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
        volume : pd.Series
            data used to calculate the indicator

        Returns
        -------
        pd.Series
            the serie containing the calculated data
        """
        adl = AccumulationDistribution().calculate_serie(high, low, close, volume)
        return adl.ewm(span=self.period, adjust=False).mean() - adl.ewm(span=self.period2, adjust=False).mean()

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
        return self.calculate_serie(df["High"], df["Low"], df["Close"], df["Volume"])


class PercentageVolumeOscillator(MarketStrengthIndicator):
    """
    A class used to reprensent "Percentage Volume Oscillator" or "PVO" indicator.
    This market strength indicator makes a two line :
        1. percent difference between n1- and n2-period simple moving averages
        2. signal line represented as a n3-period exponential moving average of the first
    Is computed using the high, low and closing prices and volume.

    Attributes
    ----------
    period : int
        number of ticks for fast SMA
    period2 : int
        number of ticks for slow SMA
    period3 : int
        number of ticks for the EMA signal

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

    def __init__(self, period: int = 9, period2: int = 16, period3: int = 9) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int
            number of ticks for fast SMA (default = 9)
        period2 : int
            number of ticks for slow SMA (default = 16)
        period3 : int
            number of ticks for the EMA (default = 9)
        """
        MarketStrengthIndicator.__init__(
            self, "PVO (" + str(period) + ", " + str(period2) + ", " + str(period3) + ")", period)
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
        pd.Series
            the serie containing the calculated data
        """
        fast_sma = SimpleMovingAverage(self.period).calculate_serie(close)
        slow_sma = SimpleMovingAverage(self.period2).calculate_serie(close)
        pvo = 100 * (fast_sma - slow_sma) / slow_sma

        signal_ema = ExponentialMovingAverage(
            self.period3).calculate_serie(pvo)
        return pvo - signal_ema

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
