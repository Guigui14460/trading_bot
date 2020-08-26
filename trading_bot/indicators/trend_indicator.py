"""All trend indicators classes.

This file allows developer to use any trend financial indicator contains in this file.

All trend indicator classes must inherit from the `TrendIndicator` class to standardize all used trend indicators.

This script requires that `pandas` and `numpy` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:
    * TrendIndicator - abstract class used to reprensent the base of each trend indicator that we will use
    * SimpleMovingAverage -  class used to reprensent "Simple trend" or "ADX" indicator
"""

import abc
from trading_bot.indicators.utils import recent_maximum_index, recent_minimum_index, wwma
from typing import Tuple

import numpy as np
import pandas as pd

from trading_bot.indicators.base_indicator import Indicator
from trading_bot.indicators.basic_indicator import AverageTrueRange


class TrendIndicator(Indicator, metaclass=abc.ABCMeta):
    """
    An abstract class used to reprensent the base of each trend indicator that we will use.

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


class DirectionalMovementIndex(TrendIndicator):
    """
    A class used to reprensent "Directional Movement Index" or "DMI" indicator.
    This trend indicator makes a line with tells to us when to be long or short.
    Is computed using the high, low and closing price.

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
        TrendIndicator.__init__(
            self, "DMI (" + str(period) + ")", period)

    def calculate_serie(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series]:
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
        di_plus : pd.Series
            the serie containing the calculated data
        di_minus : pd.Series
            the serie containing the calculated data
        """
        ohlc = pd.DataFrame({"High": high, "Low": low, "Close": close})
        up_move = high.diff()
        down_move = low.diff()

        plus = ohlc.apply(self._positive_dmi, axis=1)
        minus = ohlc.apply(self._negative_dmi, axis=1)

        di_plus = 100 * \
            wwma(plus / AverageTrueRange(self.period).calculate_serie(high,
                                                                      low, close), self.period)
        di_minus = 100 * \
            wwma(minus / AverageTrueRange(self.period).calculate_serie(high,
                                                                       low, close), self.period)
        return di_plus, di_minus

    def _positive_dmi(self, row: pd.DataFrame) -> pd.Series:
        """
        Calculate the positive DMI for a given dataframe.

        Parameters
        ----------
        row : pd.DataFrame
            data

        Returns
        -------
        pd.Series
            calculated positive DMI
        """
        return row["up"] if row["up"] > row["down"] and row["up"] > 0 else 0

    def _negative_dmi(self, row: pd.DataFrame) -> pd.Series:
        """
        Calculate the negative DMI for a given dataframe.

        Parameters
        ----------
        row : pd.DataFrame
            data

        Returns
        -------
        pd.Series
            calculated negative DMI
        """
        return row["down"] if row["down"] > row["up"] and row["down"] > 0 else 0

    def calculate(self, df: pd.DataFrame) -> Tuple[pd.Series]:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        di_plus : pd.Series
            the serie containing the calculated data
        di_minus : pd.Series
            the serie containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df["High"], df["Low"], df["Close"])


class AverageDirectionalMovementIndex(TrendIndicator):
    """
    A class used to reprensent "Average Directional Movement Index" or "ADX" indicator.
    This trend indicator makes a line with a n-period Wilder's moving average of Directional Movement Index.
    Is computed using the high, low and closing price.

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
        TrendIndicator.__init__(
            self, "ADX (" + str(period) + ")", period)

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
        di_plus, di_minus = DirectionalMovementIndex(
            self.period).calculate_serie(high, low, close)
        return 100 * wwma(abs(di_plus - di_minus) / (di_plus + di_minus), self.period)

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


class Aroon(TrendIndicator):
    """
    A class used to reprensent "Aroon" indicator.
    This trend indicator makes a two line that range from 0 to 100 and indicate how recently the highest high and lowest low over the last n-periods occurred.
    Is computed using the high and low prices.

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

    def __init__(self, period: int = 10) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 10)

        Raises
        ------
        ValueError
            If the `period` is lesser than 0
        """
        if period < 0:
            raise ValueError("period cannot be lesser than 0")
        period = int(period)
        TrendIndicator.__init__(
            self, "Aroon (" + str(period) + ")", period)

    def calculate_serie(self, high: pd.Series, low: pd.Series) -> Tuple[pd.Series]:
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
        aroon_up : pd.Series
            the serie containing the how recently the highest high occurred
        aroon_down : pd.Series
            the serie containing the how recently the lowest low occurred
        """
        periods_from_hh = high.rolling(
            self.period + 1).apply(recent_maximum_index, raw=True)
        periods_from_ll = low.rolling(
            self.period + 1).apply(recent_minimum_index, raw=True)

        aroon_up = aroon_down = 100
        aroon_up *= (1 - (periods_from_hh / self.period))
        aroon_down *= (1 - (periods_from_ll / self.period))
        return aroon_up, aroon_down

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        aroon_up : pd.Series
            the serie containing the how recently the highest high occurred
        aroon_down : pd.Series
            the serie containing the how recently the lowest low occurred

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df['High'], df['Low'])


class AroonOscillator(TrendIndicator):
    """
    A class used to reprensent "Aroon Oscillator" indicator.
    This trend indicator makes a line that is the difference between the up and down components of a n-period Aroon indicator.
    Is computed using the high, low and close prices and volume.

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

    def __init__(self, period: int = 10) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 10)

        Raises
        ------
        ValueError
            If the `period` is lesser than 0
        """
        if period < 0:
            raise ValueError("period cannot be lesser than 0")
        period = int(period)
        TrendIndicator.__init__(
            self, "Aroon Oscillator (" + str(period) + ")", period)

    def calculate_serie(self, high: pd.Series, low: pd.Series) -> Tuple[pd.Series]:
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
        aroon_up, aroon_down = Aroon(self.period).calculate_serie(high, low)
        return aroon_up - aroon_down

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


class ParabolicStopAndReversal(TrendIndicator):
    """
    A class used to reprensent "Parabolic Stop and Reversal" or "PSAR" indicator.
    This trend indicator makes a line that is the stop and reversal value for the next time period using an acceleration rate of 0.02 and a limit of 0.2.
    Is computed using the high, low and close prices.

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

    def __init__(self, acceleration_rate: float = 0.02, max_acceleration_rate: float = 0.2) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        acceleration_rate : float, optional
            rate of the acceleration of ticks that indicator is based to calculate (default = 0.02)
        max_acceleration_rate : float, optional
            max rate of the acceleration of ticks that indicator is based to calculate (default = 0.2)

        Raises
        ------
        ValueError
            If the `acceleration_rate` or `max_acceleration_rate` is lesser than 0
        """
        if acceleration_rate < 0:
            raise ValueError("acceleration_rate cannot be lesser than 0")
        if max_acceleration_rate < 0:
            raise ValueError("max_acceleration_rate cannot be lesser than 0")
        TrendIndicator.__init__(
            self, "PSAR (alpha=" + str(acceleration_rate) + " ,m=" + str(max_acceleration_rate) + ")", None)
        self.acceleration_rate = acceleration_rate
        self.max_acceleration_rate = max_acceleration_rate

    def calculate_serie(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series]:
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
        psar : pd.Series
            the serie containing the calculated data
        psar_bull : pd.Series
            the serie containing the calculated data for the bull market
        psar_bear : pd.Series
            the serie containing the calculated data for the bear market
        """
        length = high.shape[0]
        af = self.acceleration_rate
        psar = close[0:close.shape[0]]
        psar_bull = [None] * length
        psar_bear = [None] * length
        bullish = True
        high_point = high.iloc[0]
        low_point = low.iloc[0]

        for i in range(2, length):
            if bullish:
                psar[i] = psar[i-1] + af * (high_point - psar[i-1])
            else:
                psar[i] = psar[i-1] + af * (low_point - psar[i-1])

            reverse = False

            if bullish:
                if low[i] < psar[i]:
                    bullish = False
                    reverse = True
                    psar[i] = high_point
                    low_point = low[i]
                    af = self.acceleration_rate
            else:
                if high[i] > psar[i]:
                    bullish = True
                    reverse = True
                    psar[i] = low_point
                    high_point = high[i]
                    af = self.acceleration_rate

            if not reverse:
                if bullish:
                    if high[i] > high_point:
                        high_point = high[i]
                        af = min(af + self.acceleration_rate,
                                 self.max_acceleration_rate)
                    if low[i-1] < psar[i]:
                        psar[i] = low[i-1]
                    if low[i-2] < psar[i]:
                        psar[i] = low[i-2]
                else:
                    if low[i] > low_point:
                        low_point = low[i]
                        af = min(af + self.acceleration_rate,
                                 self.max_acceleration_rate)
                    if high[i-1] > psar[i]:
                        psar[i] = high[i-1]
                    if high[i-2] > psar[i]:
                        psar[i] = high[i-2]

            if bullish:
                psar_bull[i] = psar[i]
            else:
                psar_bear = psar[i]
        psar_bull = pd.Series(psar_bull, "PSAR Bull", index=close.index)
        psar_bear = pd.Series(psar_bear, "PSAR Bear", index=close.index)
        return psar, psar_bull, psar_bear

    def calculate(self, df: pd.DataFrame) -> Tuple[pd.Series]:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        psar : pd.Series
            the serie containing the calculated data
        psar_bull : pd.Series
            the serie containing the calculated data for the bull market
        psar_bear : pd.Series
            the serie containing the calculated data for the bear market

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        return self.calculate_serie(df['High'], df['Low'], df["Close"])
