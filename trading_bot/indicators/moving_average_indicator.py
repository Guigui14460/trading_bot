"""All moving average indicators classes.

This file allows developer to use any moving average financial indicator contains in this file.

All moving average indicator classes must inherit from the `MovingAverageIndicator` class to standardize all used moving average indicators.

This script requires that `pandas` and `numpy` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:

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
from trading_bot.indicators.utils import wwma


class MovingAverageIndicator(Indicator, metaclass=abc.ABCMeta):
    """
    An abstract class used to reprensent the base of each moving average indicator that we will use.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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


class SimpleMovingAverage(MovingAverageIndicator):
    """
    A class used to reprensent "Simple Moving Average" or "SMA" indicator.
    This statistical indicator makes a line with the moving average of the previous n-period ticks.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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
        MovingAverageIndicator.__init__(
            self, "SMA (" + str(period) + ")", period)

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
        return df['Close'].rolling(self.period).mean()


class ExponentialMovingAverage(MovingAverageIndicator):
    """
    A class used to reprensent "Exponential Moving Average" or "EMA" indicator.
    This statistical indicator makes a line with the moving average of the previous n-period ticks with a smoothing constant.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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
        MovingAverageIndicator.__init__(
            self, "EMA (" + str(period) + ")", period)

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
        return df['Close'].ewm(span=self.period, adjust=False).mean()


class DoubleExponentialMovingAverage(MovingAverageIndicator):
    """
    A class used to reprensent "Double Exponential Moving Average" or "DEMA" indicator.
    This statistical indicator makes a line with the moving average of the previous n-period ticks.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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
        MovingAverageIndicator.__init__(
            self, "DEMA (" + str(period) + ")", period)

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
        ema1 = ExponentialMovingAverage().calculate(df)
        ema2 = ExponentialMovingAverage().calculate(ema1)
        return 2*ema1 - ema2


class TripleExponentialMovingAverage(MovingAverageIndicator):
    """
    A class used to reprensent "Triple Exponential Moving Average" or "TEMA" indicator.
    This statistical indicator makes a line with the triple exponential moving average of the previous n-period ticks.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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
        period : int, optional
            number of ticks that indicator is based to calculate (default = 21)

        Raises
        ------
        ValueError
            If the `period` is lesser than 0
        """
        if period < 0:
            raise ValueError("period cannot be lesser than 0")
        period = int(period)
        MovingAverageIndicator.__init__(
            self, "TEMA (" + str(period) + ")", period)

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
        ema1 = ExponentialMovingAverage().calculate(df)
        ema2 = ExponentialMovingAverage().calculate(ema1)
        ema3 = ExponentialMovingAverage().calculate(ema2)
        return 3*(ema1 - ema2) + ema3


class MovingAverageEnvelopes(MovingAverageIndicator):
    """
    A class used to reprensent "Moving Average Envelopes" or "MAE" indicator.
    This statistical indicator makes three lines consisting simple moving average of the n-period ticks line and bands that are `p`% above and below.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate
    percent : float
        percent which the bands are distant

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

    calculate(df)
        Calculate the indicator for the given data.

    calculate_in_place(df)
        Calculate the indicator for the given data and put this directly in 
        the dataframe.
    """

    def __init__(self, period: int = 10, percent: float = 3.0) -> None:
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 10)

        percent : float
            percent which the bands are distant (default = 3.0)

        Raises
        ------
        ValueError
            If the `period` is lesser than 0
        """
        if period < 0:
            raise ValueError("period cannot be lesser than 0")
        period = int(period)
        if percent < 0:
            percent = 0
        MovingAverageIndicator.__init__(
            self, "MAE (" + str(period) + ")", period)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        upper_band : pd.Series
            upper band of the calculated data
        sma : pd.Series
            simple moving average of the data
        lower_band : pd.Series
            lower band of the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        sma = SimpleMovingAverage(period=self.period).calculate(df)

        distance = sma * 3 / 100
        upper_band = sma + distance
        lower_band = sma - distance
        return upper_band, sma, lower_band

    def calculate_in_place(self, df: pd.DataFrame) -> None:
        """
        Calculate the indicator for the given data and put this directly in 
        the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator
        """
        df[self.__column_name + " Up"], df[SimpleMovingAverage(
            self.period).get_column_name()], df[self.__column_name + " Down"] = self.calculate(df)


class TriangularMovingAverage(MovingAverageIndicator):
    """
    A class used to reprensent "Triangular Moving Average" or "TMA" indicator.
    This statistical indicator makes a line with the triangular moving average of the previous n-period ticks.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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
        MovingAverageIndicator.__init__(
            self, "TMA (" + str(period) + ")", period)

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
        half_length = round(0.5 * (self.period + 1))
        sma = df['Close'].rolling(half_length, min_periods=half_length).mean()
        trima = sma.rolling(half_length, min_periods=half_length).mean()
        return trima


class WildersMovingAverage(MovingAverageIndicator):
    """
    A class used to reprensent "Wilders Moving Average" or "WildersMA" indicator.
    This statistical indicator makes a line with the incremental n-period moving average.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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
        MovingAverageIndicator.__init__(
            self, "WildersMA (" + str(period) + ")", period)

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
        alpha = (1.0 / self.period) if self.period > 0 else 0.5
        return df['Close'].ewm(alpha=alpha).mean()


class WeigthtedMovingAverage(MovingAverageIndicator):
    """
    A class used to reprensent "Weighted Moving Average" or "WMA" indicator.
    This statistical indicator makes a line with the weighted moving average of the previous n-period.
    Is computed using the closing price.

    Attributes
    ----------
    period : int
        number of ticks that indicator is based to calculate

    Methods
    -------
    get_column_name()
        Return the column name associated at this indicator.

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
        MovingAverageIndicator.__init__(
            self, "WMA (" + str(period) + ")", period)

    def calculate(self, df: pd.DataFrame, asc: bool = True) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator
        asc : bool
            recent values weights more (default = True)

        Returns
        -------
        pd.Series
            the serie containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not standardize yet
        """
        total_weigth = 0.5 * self.period * (self.period + 1)
        default_weigth = pd.Series(np.arange(1, self.period + 1))
        weights = default_weigth if asc else default_weigth[::-1]
        close = df['Close'].rolling(self.period, min_periods=self.period)

        def calculate_dot_product(weights):
            def _compute(data):
                return np.dot(data, weights) / total_weigth
            return _compute

        return close.apply(calculate_dot_product(weights), raw=True)


# TODO: make VariableMovingAverage indicator after implementing ChandeMomentumOscillator
