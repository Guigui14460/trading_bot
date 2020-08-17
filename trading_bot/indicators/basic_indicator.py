import abc
import pandas as pd

from .base_indicator import Indicator


class BasicIndicator(Indicator, metaclass=abc.ABCMeta):
    """
    An abstract class used to reprensent the base of each basic indicator that we will use.

    Attributes
    ----------
    period : int
        numer of ticks that indicator is based to calculate

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


class HighestHigh(BasicIndicator):
    """
    A class used to reprensent "Highest High" indicator.
    This indicator makes a line with the highest "High" n-period ticks indicator.

    Attributes
    ----------
    period : int
        numer of ticks that indicator is based to calculate

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

    def __init__(self, period=14):
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 14)
        """
        BasicIndicator.__init__(self, "Highest High", period)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        pd.DataFrame
            the DataFrame containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not sdandardize yet
        """
        highest_high = df['High'].copy()
        if self.period <= len(highest_high):
            for i in range(0, len(highest_high)):
                subframe = df['High'][i-self.period+1:i+1]
                highest_high[i] = subframe.max(skipna=True)
        return highest_high


class LowestLow(BasicIndicator):
    """
    A class used to reprensent "Lowest Low" indicator.
    This indicator makes a line with the Lowest "Low" n-period ticks indicator.

    Attributes
    ----------
    period : int
        numer of ticks that indicator is based to calculate

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

    def __init__(self, period=14):
        """
        Initializer and constructor of the class.

        Parameters
        ----------
        period : int, optional
            number of ticks that indicator is based to calculate (default = 14)
        """
        BasicIndicator.__init__(self, "Lowest Low", period)

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator

        Returns
        -------
        pd.DataFrame
            the DataFrame containing the calculated data

        Raises
        ------
        KeyError
            If the dataframe is not sdandardize yet
        """
        lowest_low = df['Low'].copy()
        if self.period <= len(lowest_low):
            for i in range(0, len(lowest_low)):
                subframe = df['Low'][i-self.period+1:i+1]
                lowest_low[i] = subframe.min(skipna=True)
        return lowest_low
