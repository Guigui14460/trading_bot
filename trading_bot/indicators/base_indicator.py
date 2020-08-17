"""Base Indicator class

This file allows developer to create any financial indicator which it inherit from the class contains in this file.

All indicator classes must inherit from this class to standardize all used indicators.

This script requires that `pandas` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:

    * Indicator - abstract class represents all indicator possibilities
"""

import abc
import pandas as pd


class Indicator(abc.ABC):
    """
    An abstract class used to reprensent the base of each indicator that we will use.

    Attributes
    ----------
    __column_name : str
        indicator name to put in the pandas dataframe.
        To access to the value of this private attribute, you must use
        :func:`~trading_bot.indicators.Indicator.get_column_name` method.
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
        self.__column_name = column_name
        self.period = period

    def get_column_name(self) -> str:
        """
        Return the column name associated at this indicator.
        It returns the value of the `__column_name` private attribute.

        Returns
        -------
        str
            column name for the pandas dataframe
        """
        return self.__column_name

    @abc.abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
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
        """
        pass

    def calculate_in_place(self, df: pd.DataFrame) -> None:
        """
        Calculate the indicator for the given data and put this directly in 
        the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            data used to calculate the indicator
        """
        df[self.__column_name] = self.calculate(df)
