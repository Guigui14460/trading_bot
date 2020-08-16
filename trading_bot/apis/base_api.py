"""Base API class

This file allows developer to register any API which it inherit from the class contains in this file.

All API classes must inherit from this class to standardize all used APIs.

This script requires that `pandas` be installed within the Python 
environment you are running this script in.

This file can also be imported as a module and contains of the following 
classes and functions:

    * BaseAPI - abstract class represents all API possibilities
"""

import abc
from datetime import datetime
import pandas as pd


class BaseAPI(abc.ABC):
    """
    An abstract class used to reprensent the base of each API that we will use.

    Attributes
    ----------
    generic_columns : list[str]
        list of strings represents the minimal generic dataframe column names
        to properly use the data
    base_url : str
        base URL of the API
    df : pd.DataFrame
        dataframe object representing the data

    Methods
    -------
    load_data(endpoint, start_date, end_date)
        Load the data at the `endpoint`, from `start_date` to `end_date`.
    """

    generic_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def __init__(self, base_url: str, df: pd.DataFrame = None) -> None:
        """
        Parameters
        ----------
        base_url : str
            base URL of the API
        df : pd.DataFrame, optional
            initial dataframe object representing the data (default is None)
        """
        self.base_url = base_url
        self.df = df

    @abc.abstractmethod
    def load_data(self, endpoint: str, start_date: datetime, end_date: datetime) -> None:
        """
        Load the data at the `endpoint`, from `start_date` to `end_date`.

        Parameters
        ----------
        endpoint : str
            The API endpoint without the base URL
        start_date : datetime
            A date representing the start date of the data to load
        end_date : datetime
            A date representing the end date of the data to load

        Raises
        ------
        NotImplementedError
            If this method is not override.
        """
        raise NotImplementedError

    def check_column_names(self) -> bool:
        """
        Check if the minimal columns are all in the `df` attribute.

        Returns
        -------
        bool
            represents if all columns are in the `df` attribute

        Raises
        ------
        ValueError
            If the `df` attribute is equals to `None`
        """
        if self.df is None:
            raise ValueError(
                "To check column names, the df attribute must not be equals to None")
        return all([name in self.df.columns] for name in self.generic_columns)

    def rename_column_name(self, columns: dict) -> None:
        """
        Rename column names in the current `df` attribute value.

        Parameters
        ----------
        columns : dict
            dictionnary representing all the columns to change with
            key, the old name and it associated value, the new name

        Raises
        ------
        ValueError
            If the `df` attribute is equals to `None`

        Examples
        --------
        >>> api = BaseAPI(...)
        >>> api.load_data(...)
        >>> list(api.df.columns.value)
        ["A", "B", "C"]
        >>> api.rename_column_name({"A": "a", "B": "b"})
        >>> list(api.df.columns.value)
        ["a", "b", "C"]
        """
        if self.df is None:
            raise ValueError(
                "To rename columns, the df attribute must not be equals to None")
        self.df.rename(columns=columns, inplace=True, errors='raise')
