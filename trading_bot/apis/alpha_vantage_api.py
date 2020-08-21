from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd
import time

from trading_bot.settings import ALPHA_VANTAGE_API_KEY
from .base_api import BaseAPI


class AlphaVantage(BaseAPI):
    def __init__(self) -> None:
        BaseAPI.__init__(self, None)

    def load_data(self, start_date: str, end_date: str, symbol: str) -> pd.DataFrame:
        """ This method will load datas from any market by requesting Alpha_Vantage website.

        Args:
            start_date (str): contains the starting date 
            end_date (str): contains the ending date 
            symbol (str): name of a Corp. 

        Returns:
            pd.DataFrame: contains all the datas that we want
        """
      
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        data, meta_data = ts.get_intraday(symbol, interval="1min", outputsize="full")
        self.df = data
        dataDict = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }
        self.rename_column_name(dataDict)

        expected_output = self.df[np.logical_and(self.df.index >= start_date, self.df.index <= end_date)]

        return expected_output

    def update_datas(self, datas: dict) -> pd.DataFrame: 
        pass
        
        


        


