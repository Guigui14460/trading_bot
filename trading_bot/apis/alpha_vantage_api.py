from alpha_vantage.timeseries import TimeSeries
import pandas as pd

from trading_bot.settings import ALPHA_VANTAGE_API_KEY
from .base_api import BaseAPI


class AlphaVantage(BaseAPI):
    def __init__(self) -> None:
        BaseAPI.__init__(self, None)

    def load_data(self, start_date: str, end_date: str, symbol: str) -> pd.DataFrame:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        data, meta_data = ts.get_intraday(symbol, interval="1min")
        self.df = data
        dataDict = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }
        self.rename_column_name(dataDict)
        return data
