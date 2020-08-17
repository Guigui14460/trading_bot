from trading_bot.apis.base_api import BaseAPI
from trading_bot.settings import ALPHA_VANTAGE_API_KEY
from alpha_vantage.timeseries import TimeSeries

class AlphaVantage(BaseAPI):

    def __init__(self):
        BaseAPI.__init__(self, None)

    def load_data(self, start_date, end_date, symbol: str) -> dict:
        ts = TimeSeries(key="ALPHA_VANTAGE_API_KEY", output_format="pandas")
        data, meta_data = ts.get_intraday(symbol, interval="1min")
        self.df = data
        dataDict = {"1. open": "Open", 
        "2. high": "High", 
        "3. low": "Low", 
        "4. close": "Close", 
        "5. volume": "Volume"
}
        self.rename_column_name(dataDict)
        return data



