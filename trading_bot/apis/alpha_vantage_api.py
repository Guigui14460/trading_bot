from .base_api import BaseAPI
from trading_bot.settings import ALPHA_VANTAGE_API_KEY
import alpha_vantage 

class AlphaVantage(BaseAPI):

    def __init__(self):
        BaseAPI.__init__(self, None)

    def load_data(self, endpoint, start_date, end_date):
        return super().load_data(endpoint, start_date, end_date)

