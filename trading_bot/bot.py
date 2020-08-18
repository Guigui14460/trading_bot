from trading_bot.apis.alpha_vantage_api import AlphaVantage


class TradingBot:
    def run(self):
        api = AlphaVantage()
        data = api.load_data("2019-01-01", '2020-01-01', "MSFT")
        print(data)
        print(data.index)
