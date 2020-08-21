from trading_bot.apis.alpha_vantage_api import AlphaVantage


class TradingBot:
    def run(self):
        api = AlphaVantage()
        data = api.load_data("2020-08-20 18:00:00", '2020-08-20 18:50:00', "MSFT")
        print(data)
        print(data.index)
