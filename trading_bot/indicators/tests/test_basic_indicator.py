import os
import unittest

import numpy as np
import pandas as pd

from trading_bot.indicators import (
    HighestHigh, LowestLow, MedianPrice,
    TypicalPrice, AverageTrueRange,
)
from trading_bot.settings import DATA_PATH


class BasicIndicatorTest(unittest.TestCase):
    """
    Test class used to check the calculation of all the basic indicators classes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(os.path.join(DATA_PATH, "MSFT.csv"))

    def test_highest_high(self) -> None:
        hh = HighestHigh()
        hh.calculate_in_place(self.data)
        df2 = self.data.copy().dropna()
        self.assertTrue(np.all(df2['High'] <= df2[hh.get_column_name()]))

    def test_lowest_low(self) -> None:
        ll = LowestLow()
        ll.calculate_in_place(self.data)
        df2 = self.data.copy().dropna()
        self.assertTrue(np.all(df2['Low'] >= df2[ll.get_column_name()]))

    def test_median_price(self):
        mp = MedianPrice()
        mp.calculate_in_place(self.data)
        for i in range(len(self.data[mp.get_column_name()])):
            median = (self.data['High'][i] + self.data['Low'][i]) / 2
            self.assertTrue(median == self.data[mp.get_column_name()][i])

    def test_typical_price(self):
        tp = TypicalPrice()
        tp.calculate_in_place(self.data)
        for i in range(len(self.data[tp.get_column_name()])):
            typical = (self.data['High'][i] + self.data['Low']
                       [i] + self.data['Close'][i]) / 3
            self.assertTrue(typical == self.data[tp.get_column_name()][i])


if __name__ == '__main__':
    unittest.main()
