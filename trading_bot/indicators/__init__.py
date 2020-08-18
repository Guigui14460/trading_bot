"""Indicators module.

This module provides a lot of known indicators used by a large amount of traders.

You can create simplely your own indicators.

This module requires `pandas` be installed within the Python 
environment you are running this script in.

This file can be imported as a module and contains of the following 
classes:

    * Indicator - abstract class represents all indicator possibilities
    * BasicIndicator - abstract class used to reprensent the base of each basic indicator that we will use
    * HighestHigh - class used to reprensent "Highest High" indicator
    * LowestLow - class used to reprensent "Lowest Low" indicator
    * MedianPrice - class used to reprensent "Median price" indicator
    * TypicalPrice - class used to reprensent "Typical price" indicator
    * AverageTrueRange - class used to reprensent "Average True Range" or "ATR" indicator
"""

from .base_indicator import Indicator
from .basic_indicator import (
    BasicIndicator, HighestHigh,
    LowestLow, MedianPrice,
    TypicalPrice, AverageTrueRange,
)

__all__ = [
    'Indicator',
    'BasicIndicator',
    'HighestHigh',
    'LowestLow',
    'MedianPrice',
    'TypicalPrice',
    'AverageTrueRange',
]
