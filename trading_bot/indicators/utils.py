import pandas as pd


def wwma(values: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Welle Wilder's Exponential Moving Average.

    Parameters
    ----------
    values : pd.Series
        serie of data
    period : int
        number of period
    """
    return values.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
