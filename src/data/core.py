from datetime import date, timedelta
from dataclasses import dataclass
from pandas import DataFrame
from data.features import FeatureEngineering as fe
import yfinance as yf
import pandas as pd


@dataclass
class StockData:

    stock_symbol: str
    stock_years: int
    stock_df: DataFrame
    stock_x: DataFrame
    stock_y: DataFrame
    stock_x_train: DataFrame
    stock_y_train: DataFrame
    stock_x_test: DataFrame
    stock_y_test: DataFrame

    def __init__(self, stock_symbol: str, stock_years: int) -> None:
        """Stock data class for containing stock data for modelling

        :param stock_symbol: Stock symbol to collect data for.
        :type stock_symbol: str
        :param stock_years: number of years to collect data for.
        :type stock_years: int
        """
        self.stock_symbol = stock_symbol
        self.stock_years = stock_years
        self.stock_df = self._data_extract()
        self.stock_df = fe.build_features(self.stock_df)
        self.stock_df = self._clean_df()
        self.stock_x, self.stock_y = self.x_y_split(self.stock_df)
        self.stock_x_train, self.stock_x_test = self.train_test_split(self.stock_x, 0.2)
        self.stock_y_train, self.stock_y_test = self.train_test_split(self.stock_y, 0.2)

    def _data_extract(self) -> DataFrame:
        """Extract stock data from yahoo finance

        :return: stock data dataframe
        :rtype: DataFrame
        """
        today = date.today()
        today_min = date.today() - timedelta(days=365 * self.stock_years)
        return yf.download(self.stock_symbol, end=today, start=today_min)

    def _clean_df(self) -> DataFrame:
        """Clean stock dataframe.

        :return: cleaned dataframe
        :rtype: DataFrame
        """
        df = self.stock_df.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis=1)
        df = df.dropna()
        return df

    @staticmethod
    def train_test_split(df: DataFrame, test_pct: float) -> tuple[DataFrame, DataFrame]:
        """Create time based train test split.

        :param df: Dataframe to create train & test datasets from
        :type df: DataFrame
        :param test_pct: test percentage to use (will be taken from end of timeseries)
        :type test_pct: float
        :return: train & test dataframes
        :rtype: tuple[DataFrame, DataFrame]
        """
        test_days = round(len(df.index) * test_pct)
        test = df[df.index >= df.index.max() - timedelta(days=test_days)]
        train = df[df.index < df.index.max() - timedelta(days=test_days)]
        return train, test

    @staticmethod
    def x_y_split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Create X and y datasets.

        :param df: Dataframe to create X & y datasets from
        :type df: DataFrame
        :return: X & y dataframes
        :rtype: tuple[DataFrame, DataFrame]
        """
        x = df.drop(["Close"], axis=1)
        y = df["Close"]
        return x, y

    def get_x_cols(self) -> list:
        """return X columns data.

        :return: list of x column names
        :rtype: list
        """
        return self.stock_x.columns

    @staticmethod
    def _onehotencode(df: DataFrame, cols_to_encode: list) -> DataFrame:
        """One hot encoder"""
        return pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode)

    def ohe_cat_cols(self) -> None:
        """One hot encode categorical variables."""
        ohe_cols = [col for col in self.stock_x if col.startswith("day")]
        self.stock_x = self._onehotencode(self.stock_x, ohe_cols)
        self.stock_x_train, self.stock_x_test = self.train_test_split(self.stock_x, 0.2)


if __name__ == "__main__":
    pass
