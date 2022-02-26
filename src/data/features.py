from pandas import DataFrame
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings


class FeatureEngineering:

    """Create features from stock data."""

    @staticmethod
    def _time_features(df: DataFrame) -> DataFrame:
        """Create time features from dataframe."""
        df["day_of_year"] = df.index.dayofyear
        df["day_of_month"] = df.index.day
        df["day_of_week"] = df.index.dayofweek
        return df

    @staticmethod
    def _closing_lags(df: DataFrame) -> DataFrame:
        """Create closing price lags."""
        for lags in range(1, 30):
            df[f"close_lag_{lags}"] = df["Close"].shift(lags)
        return df

    @staticmethod
    def _closing_sma(df: DataFrame) -> DataFrame:
        """Create closing simple moving average."""
        for window in [5, 10, 30, 60, 90]:
            df[f"close_sma_{window}"] = df["Close"].rolling(window).mean()
        return df

    @staticmethod
    def _closing_ses(df: DataFrame) -> DataFrame:
        """Create closing simple exponential smoothing features."""
        warnings.filterwarnings("ignore")
        for alpha in [0.2, 0.4, 0.6, 0.8, None]:
            se = SimpleExpSmoothing(df["Close"])
            se_fitted = se.fit(smoothing_level=alpha)
            df[f"close_ses_{alpha}"] = se_fitted.predict(0)
        return df

    def build_features(df: DataFrame) -> DataFrame:
        """Build engineered features."""
        df = FeatureEngineering._time_features(df)
        df = FeatureEngineering._closing_lags(df)
        df = FeatureEngineering._closing_sma(df)
        df = FeatureEngineering._closing_ses(df)
        df = df.dropna()
        return df


if __name__ == "__main__":
    pass
