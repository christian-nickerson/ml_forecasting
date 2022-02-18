from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame


class ArrayTransformer(TransformerMixin, BaseEstimator):

    """Create 3D array from dataframe required by LSTM nework."""

    def __init__(self):
        pass

    def fit(self, X: DataFrame, y: DataFrame = None):
        return self

    def transform(self, X: DataFrame, y: DataFrame = None):
        """Loop throuhg X DataFrame columns and apply filter"""
        return X.to_numpy().reshape(X.shape[0], 1, X.shape[1])


class StandardScalerNumericColsOnly(TransformerMixin, BaseEstimator):

    """Standard scale all but one hot encoded data."""

    def __init__(self):
        pass

    def fit(self, X: DataFrame, y: DataFrame = None):
        return self

    def transform(self, X: DataFrame, y: DataFrame = None):
        """Scale data for all but one hot encoded data."""
        scaler = StandardScaler()
        non_ohe_cols = [col for col in X if not col.startswith("day")]
        X[non_ohe_cols] = scaler.fit_transform(X[non_ohe_cols])
        return X


class MinMaxScalerNumericColsOnly(TransformerMixin, BaseEstimator):

    """Min max scale all but one hot encoded data."""

    def __init__(self):
        pass

    def fit(self, X: DataFrame, y: DataFrame = None):
        return self

    def transform(self, X: DataFrame, y: DataFrame = None):
        """Scale data for all but one hot encoded data."""
        scaler = MinMaxScaler()
        non_ohe_cols = [col for col in X if not col.startswith("day")]
        X[non_ohe_cols] = scaler.fit_transform(X[non_ohe_cols])
        return X


if __name__ == "__main__":
    pass
