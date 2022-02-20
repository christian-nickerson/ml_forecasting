from models.base import ModelBase
from sklearn.pipeline import Pipeline
from data import StockData
from xgboost import XGBRegressor
import numpy as np


class XGB(ModelBase):

    """XGBoost Model."""

    def __init__(self, data: StockData) -> None:
        """XGBoost class initialiser"""
        pass

    @staticmethod
    def build() -> XGBRegressor:
        """Return model instance

        :return: XGBoost Regressor instance
        :rtype: XGBRegressor
        """
        return XGBRegressor(verbosity=0, seed=123, tree_method="hist")

    @staticmethod
    def preprocess() -> Pipeline:
        """Pre-processing steps for XGBoost regressor

        :return: Pipeline for pre-procesing
        :rtype: Pipeline
        """
        return None

    @staticmethod
    def params() -> dict:
        """Return pipeline hyperparameters to tune.

        :return: parameter dictionary
        :rtype: dict
        """
        return {
            # tree parameters
            "model__max_depth": np.arange(2, 16, 2),
            "model__subsample": np.arange(0.3, 1.0, 0.1),
            "model__colsample_bytree": np.arange(0.3, 1.0, 0.1),
            "model__colsample_bylevel": np.arange(0.3, 1.0, 0.1),
            "model__objective": ["reg:squarederror", "count:poisson"],
            "model__learning_rate": [0.1, 0.01, 0.0001],
            "model__n_estimators": [100, 1000, 10000],
            # regularisation parameters
            "model__reg_alpha": [0],
            "model__reg_lambda": [0],
            "model__gamma": [0],
            # fit parameters
            "model__nthread": [4],
            "model__early_stopping_rounds": [25],
        }

    def fit_params(self) -> dict:
        """Return pipeline model fit paramters

        :return: parameter dictionary
        :rtype: dict
        """
        return {}


if __name__ == "__main__":
    pass
