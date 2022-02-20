from data import StockData
from models import ModelRegistry
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import pathlib
import joblib


class ModelTrain:

    """Class for managing the training and hyper paramter tuning of a model"""

    def __init__(self, model_name: str, data: StockData) -> None:
        """Model training class

        :param model_name: Name of model to train
        :type model_name: str
        :param data: stock data to train model on
        :type data: StockData
        """
        self.model_name = model_name
        self.data = data
        self._one_hot_encode_data()
        self.model_registry = ModelRegistry()
        self.model = self.model_registry.get_model(model_name, data)

    def _one_hot_encode_data(self) -> None:
        """One hot encode data class."""
        if self.model_name not in ["xgboost"]:
            self.data.ohe_cat_cols()

    def _estimator(self) -> Pipeline:
        """Return pipeline estimator.

        :return: Sklearn pipeline estimator
        :rtype: Pipeline
        """
        return Pipeline([("preprocessing", self.model.preprocess()), ("model", self.model.build())])

    def _pipeline(self, parameter_samples: int) -> BayesSearchCV:
        """Build training & parameter tuning pipeline.

        :param parameter_samples: Number of samples to select from paramter space
            for parameter tuning.
        :type parameter_samples: int

        :return: Bayesian search parameter tunining pipeline
        :rtype: BayesSearchCV
        """
        return BayesSearchCV(
            estimator=self._estimator(),
            search_spaces=self.model.params(),
            scoring="neg_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=5),
            n_iter=parameter_samples,
            n_jobs=-1,
            n_points=5,
            verbose=0,
        )

    def _write_model(self, pipeline: BayesSearchCV) -> None:
        """Write model to artifact library.

        :param pipeline: traineed pipeline instance to write
        :type pipeline: BayesSearchCV
        """
        pathlib.Path(f"artifacts/{self.data.stock_symbol}/").mkdir(parents=True, exist_ok=True)
        model_file_name = f"artifacts/{self.data.stock_symbol}/{self.model_name}.sav"
        joblib.dump(pipeline, model_file_name)

    def train(self, parameter_samples: int) -> None:
        """Train model on stock data and save to artifact library.

        :param parameter_samples: Number of samples to select from paramter space
            for parameter tuning.
        :type parameter_samples: int
        """
        self.pipeline = self._pipeline(parameter_samples)
        self.pipeline.fit(self.data.stock_x_train, self.data.stock_y_train, **self.model.fit_params())
        self._write_model(self.pipeline)


if __name__ == "__main__":
    pass
