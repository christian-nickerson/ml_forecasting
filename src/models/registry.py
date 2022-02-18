from data import StockData
from models.base import ModelBase
from models.xgboost import XGB
from models.lstm import LSTMNetwork


class ModelRegistry(object):

    """Model Registry (factory pattern) class for registering and building models."""

    def __init__(self) -> None:
        """Model registry"""
        super().__init__()
        self.model_registry = {}
        self.compile_models()

    def register_model(self, model_name: str, model: ModelBase) -> None:
        """Register model within registry

        :param model_name: name of model
        :type model_name: str
        :param model: model class
        :type model: ModelBase
        :param model_category: regressor or timeseries
        :type model_category: str
        """
        self.model_registry[model_name] = model

    def compile_models(self) -> None:
        """Compile models into class"""
        self.register_model("lstm", LSTMNetwork)
        self.register_model("xgboost", XGB)

    def get_model(self, model_name: str, data: StockData) -> ModelBase:
        """Return model from registry.

        :param model_name: Name of model to to return from registry.
        :param data: Stock data instantiate model with.
        """
        return self.model_registry[model_name](data)


if __name__ == "__main__":
    pass
