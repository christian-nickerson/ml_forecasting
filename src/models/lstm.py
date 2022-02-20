from models.base import ModelBase
from models.transformers import ArrayTransformer, StandardScalerNumericColsOnly
from data import StockData
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.optimizer_v2.adam import Adam
import tensorflow as tf
import numpy as np
import os


class LSTMNetwork(ModelBase):

    """LSTM Neural Network Model."""

    def __init__(self, data: StockData) -> None:
        """LSTM Network."""
        self._quiet_mode(True)
        self.features = len(data.get_x_cols())

    @staticmethod
    def _quiet_mode(toggle: bool = False) -> None:
        """Set Keras logging to lowest level"""
        if toggle:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            tf.autograph.set_verbosity(3)
            tf.get_logger().setLevel("ERROR")

    def build(self) -> KerasRegressor:
        return KerasRegressor(build_fn=self.compile, epochs=100000, verbose=0)

    @staticmethod
    def preprocess() -> Pipeline:
        """Pre-processing steps for dense newtork regressor

        :return: Pipeline for pre-procesing
        :rtype: Pipeline
        """
        return Pipeline(
            [
                ("scaler", StandardScalerNumericColsOnly()),
                ("array", ArrayTransformer()),
            ]
        )

    def compile(self, layers: int, learning_rate: float, drop_out_rate: float) -> Sequential:
        """Compile network for KerasRegressor

        :param layers: number of layers to build
        :type layers: int
        :param learning_rate: learning rate of network
        :type learning_rate: float
        :param drop_out_rate: drop out regularisation rate
        :type drop_out_rate: float

        :return: nueral network model
        :rtype: Sequential
        """
        model = Sequential()
        for _ in range(layers - 1):
            model.add(LSTM(self.features, activation="relu", input_shape=(1, self.features), return_sequences=True))
            model.add(Dropout(drop_out_rate))
        model.add(Dense(1))
        adam = Adam(learning_rate=learning_rate)
        model.compile(loss="mean_squared_error", optimizer=adam)
        return model

    @staticmethod
    def params() -> dict:
        """LSTM hyperparameters"""
        return {
            "model__layers": np.arange(1, 10, 1),
            "model__batch_size": [7, 30, 60, 90, 180, 365],
            "model__learning_rate": [0.001, 0.0001, 0.00001],
            "model__drop_out_rate": np.arange(0, 0.5, 0.05),
        }

    def fit_params(self) -> dict:
        """LSTM fit parameters"""
        stopping = EarlyStopping(monitor="loss", patience=50)
        return {"model__callbacks": [stopping]}


if __name__ == "__main__":
    pass
