from data import StockData
from pipeline import ModelTrain
from reporting import StockChart
from common import Log
from datetime import timedelta
from time import time
from dotenv import load_dotenv
import os


class StockPricePrediction:

    """Stock Price Prediction runner class."""

    def __init__(self, stock_symbol: str, model_name: str, data_years: int = 10) -> None:
        """Initialise stock price prediction.

        :param stock_symbol: symbol to predict price for.
        :type stock_symbol: str
        :param model_name: model name to use for prediction.
        :type model_name: str
        :param data_years: years of historical data to train model on, defaults to 10
        :type data_years: int, optional
        """
        self.logger = Log.set_logger(f"stock prediction: {stock_symbol}")
        self.stock_symbol = stock_symbol
        self.model_name = model_name
        self.data_years = data_years

    @staticmethod
    def load_env_vars() -> None:
        """Load local .env file if in root dir."""
        files = [f for f in os.listdir(".") if os.path.isfile(f)]
        if ".env" in files:
            load_dotenv()

    def fetch_data(self) -> StockData:
        """Return Stock data for training.

        :return: StockData for modelling
        :rtype: StockData
        """
        self.logger.info(f"fechting price data")
        return StockData(self.stock_symbol, self.data_years)

    def train_model(self, data: StockData, param_samples: int = 100) -> None:
        """Train model for prediction.

        :param data: StockData instance for trianing.
        :type data: StockData
        """
        start = time()
        self.logger.info(f"training {self.model_name}")
        model = ModelTrain(self.model_name, data)
        model.train(param_samples)
        self.logger.info(f"training complete: {timedelta(seconds = time() - start)}")

    def model_report(self, data: StockData) -> None:
        """Create model report.

        :param data: StockData instance for trianing.
        :type data: StockData
        """
        chart = StockChart(self.model_name, data)
        self.logger.info(f"creating {self.model_name} report")
        chart.create_report()


if __name__ == "__main__":

    StockPricePrediction.load_env_vars()

    stock_prediction = StockPricePrediction(
        stock_symbol=os.getenv("STOCK_SYMBOL"),
        model_name=os.getenv("MODEL_NAME"),
        data_years=int(os.getenv("DATA_YEARS")),
    )

    stock_data = stock_prediction.fetch_data()
    stock_prediction.train_model(stock_data, int(os.getenv("PARAM_SAMPLES")))
    stock_prediction.model_report(stock_data)
