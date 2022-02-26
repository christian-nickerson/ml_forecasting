from typing import Any
from data import StockData
from pandas import DataFrame
from plotly.subplots import make_subplots
from keras.models import load_model
from common import TarZip
import plotly.graph_objects as go
import pathlib
import joblib
import os


class StockChart:

    """Stock charting for measuring modelling performance."""

    def __init__(self, model_name: str, data: StockData) -> None:
        """Stock chart initialiser"""
        self.model_name = model_name
        self.data = data

    def _load_model(self) -> None:
        """Load model from artifact library"""
        self._load_joblib()
        tf_models = ["lstm"]
        if self.model_name in tf_models:
            self._load_keras()

    def _load_joblib(self) -> None:
        """Load joblib model from artifact library"""
        model_file_name = f"artifacts/{self.data.stock_symbol}/{self.model_name}.sav"
        self.model = joblib.load(model_file_name)

    def _load_keras(self) -> None:
        """Load keras estimator from artifact library"""
        file_directory = f"artifacts/{self.data.stock_symbol}/"
        TarZip.extract(file_directory + f"{self.model_name}.tar.gz", ".")
        self.model.best_estimator_.named_steps["model"].model = load_model(file_directory + f"{self.model_name}.h5")
        os.remove(file_directory + f"{self.model_name}.h5")

    @staticmethod
    def _prepare_df(y: DataFrame, pred: DataFrame) -> DataFrame:
        """Prepare dataframe from graphing"""
        df = DataFrame(index=y.index)
        df["y"] = y
        df["pred"] = pred
        df["residuals"] = df["pred"] - df["y"]
        return df

    def _inference(self) -> None:
        """create test inference data"""
        train_pred = self.model.predict(self.data.stock_x_train)
        test_pred = self.model.predict(self.data.stock_x_test)
        self.train_df = self._prepare_df(self.data.stock_y_train, train_pred)
        self.test_df = self._prepare_df(self.data.stock_y_test, test_pred)

    def _save_fig(self, fig: Any) -> None:
        """Write and show figure"""
        pathlib.Path(f"reports/{self.data.stock_symbol}/").mkdir(parents=True, exist_ok=True)
        fig.write_html(f"reports/{self.data.stock_symbol}/{self.model_name}.html")
        fig.show()

    def create_report(self) -> None:
        """Create report"""
        self._load_model()
        self._inference()

        # histogram residual
        train_histogram = go.Histogram(x=self.train_df["residuals"], name="Train hist. residuals")
        test_histogram = go.Histogram(x=self.test_df["residuals"], name="Test hist. residuals")

        # scatter residual
        train_scatter = go.Scatter(
            x=self.train_df["y"],
            y=self.train_df["pred"],
            mode="markers",
            name="Train fit residuals",
        )
        test_scatter = go.Scatter(
            x=self.test_df["y"],
            y=self.test_df["pred"],
            mode="markers",
            name="Test fit residuals",
        )

        # timeseries residuals
        train_timeseries = go.Scatter(
            x=self.train_df.index,
            y=self.train_df["residuals"],
            name="Train timeseries residuals",
        )
        test_timeseries = go.Scatter(
            x=self.test_df.index,
            y=self.test_df["residuals"],
            name="Test timeseries residuals",
        )

        # performance
        train_performance_y = go.Scatter(
            x=self.train_df.index,
            y=self.train_df["y"],
            name="Train actual closing price",
        )
        train_performance_pred = go.Scatter(
            x=self.train_df.index,
            y=self.train_df["pred"],
            name="Train predicted closing price",
        )
        test_performance_y = go.Scatter(
            x=self.test_df.index,
            y=self.test_df["y"],
            name="Test actual closing price",
        )
        test_performance_pred = go.Scatter(
            x=self.test_df.index,
            y=self.test_df["pred"],
            name="Test predicted closing price",
        )

        fig = make_subplots(
            rows=5,
            cols=2,
            vertical_spacing=0.05,
            specs=[[{"colspan": 2}, None], [{"colspan": 2}, None], [{}, {}], [{}, {}], [{}, {}]],
            subplot_titles=(
                "Closing price: Actual vs Predicted (Train)",
                "Closing price: Actual vs Predicted (Test)",
                "Training Residuals Histogram",
                "Testing Residuals Histogram",
                "Training Fitted Residuals",
                "Testing Fitted Residuals",
                "Training Residuals over Time",
                "Testing Residuals over Time",
            ),
        )

        fig.add_trace(train_performance_y, row=1, col=1)
        fig.add_trace(train_performance_pred, row=1, col=1)

        fig.add_trace(test_performance_y, row=2, col=1)
        fig.add_trace(test_performance_pred, row=2, col=1)

        fig.add_trace(train_histogram, row=3, col=1)
        fig.add_trace(test_histogram, row=3, col=2)

        fig.add_trace(train_scatter, row=4, col=1)
        fig.add_trace(test_scatter, row=4, col=2)

        # update scatter with 1:1 line
        fig.add_shape(
            type="line",
            line=dict(dash="dash"),
            x0=self.train_df["y"].min(),
            y0=self.train_df["pred"].min(),
            x1=self.train_df["y"].max(),
            y1=self.train_df["pred"].max(),
            yref="paper",
            row=4,
            col=1,
        )
        fig.add_shape(
            type="line",
            line=dict(dash="dash"),
            x0=self.test_df["y"].min(),
            y0=self.test_df["pred"].min(),
            x1=self.test_df["y"].max(),
            y1=self.test_df["pred"].max(),
            yref="paper",
            row=4,
            col=2,
        )

        fig.add_trace(train_timeseries, row=5, col=1)
        fig.add_trace(test_timeseries, row=5, col=2)

        title = f"<b>{self.data.stock_symbol}: {self.model_name}</b>"
        fig.update_layout(height=2000, width=1600, template="plotly_dark", title_text=title)
        self._save_fig(fig)


if __name__ == "__main__":
    pass
