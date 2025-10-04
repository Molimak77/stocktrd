# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from stock_data_extractor.processing import StockProcessing


class TestStockProcessing(unittest.TestCase):
    @patch("stock_data_extractor.processing.ExtraDataStock.__init__")
    def setUp(self, mock_extra_data_stock_init):
        # Mock the base class initializer to avoid actual data fetching
        mock_extra_data_stock_init.return_value = None

        # Create an instance of StockProcessing
        self.stock_processor = StockProcessing(ticker="AAPL")

        # Create sample data for testing
        self.sample_data = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "Open": [150.0, 151.0],
                "High": [155.0, 156.0],
                "Low": [149.0, 150.0],
                "Close": [153.0, 154.0],
                "Volume": [1000000, 1100000],
            }
        )
        self.stock_processor.data = self.sample_data
        self.stock_processor.dico_stock = {
            "Open": self.sample_data[["Date", "Open"]],
            "Close": self.sample_data[["Date", "Close"]],
        }
        self.stock_processor.ticker = "AAPL"

    @patch("stock_data_extractor.processing.px.line")
    def test_graph_stock(self, mock_px_line):
        # Mock the line function from plotly.express
        mock_fig = MagicMock()
        mock_px_line.return_value = mock_fig

        # Call the function to be tested
        self.stock_processor.graph_stock("Open")

        # Assert that the plotting function was called with the correct arguments
        mock_px_line.assert_called_once_with(
            self.stock_processor.dico_stock["Open"],
            x="Date",
            y="Open",
            title="AAPL - Open",
        )
        mock_fig.show.assert_called_once_with(renderer="browser")

    @patch("stock_data_extractor.processing.go.Figure")
    def test_graph_ohlc(self, mock_go_figure):
        # Mock the Figure object from plotly.graph_objects
        mock_fig = MagicMock()
        mock_go_figure.return_value = mock_fig

        # Call the function to be tested
        self.stock_processor.graph_ohlc()

        # Assert that the figure's update_layout was called with the correct title
        mock_fig.update_layout.assert_called()
        self.assertIn(
            "AAPL - Open, High, Low, Close Prices",
            mock_fig.update_layout.call_args[1]["title"],
        )
        mock_fig.show.assert_called_once_with(renderer="browser")

    def test_feature_scaling(self):
        # Test the feature scaling functionality
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        scaled_df, scaler = StockProcessing.feature_scaling(df, "value")

        # Assert that the scaled values are within the expected range [0, 1]
        self.assertTrue((scaled_df["value"] >= 0).all() and (scaled_df["value"] <= 1).all())


if __name__ == "__main__":
    unittest.main()