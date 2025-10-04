# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from stock_data_extractor.fetcher import ExtraDataStock


class TestExtraDataStock(unittest.TestCase):
    @patch("stock_data_extractor.fetcher.requests.get")
    def test_fetch_yahoofinance_paid_success(self, mock_get):
        # Mocking the successful response from Yahoo Finance API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1672531200],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [150.0],
                                    "high": [155.0],
                                    "low": [149.0],
                                    "close": [153.0],
                                    "volume": [1000000],
                                }
                            ]
                        },
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        # Initializing ExtraDataStock with mocked data
        stock = ExtraDataStock(
            ticker="AAPL",
            start="2023-01-01",
            end="2023-01-02",
            source="yahoofinance",
            yf_api_key="fake_api_key",
        )

        # Assertions to verify data integrity
        self.assertIsNotNone(stock.data)
        self.assertEqual(stock.data["Open"][0], 150.0)
        self.assertEqual(stock.data["Close"][0], 153.0)

    @patch("stock_data_extractor.fetcher.IB")
    def test_fetch_ibkr_success(self, mock_ib):
        # Mocking the successful connection and data retrieval from Interactive Brokers
        mock_ib_instance = MagicMock()
        mock_ib_instance.reqHistoricalData.return_value = [
            MagicMock(
                date="2023-01-01",
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=1000000,
            )
        ]
        mock_ib.return_value = mock_ib_instance

        # Initializing ExtraDataStock for IBKR
        stock = ExtraDataStock(
            ticker="AAPL",
            start="2023-01-01",
            end="2023-01-02",
            source="ibkr",
        )

        # Assertions to ensure data is correctly processed
        self.assertIsNotNone(stock.data)
        self.assertEqual(stock.data["Open"][0], 150.0)
        self.assertEqual(stock.data["Close"][0], 153.0)


if __name__ == "__main__":
    unittest.main()