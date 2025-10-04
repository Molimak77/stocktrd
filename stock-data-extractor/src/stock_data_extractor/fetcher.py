# -*- coding: utf-8 -*-
"""
This module is responsible for fetching stock data from various sources.
"""

import logging
from datetime import date, datetime
from typing import Dict, Optional, Union

import pandas as pd
import pandas_market_calendars as mcal
import requests
from ib_insync import IB, Stock, util

from .config import config

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ExtraDataStock:
    """
    Fetches, manages, and provides historical stock data from multiple sources.

    Attributes:
        ticker (str): Stock ticker symbol in uppercase (e.g., 'AAPL').
        source (str): Data source used, in lowercase (e.g., 'yahoofinance').
        start_date (str): Calculated start date for data range (YYYY-MM-DD).
        end_date (str): Calculated end date for data range (YYYY-MM-DD).
        data (Optional[pd.DataFrame]): OHLCV data (Open, High, Low, Close, Volume).
        dico_stock (Dict[str, pd.DataFrame]): Maps data types (e.g., 'Close') to DataFrames.
    """

    def __init__(
        self,
        ticker: str,
        start: Union[int, str] = "2000-01-01",
        end: Optional[str] = None,
        source: str = "yahoofinance",
        yf_api_key: Optional[str] = None,
        ib_host: str = "127.0.0.1",
        ib_port: int = 7497,
        ib_client_id: int = 1,
    ):
        self.ticker = ticker.upper()
        self.source = source.lower()
        self.data: Optional[pd.DataFrame] = None
        self.dico_stock: Dict[str, pd.DataFrame] = {}
        self.end_date = end if end else date.today().strftime("%Y-%m-%d")
        self.start_date = self._calculate_start_date(start)

        if self.start_date > self.end_date:
            raise ValueError("Start date cannot be later than end date.")

        logging.info(
            f"Fetching data for {self.ticker} from source: '{self.source.upper()}'"
        )
        if self.source == "yahoofinance":
            api_key = yf_api_key or (config.yf_api_key if config else None)
            if not api_key:
                raise ValueError("API key required for 'yahoofinance' source.")
            self.data = self._fetch_yahoofinance_paid(api_key)
        elif self.source == "ibkr":
            self.data = self._fetch_ibkr(ib_host, ib_port, ib_client_id)
        else:
            raise ValueError("Source must be 'yahoofinance' or 'ibkr'.")

        if self.data is not None:
            logging.info("Data fetched successfully.")
            self.columns = self.data.columns.tolist()
            self.dico_stock = self._create_dico_from_data()
        else:
            logging.warning(f"Failed to fetch data for {self.ticker}.")
            self.columns = []

    def _calculate_start_date(self, start: Union[int, str]) -> str:
        if isinstance(start, str):
            return start
        if not isinstance(start, int) or start <= 0:
            raise TypeError(
                "Start must be a positive integer (trading days) or date string "
                "('YYYY-MM-DD')."
            )

        nyse = mcal.get_calendar("NYSE")
        end_dt = pd.to_datetime(self.end_date)
        lookback_days = int(start * 1.8 + 20)
        start_search_dt = end_dt - pd.Timedelta(days=lookback_days)
        trading_days = nyse.valid_days(start_date=start_search_dt, end_date=end_dt)

        if len(trading_days) < start:
            raise ValueError(
                f"Found {len(trading_days)} trading days before "
                f"{self.end_date}, but {start} requested."
            )

        return trading_days[-start].strftime("%Y-%m-%d")

    def _fetch_yahoofinance_paid(self, api_key: str) -> Optional[pd.DataFrame]:
        """Fetches data using the APIDOJO Yahoo Finance API on RapidAPI."""
        url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-chart"
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "apidojo-yahoo-finance-v1.p.rapidapi.com",
        }
        params = {
            "symbol": self.ticker,
            "region": "US",
            "interval": "1d",
            "period1": int(
                datetime.strptime(self.start_date, "%Y-%m-%d").timestamp()
            ),
            "period2": int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp()),
        }

        logging.info(f"Connecting to APIDOJO endpoint for {self.ticker}...")
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return self._parse_yahoo_json(response.json())
        except requests.exceptions.HTTPError as e:
            logging.error(
                f"HTTP Error for {self.ticker}: {e.response.status_code} "
                f"{e.response.reason}"
            )
            if e.response.status_code in [401, 403]:
                logging.error(
                    "Authentication failed. Check API key and subscription plan."
                )
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

    def _parse_yahoo_json(self, json_data: dict) -> Optional[pd.DataFrame]:
        try:
            chart_data = json_data["chart"]["result"][0]
            timestamps = chart_data["timestamp"]
            ohlc = chart_data["indicators"]["quote"][0]

            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(timestamps, unit="s").date,
                    "Open": ohlc["open"],
                    "High": ohlc["high"],
                    "Low": ohlc["low"],
                    "Close": ohlc["close"],
                    "Volume": ohlc["volume"],
                }
            )
            return df.dropna().reset_index(drop=True)
        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"Could not parse JSON response: {e}")
            return None

    def _fetch_ibkr(
        self, host: str, port: int, client_id: int
    ) -> Optional[pd.DataFrame]:
        logging.info(f"Connecting to Interactive Brokers on {host}:{port}...")
        try:
            ib = IB()
            ib.connect(host, port, clientId=client_id, readonly=True, timeout=10)
            contract = Stock(self.ticker, "SMART", "USD")
            ib.qualifyContracts(contract)
            duration_days = max(
                1,
                (
                    datetime.strptime(self.end_date, "%Y-%m-%d")
                    - datetime.strptime(self.start_date, "%Y-%m-%d")
                ).days,
            )
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=self.end_date,
                durationStr=f"{duration_days} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
            )
            ib.disconnect()

            if not bars:
                return None
            df = util.df(bars)
            df.rename(
                columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df[["Date", "Close", "High", "Low", "Open", "Volume"]]
        except Exception as e:
            logging.error(f"Interactive Brokers error: {e}")
            return None

    def _create_dico_from_data(self) -> Dict[str, pd.DataFrame]:
        if self.data is None:
            return {}
        dico = {}
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            if col in self.data.columns:
                dico[col] = self.data[["Date", col]].copy()
        return dico

    def get_stock_info(self) -> Dict[str, str]:
        """Return basic metadata about the loaded stock."""
        return {
            "Stock": self.ticker,
            "Start Date": self.start_date,
            "End Date": self.end_date,
            "Columns": ", ".join(self.columns),
        }

    def get_specific_data(self, data_type: str = "Close") -> Optional[pd.DataFrame]:
        """Retrieves a specific column of data (e.g., 'Close')."""
        key_to_find = data_type.lower()
        actual_key = next(
            (col for col in self.dico_stock.keys() if col.lower() == key_to_find), None
        )
        if actual_key:
            return self.dico_stock[actual_key]
        logging.warning(
            f"Data type '{data_type}' not found. "
            f"Available types: {list(self.dico_stock.keys())}"
        )
        return None