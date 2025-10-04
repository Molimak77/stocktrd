# -*- coding: utf-8 -*-
"""
stock_data_extractor

A Python package for fetching, processing, and analyzing stock market data from various sources.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import AppConfig
from .fetcher import ExtraDataStock
from .processing import StockProcessing
from .transformation import TransformStock, VectorStock

__all__ = [
    "AppConfig",
    "ExtraDataStock",
    "StockProcessing",
    "TransformStock",
    "VectorStock",
]