# Stock Data Extractor Documentation

## Introduction
The `stock-data-extractor` is a Python package designed to simplify the process of fetching, processing, and transforming stock market data from various financial sources. This documentation provides a comprehensive overview of the package's architecture, features, and usage.

## Core Components
The package is organized into four main modules:

### 1. `fetcher.py`
This module is responsible for retrieving historical stock data. The `ExtraDataStock` class serves as the primary interface for this functionality, supporting multiple data sources like Yahoo Finance and Interactive Brokers.

### 2. `processing.py`
The `processing.py` module contains the `StockProcessing` class, which is designed for data analysis and visualization. It allows users to plot various stock indicators and apply feature scaling techniques to normalize the data.

### 3. `transformation.py`
This module includes the `VectorStock` and `TransformStock` classes, which are essential for preparing data for time series forecasting models. These classes help in creating sliding windows of data and transforming them into the required 3D matrix format.

### 4. `config.py`
The `config.py` module manages the package's configuration, primarily handling API keys and other sensitive information through environment variables.

## Getting Started
To begin using the `stock-data-extractor`, you first need to install it. Once installed, you can import the necessary classes and start fetching data. For detailed instructions on installation and usage, please refer to the `README.md` file.