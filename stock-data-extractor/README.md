# Stock Data Extractor

`stock-data-extractor` is a powerful and easy-to-use Python package for fetching, processing, and transforming stock market data from various sources.

## Features
- **Multiple Data Sources**: Fetch data from Yahoo Finance and Interactive Brokers.
- **Data Processing**: Visualize stock data with interactive plots.
- **Feature Scaling**: Normalize data using `MinMaxScaler` and `StandardScaler`.
- **Data Transformation**: Reshape data for time series analysis and machine learning models.

## Installation
You can install the package directly from PyPI:
```bash
pip install stock-data-extractor
```

## Configuration
To use the Yahoo Finance data source, you need to provide an API key. You can do this in two ways:
1. **Environment Variable**: Create a `.env` file in your project's root directory and add the following line:
   ```
   YF_API_KEY=your_api_key_here
   ```
2. **Directly in Code**: Pass the API key as an argument when creating a `StockProcessing` instance:
   ```python
   from stock_data_extractor.processing import StockProcessing

   stock_processor = StockProcessing(ticker="AAPL", yf_api_key="your_api_key_here")
   ```

## Quick Start
Here is a simple example of how to fetch and plot stock data:
```python
from stock_data_extractor.processing import StockProcessing

# Initialize with your ticker and API key
stock_processor = StockProcessing(
    ticker="TSLA",
    start="2022-01-01",
    end="2023-01-01",
    yf_api_key="YOUR_API_KEY"  # Replace with your key
)

# Plot the closing price
if stock_processor.data is not None:
    stock_processor.graph_stock("Close")
```

## Documentation
For more detailed information on the package's features and functionalities, please refer to the full [documentation](./docs/documentation.md).