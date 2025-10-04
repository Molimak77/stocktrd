# -*- coding: utf-8 -*-
"""
This script demonstrates how to use the stock_data_extractor package
to fetch, process, and transform stock data.
"""
from stock_data_extractor.processing import StockProcessing
from stock_data_extractor.transformation import TransformStock


from stock_data_extractor.evaluation import EvalModel


def main():
    """
    Main function to demonstrate package functionalities.
    """
    # Initialize StockProcessing to fetch data for a specific ticker
    # NOTE: This example uses a mock API key. Replace with a valid key for real use.
    try:
        stock_processor = StockProcessing(
            ticker="GOOGL",
            start="2022-01-01",
            end="2023-01-01",
            source="yahoofinance",
            yf_api_key="YOUR_API_KEY",  # Replace with your actual API key
        )
    except ValueError as e:
        print(f"Error initializing StockProcessing: {e}")
        print(
            "Please ensure you have a .env file with YF_API_KEY or pass it as an argument."
        )
        return

    # Check if data was fetched successfully
    if stock_processor.data is None:
        print("Failed to fetch data. Please check your API key and network connection.")
        return

    # Print basic information about the fetched stock data
    print("Stock Info:", stock_processor.get_stock_info())

    # Visualize the closing price of the stock
    print("Displaying graph of the closing price...")
    stock_processor.graph_stock("Close")

    # Get the 'Close' price data as a pandas DataFrame
    close_prices_df = stock_processor.get_specific_data("Close")
    if close_prices_df is not None:
        close_prices = close_prices_df["Close"].values
        # Transform the data into a format suitable for time series models
        transformer = TransformStock(close_prices, window_size=10)
        X, y = transformer.split_xy(for_training=True)

        print("\nTransformed Data Shapes:")
        print("Features (X) shape:", X.shape)
        print("Target (y) shape:", y.shape)

        # --- Demonstrate EvalModel ---
        # Create dummy predictions for demonstration
        # In a real scenario, these would come from a trained model
        y_pred = y * 0.95  # Simulate predictions with a 5% error

        print("\n--- Model Evaluation ---")
        evaluator = EvalModel(y_true=list(y), y_pred=list(y_pred))

        # Get and print evaluation metrics
        metrics = evaluator.metric()
        print("\nEvaluation Metrics:")
        print(metrics)

        # Generate and print the confusion matrix
        matrix, summary, _ = evaluator.confusion_matrix()
        print("\nConfusion Matrix:")
        print(matrix)
        print("\nMetrics Summary:")
        print(summary)

        # Plot the evaluation
        print("\nDisplaying evaluation graph...")
        evaluator.graph_evaluation(title_name="Model Performance: GOOGL")


if __name__ == "__main__":
    main()