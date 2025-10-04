# -*- coding: utf-8 -*-
"""
This module provides classes for processing and visualizing stock market data.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .fetcher import ExtraDataStock


class StockProcessing(ExtraDataStock):
    """
    Processes, visualizes, and scales stock market data.
    """

    def graph_stock(self, param_stock: str = "Open") -> None:
        """Plot a selected stock indicator using Plotly."""
        if not self.dico_stock or param_stock not in self.dico_stock:
            raise ValueError(
                f"Invalid column '{param_stock}'. Available: "
                f"{', '.join(self.dico_stock.keys())}"
            )
        df = self.dico_stock[param_stock]
        fig = px.line(
            df, x="Date", y=param_stock, title=f"{self.ticker} - {param_stock}"
        )
        fig.show(renderer="browser")

    def graph_ohlc(self, columns_to_plot: Optional[List[str]] = None) -> None:
        """Creates a plot overlaying specified price columns."""
        if self.data is None:
            print("No data available to graph.")
            return
        plot_list = (
            ["Open", "High", "Low", "Close"]
            if columns_to_plot is None
            else columns_to_plot
        )
        valid_columns = [col for col in plot_list if col in self.data.columns]
        for col in plot_list:
            if col not in valid_columns:
                print(
                    f"Warning: Column '{col}' not found in data and will be ignored."
                )
        if not valid_columns:
            print("Error: None of the requested columns are available to plot.")
            return

        print(f"Generating plot for: {', '.join(valid_columns)}...")
        fig = go.Figure()
        for col in valid_columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data["Date"], y=self.data[col], mode="lines", name=col
                )
            )
        title_str = ", ".join(valid_columns)
        fig.update_layout(
            title=f"{self.ticker} - {title_str} Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Price Type",
            template="plotly_white",
        )
        fig.show(renderer="browser")

    @staticmethod
    def feature_scaling(
        df: pd.DataFrame,
        name: str,
        scale_type: int = 1,
        scaler: Optional[object] = None,
    ) -> Tuple[pd.DataFrame, object]:
        """Apply feature scaling to a DataFrame column."""
        if name not in df.columns:
            raise ValueError(f"Column '{name}' not found in DataFrame.")
        df_copy = df.copy()
        if scaler is None:
            scaler = MinMaxScaler() if scale_type == 1 else StandardScaler()
            df_copy[name] = scaler.fit_transform(df_copy[[name]])
        else:
            df_copy[name] = scaler.transform(df_copy[[name]])
        return df_copy, scaler

    def transf_featur_scaling(
        self,
        scale_type: int = 1,
        scaler_list: Optional[Dict[str, object]] = None,
        pred_npoints: int = 0,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object], Dict[str, pd.DataFrame]]:
        """Apply feature scaling to all stock indicators."""
        dico_data_scaling: Dict[str, pd.DataFrame] = {}
        dico_scaler: Dict[str, object] = {}
        dico_original_data: Dict[str, pd.DataFrame] = {}

        for key, df in self.dico_stock.items():
            name = df.columns[1]
            truncated_df = df.iloc[-pred_npoints:] if pred_npoints > 0 else df
            dico_original_data[key] = truncated_df
            scaler = scaler_list.get(f"fct_scaling{name}") if scaler_list else None
            scaled_df, fitted_scaler = self.feature_scaling(
                truncated_df, name, scale_type=scale_type, scaler=scaler
            )
            dico_data_scaling[f"dat{name}"] = scaled_df
            dico_scaler[f"fct_scaling{name}"] = fitted_scaler

        return dico_data_scaling, dico_scaler, dico_original_data