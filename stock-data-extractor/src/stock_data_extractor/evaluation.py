import os
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn import metrics

if os.getenv("CI"):
    pio.renderers.default = "json"
else:
    pio.renderers.default = "browser"

pd.options.mode.chained_assignment = None


class EvalModel:
    """A class to evaluate model predictions against true values, providing metrics and visualizations.

    This class supports evaluating time-series predictions by computing standard metrics,
    generating plots, and analyzing trends through confusion matrices. It is designed for
    stock price or similar time-series data evaluation.

    Attributes:
        y_true (list[float]): The true values of the time-series.
        y_pred (list[float] | None): The predicted values, if provided.
        periode (int): The period for calculating differences (default is 1).
        dataset (pd.DataFrame): DataFrame containing true and predicted values.
        data_metric (pd.DataFrame | None): DataFrame with computed metrics, if built.
    """

    def __init__(self, y_true: list[float], y_pred: list[float] | None = None, periode: int = 1):
        """Initialize the EvalModel with true and optional predicted values.

        Args:
            y_true (list[float]): The true values of the time-series.
            y_pred (list[float] | None): The predicted values, if available (default: None).
            periode (int): The period for differencing in metric calculations (default: 1).

        Raises:
            ValueError: If lengths of y_true and y_pred do not match when y_pred is provided.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.periode = periode

        if self.y_pred is not None and len(self.y_true) != len(self.y_pred):
            raise ValueError("Length of y_true and y_pred must be equal")

        data = {"y_true": self.y_true}
        if self.y_pred is not None:
            data["y_pred"] = self.y_pred
        self.dataset = pd.DataFrame(data)
        self.data_metric = None

    def print(self) -> None:
        """Print the dataset and period information.

        Displays the DataFrame containing true and predicted values, followed by the period.
        """
        print(self.dataset)
        print(f"Period: {self.periode}")

    def get_info(self) -> dict:
        """Retrieve metadata about the dataset and model.

        Returns:
            dict: A dictionary containing y_true, y_pred, periode, and dataset shape.
        """
        return {
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "periode": self.periode,
            "dataset_shape": self.dataset.shape,
        }

    def graph_evaluation(
        self,
        title_name: str,
        save_fig: bool = False,
        path_save: str | None = None,
        name_graph: str | None = None,
    ) -> str | None:
        """Generate and optionally save a line plot comparing true and predicted values.

        Args:
            title_name (str): The title of the plot.
            save_fig (bool): Whether to save the plot as an HTML file (default: False).
            path_save (str | None): Directory to save the plot (default: current working directory).
            name_graph (str | None): Name of the output file (default: timestamp-based name).

        Returns:
            str | None: Path to the saved HTML file if save_fig is True, else None.
        """
        df = self.dataset.copy()
        df["Time"] = list(range(len(self.y_true)))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df["y_true"],
                mode="lines+markers",
                name="Real value",
                line=dict(color="blue"),
            )
        )
        if "y_pred" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Time"],
                    y=df["y_pred"],
                    mode="lines+markers",
                    name="Estimator value",
                    line=dict(color="red"),
                    marker=dict(symbol="star"),
                )
            )

        fig.update_layout(
            title=title_name,
            xaxis_title="Time",
            yaxis_title="Ticker value",
            legend=dict(x=1, y=0, xanchor="right", yanchor="bottom"),
            showlegend=True,
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        path_save = os.getcwd() if path_save is None else path_save
        if name_graph is None:
            current_time = datetime.now().strftime("%d-%m-%y-%H-%M-%S")
            name_graph = f"graph_{current_time}"

        path_save_fig = os.path.join(path_save, f"{name_graph}.html")

        if save_fig:
            os.makedirs(path_save, exist_ok=True)
            fig.write_html(path_save_fig)

        if not os.getenv("CI"):
            fig.show()

        return path_save_fig if save_fig else None

    def _safe_division(self, numerator: float, denominator: float) -> float | None:
        """Perform division with handling for division by zero.

        Args:
            numerator (float): The numerator of the division.
            denominator (float): The denominator of the division.

        Returns:
            float | None: The result of the division, or None if denominator is zero.
        """
        try:
            return numerator / denominator
        except ZeroDivisionError:
            print(f"Warning: Division by zero encountered with denominator {denominator}")
            return None

    def prediction_evaluation(self) -> pd.Series:
        """Compute standard regression metrics for the predictions.

        Calculates metrics such as explained variance, R² score, mean absolute error,
        and mean squared error.

        Returns:
            pd.Series: A Series containing the computed metrics.
        """
        if self.y_pred is None:
            raise ValueError("y_pred is required for prediction evaluation")

        standard_metrics = pd.Series(dtype="float64")

        standard_metrics["explained_variance"] = metrics.explained_variance_score(
            self.dataset["y_true"], self.dataset["y_pred"]
        )
        standard_metrics["r2_score"] = metrics.r2_score(self.dataset["y_true"], self.dataset["y_pred"])

        df = self.dataset.copy()
        df["y_true_squared"] = df["y_true"] ** 2
        df["y_pred_squared"] = df["y_pred"] ** 2

        df["abs_error"] = df.apply(
            lambda row: self._safe_division(abs(row["y_true"] - row["y_pred"]), row["y_true"]), axis=1
        )
        df["mse_error"] = df.apply(
            lambda row: self._safe_division(
                abs(row["y_true_squared"] - row["y_pred_squared"]), row["y_true_squared"]
            ),
            axis=1,
        )

        standard_metrics["mean_absolute_error"] = df["abs_error"].mean()
        standard_metrics["std_absolute_error"] = df["abs_error"].std()
        standard_metrics["mean_squared_error"] = df["mse_error"].mean()
        standard_metrics["std_squared_error"] = df["mse_error"].std()

        return standard_metrics

    @staticmethod
    def produit_sign(x1: float, x2: float) -> str:
        """Determine the sign product of two numbers.

        Args:
            x1 (float): First number.
            x2 (float): Second number.

        Returns:
            str: A string representing the sign combination (e.g., '++', '+-', '00').
        """
        def signe(a: float) -> str:
            if a == 0:
                return "0"
            return "+" if np.sign(a) == 1 else "-"

        return f"{signe(x1)}{signe(x2)}"

    def build_data_metric(self, periode: int | None = None) -> "EvalModel":
        """Build a DataFrame with differenced metrics for true and predicted values.

        Computes differences over the specified period for trend analysis.

        Args:
            periode (int | None): Period for differencing (default: self.periode).

        Returns:
            EvalModel: The instance itself with updated data_metric attribute.
        """
        if periode is None:
            periode = self.periode
        else:
            self.periode = periode

        df = self.dataset.copy()
        df["delta_ytf"] = df["y_true"].diff(periods=periode)
        df["delta_ypf"] = df["y_pred"].diff(periods=periode) if self.y_pred is not None else pd.Series()

        df["delta_yt"] = df["delta_ytf"] / periode
        df["delta_yp"] = df["delta_ypf"] / periode if self.y_pred is not None else pd.Series()

        self.data_metric = df.dropna()
        return self

    def metric(self) -> pd.DataFrame:
        """Compute RMSE, MAE, R2, and std deviation of residuals."""
        if self.y_pred is None:
            raise ValueError("y_pred must be provided to compute metrics")

        residuals = np.array(self.y_true) - np.array(self.y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(self.y_true, self.y_pred))
        mae = metrics.mean_absolute_error(self.y_true, self.y_pred)
        r2 = metrics.r2_score(self.y_true, self.y_pred)
        std_error = np.std(residuals)

        self.data_metric = pd.DataFrame(
            {"RMSE": [rmse], "MAE": [mae], "R2": [r2], "STD_Error": [std_error]}
        )
        return self.data_metric

    def get_classification(self, rho: float = 0.0, periode: int = 1) -> pd.DataFrame:
        """Classify trends based on true and predicted value differences.

        Assigns classifications (e.g., true positive slope, false negative slope) based
        on the sign and magnitude of differences.

        Args:
            rho (float): Threshold for considering slopes equivalent (default: 0.0).
            periode (int): Period for differencing (default: 1).

        Returns:
            pd.DataFrame: DataFrame with classification column.
        """
        def map_confusion_mat(a: float, b: float, rho: float = 0.0) -> str:
            value = abs(a - b)
            sign = self.produit_sign(a, b)

            if sign in ["++", "+-", "+0"]:
                if sign == "++" and value < rho:
                    return "Tps"
                if sign == "++" and value > rho:
                    return "Fps/ps"
                if sign == "+-" and value > rho:
                    return "Fps/ns"
                if sign == "+0" and value > rho:
                    return "Fps/cs"
            elif sign in ["-+", "--", "-0"]:
                if sign == "--" and value < rho:
                    return "Tns"
                if sign == "--" and value > rho:
                    return "Fns/ns"
                if sign == "-+" and value > rho:
                    return "Fns/ps"
                if sign == "-0" and value > rho:
                    return "Fns/cs"
            elif sign in ["0+", "0-", "00"]:
                if sign == "00" and value < rho:
                    return "Tcs"
                if sign == "00" and value > rho:
                    return "Fcs/cs"
                if sign == "0+" and value > rho:
                    return "Fcs/ps"
                if sign == "0-" and value > rho:
                    return "Fcs/ns"
            return "undefined"

        obj = self.build_data_metric(periode=periode)
        df = obj.data_metric
        df["classification"] = df[["delta_yt", "delta_yp"]].apply(
            lambda x: map_confusion_mat(x["delta_yt"], x["delta_yp"], rho=rho), axis=1
        )
        return df

    def confusion_matrix(self, rho: float = 0.0, periode: float = 1.0) -> tuple:
        """Generate confusion matrices and metrics for trend classification.

        Creates matrices for positive, negative, and constant slopes, along with
        summary metrics like accuracy and rate of correct classifications.

        Args:
            rho (float): Threshold for slope equivalence (default: 0.0).
            periode (float): Period for differencing (default: 1.0).

        Returns:
            tuple: A tuple containing:
                - dict: Confusion matrices for positive, negative, and constant slopes.
                - pd.Series: Summary metrics (accuracy, rates).
                - dict: Classification counts.
        """
        df = self.get_classification(rho=rho, periode=int(periode))
        classifications = {
            "Tps",
            "Fps/ps",
            "Fps/ns",
            "Fps/cs",
            "Tns",
            "Fns/ns",
            "Fns/ps",
            "Fns/cs",
            "Tcs",
            "Fcs/cs",
            "Fcs/ps",
            "Fcs/ns",
        }
        dico_clf = {cls: len(df[df["classification"] == cls]) for cls in classifications}

        positive_matrix = pd.DataFrame(
            {
                "Fps/ps": [dico_clf["Fps/ps"]],
                "Tps": [dico_clf["Tps"]],
                "Fps/ns": [dico_clf["Fps/ns"]],
                "Fps/cs": [dico_clf["Fps/cs"]],
            },
            index=["Eval positive slope"],
        )

        negative_matrix = pd.DataFrame(
            {
                "Fns/ns": [dico_clf["Fns/ns"]],
                "Fns/ps": [dico_clf["Fns/ps"]],
                "Tns": [dico_clf["Tns"]],
                "Fns/cs": [dico_clf["Fns/cs"]],
            },
            index=["Eval negative slope"],
        )

        constant_matrix = pd.DataFrame(
            {
                "Fcs/cs": [dico_clf["Fcs/cs"]],
                "Fcs/ps": [dico_clf["Fcs/ps"]],
                "Fcs/ns": [dico_clf["Fcs/ns"]],
                "Tcs": [dico_clf["Tcs"]],
            },
            index=["Eval constant slope"],
        )

        sum_p = sum([dico_clf[k] for k in ["Fps/ps", "Tps", "Fps/ns", "Fps/cs"]])
        sum_n = sum([dico_clf[k] for k in ["Fns/ns", "Fns/ps", "Tns", "Fns/cs"]])
        sum_c = sum([dico_clf[k] for k in ["Fcs/cs", "Fcs/ps", "Fcs/ns", "Tcs"]])
        if sum_c == 0:
            print("Warning: there is no constant slope detected")
            sum_c = 1
        sum_md = sum_p + sum_n + sum_c

        dico_matrix = {
            "Eval positive slope": positive_matrix,
            "Eval negative slope": negative_matrix,
            "Eval constant slope": constant_matrix,
        }
        dico_matrix1 = pd.DataFrame(
            {"--":['Est Ps', 'Est Ns' ,'Est Cs'],
            "Real Ps":[positive_matrix[1], positive_matrix[2],positive_matrix[3]+positive_matrix[0]],
            "Real Ns":[negative_matrix[1], negative_matrix[2],negative_matrix[3]+negative_matrix[0]],
            "Real Cs":[constant_matrix[1], constant_matrix[2],constant_matrix[3]+constant_matrix[0]]
            })

        metrics_summary = pd.Series(dtype="float64")
        metrics_summary["Accuracy"] = np.round(
            100 * (dico_clf["Tps"] + dico_clf["Tns"] + dico_clf["Tcs"]) / sum_md, 3
        )
        metrics_summary["Rate positive slope"] = np.round(100 * dico_clf["Tps"] / (sum_p or 1), 3)
        metrics_summary["Relative rate positive slope"] = np.round(
            100 * dico_clf["Tps"] / sum_md, 3
        )
        metrics_summary["Rate negative slope"] = np.round(100 * dico_clf["Tns"] / (sum_n or 1), 3)
        metrics_summary["Relative rate negative slope"] = np.round(
            100 * dico_clf["Tns"] / sum_md, 3
        )
        metrics_summary["Rate constant slope"] = np.round(100 * dico_clf["Tcs"] / sum_c, 3)
        metrics_summary["Relative constant slope"] = np.round(100 * dico_clf["Tcs"] / sum_md, 3)

        return dico_matrix1, metrics_summary, dico_clf