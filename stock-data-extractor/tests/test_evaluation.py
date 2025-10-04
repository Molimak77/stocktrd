# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
from stock_data_extractor.evaluation import EvalModel


class TestEvalModel(unittest.TestCase):
    def setUp(self):
        """Set up a simple EvalModel instance for testing."""
        self.y_true = [100, 102, 105, 103, 106]
        self.y_pred = [101, 103, 104, 104, 105]
        self.eval_model = EvalModel(self.y_true, self.y_pred)

    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.eval_model.y_true, self.y_true)
        self.assertEqual(self.eval_model.y_pred, self.y_pred)
        self.assertEqual(self.eval_model.periode, 1)
        self.assertIsInstance(self.eval_model.dataset, pd.DataFrame)
        self.assertIn("y_true", self.eval_model.dataset.columns)
        self.assertIn("y_pred", self.eval_model.dataset.columns)

    def test_metric_calculation(self):
        """Test the metric calculation method."""
        metrics_df = self.eval_model.metric()
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertIn("RMSE", metrics_df.columns)
        self.assertIn("MAE", metrics_df.columns)
        self.assertIn("R2", metrics_df.columns)

    def test_build_data_metric(self):
        """Test the build_data_metric method."""
        self.eval_model.build_data_metric()
        self.assertIsNotNone(self.eval_model.data_metric)
        self.assertIsInstance(self.eval_model.data_metric, pd.DataFrame)
        self.assertIn("delta_yt", self.eval_model.data_metric.columns)
        self.assertIn("delta_yp", self.eval_model.data_metric.columns)
        self.assertEqual(len(self.eval_model.data_metric), len(self.y_true) - 1)

    def test_get_classification(self):
        """Test the get_classification method."""
        classification_df = self.eval_model.get_classification()
        self.assertIsInstance(classification_df, pd.DataFrame)
        self.assertIn("classification", classification_df.columns)
        # Check if classifications are among the expected values
        expected_classifications = {
            "Tps", "Fps/ps", "Fps/ns", "Fps/cs",
            "Tns", "Fns/ns", "Fns/ps", "Fns/cs",
            "Tcs", "Fcs/cs", "Fcs/ps", "Fcs/ns",
            "undefined"
        }
        self.assertTrue(set(classification_df["classification"].unique()).issubset(expected_classifications))

    def test_confusion_matrix(self):
        """Test the confusion_matrix method."""
        dico_matrix, metrics_summary, dico_clf = self.eval_model.confusion_matrix()
        self.assertIsInstance(dico_matrix, pd.DataFrame)
        self.assertIsInstance(metrics_summary, pd.Series)
        self.assertIsInstance(dico_clf, dict)
        self.assertIn("Accuracy", metrics_summary.index)


if __name__ == "__main__":
    unittest.main()