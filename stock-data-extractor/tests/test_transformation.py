# -*- coding: utf-8 -*-
import unittest

import numpy as np
from stock_data_extractor.transformation import TransformStock, VectorStock


class TestTransformation(unittest.TestCase):
    def test_vector_stock(self):
        # Test VectorStock initialization and basic properties
        data = [1, 2, 3, 4, 5]
        vector = VectorStock(data, window_size=3)
        self.assertEqual(len(vector), 3)
        self.assertEqual(vector[0], [1, 2, 3])
        self.assertEqual(vector[-1], [3, 4, 5])

    def test_transform_stock(self):
        # Test TransformStock for correct matrix transformation
        data = [1, 2, 3, 4, 5]
        transformer = TransformStock(data, window_size=3)
        matrix = transformer.matrix
        self.assertEqual(matrix.shape, (3, 3, 1))
        self.assertTrue(np.array_equal(matrix[0], np.array([[1], [2], [3]])))

    def test_transform_stock_split(self):
        # Test the split_xy method for creating training data
        data = [1, 2, 3, 4, 5]
        transformer = TransformStock(data, window_size=3)
        X, y = transformer.split_xy(for_training=True)

        # Assert shapes and values are correct for training
        self.assertEqual(X.shape, (2, 3, 1))
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)


if __name__ == "__main__":
    unittest.main()