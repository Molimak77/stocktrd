# -*- coding: utf-8 -*-
"""
This module provides classes for transforming stock data into formats
suitable for time series analysis.
"""
from typing import Any, List, Sequence, Tuple

import numpy as np


class VectorStock(Sequence[List[Any]]):
    """
    Creates an iterator and sequence-like view over a sliding window of data.
    """

    def __init__(self, data: Sequence[Any], window_size: int) -> None:
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if len(data) < window_size:
            raise ValueError("Data length cannot be smaller than window_size.")
        self.data = data
        self.window_size = window_size
        self._len = len(self.data) - self.window_size + 1

    def __len__(self) -> int:
        """Returns the total number of windows."""
        return self._len

    def __getitem__(self, index: int) -> List[Any]:
        """
        Retrieves a window at a specific index, supporting negative indexing.
        """
        if isinstance(index, slice):
            raise TypeError("Slicing a VectorStock object is not supported.")
        if not -self._len <= index < self._len:
            raise IndexError("Index out of range.")
        if index < 0:
            index += self._len
        return list(self.data[index : index + self.window_size])

    def __setitem__(self, index: int, value: List[Any]) -> None:
        """Assigns a new list to a window, modifying the underlying data."""
        if not -self._len <= index < self._len:
            raise IndexError("Index out of range.")
        if len(value) != self.window_size:
            raise ValueError(
                f"Assigned value must have length {self.window_size}, got {len(value)}."
            )
        if index < 0:
            index += self._len
        self.data[index : index + self.window_size] = value

    def __repr__(self) -> str:
        """Provides an unambiguous string representation."""
        return f"VectorStock(data={self.data!r}, window_size={self.window_size})"


class TransformStock:
    """
    Transforms 1D sequence into a 3D data matrix for time series models.
    """

    def __init__(
        self, data: Sequence[Any], window_size: int, ref_period: bool = False
    ):
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if len(data) < window_size:
            raise ValueError("Data length cannot be smaller than window_size.")

        data_arr = np.asarray(data, dtype=float)
        shape = (data_arr.shape[0] - window_size + 1, window_size)
        strides = (data_arr.strides[0], data_arr.strides[0])
        matrix_2d = np.lib.stride_tricks.as_strided(
            data_arr, shape=shape, strides=strides
        )
        self._matrix = matrix_2d[:, :, np.newaxis]

        if ref_period:
            self._matrix = self._matrix[-1:, :, :]

    @property
    def matrix(self) -> np.ndarray:
        """Returns the internal 3D data matrix."""
        return self._matrix

    def __add__(self, other: "TransformStock") -> "TransformStock":
        """Combines two TransformStock objects by concatenating features."""
        if not isinstance(other, TransformStock):
            return NotImplemented
        if self.matrix.shape[:2] != other.matrix.shape[:2]:
            raise ValueError(
                "Objects must have same number of samples and timesteps."
            )
        new_3d_matrix = np.concatenate((self.matrix, other.matrix), axis=2)
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._matrix = new_3d_matrix
        return new_instance

    def split_xy(
        self, target_feature_index: int = 0, for_training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Splits data into features (X) and target (y)."""
        X = self.matrix
        y = X[:, 0, target_feature_index]
        if for_training:
            if len(X) <= 1:
                return np.array([]), np.array([])
            X = X[1:]
            y = y[:-1]
        return X, y