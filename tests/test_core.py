#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyVALION.library functions."""

import numpy as np

from PyVALION.library import nearest_element
from PyVALION.library import freq2den

def test_nearest_element_basic():
    arr = np.array([1, 3, 5, 7, 9])
    val = 6
    # 5 is closest to 6, at index 2
    assert nearest_element(arr, val) == 2

def test_nearest_element_exact_match():
    arr = np.array([10, 20, 30, 40])
    val = 30
    # Exact match at index 2
    assert nearest_element(arr, val) == 2

def test_nearest_element_negative_values():
    arr = np.array([-10, -5, 0, 5, 10])
    val = -3
    # -5 is closest to -3, at index 1
    assert nearest_element(arr, val) == 1

def test_nearest_element_float_input():
    arr = np.array([0.1, 0.5, 0.9])
    val = 0.4
    # 0.5 is closest to 0.4
    assert nearest_element(arr, val) == 1

def test_nearest_element_empty_array():
    arr = np.array([])
    val = 10
    try:
        nearest_element(arr, val)
        assert False, "Expected ValueError for empty array"
    except ValueError:
        pass  # expected

def test_freq2den_scalar():
    freq = 1  # MHz
    expected = 1.24e10  # m^-3
    assert freq2den(freq) == expected

def test_freq2den_array():
    freq = np.array([1, 2, 3])
    expected = 1.24e10 * freq**2
    result = freq2den(freq)
    np.testing.assert_allclose(result, expected)

def test_freq2den_zero():
    freq = 0
    expected = 0.0
    assert freq2den(freq) == expected

def test_freq2den_negative_input():
    freq = -2
    expected = 1.24e10 * freq**2  # Should still return positive density
    assert freq2den(freq) == expected
