#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyVALION.library functions."""

import datetime
import numpy as np

from PyVALION.library import freq2den
from PyVALION.library import nearest_element
from PyVALION.library import extract_first_date
from PyVALION.library import extract_cycle_number


def test_nearest_element_basic():
    """Test nearest_element basic."""
    arr = np.array([1, 3, 5, 7, 9])
    val = 6
    # 5 is closest to 6, at index 2
    assert nearest_element(arr, val) == 2


def test_nearest_element_exact_match():
    """Test nearest_element exact."""
    arr = np.array([10, 20, 30, 40])
    val = 30
    # Exact match at index 2
    assert nearest_element(arr, val) == 2


def test_nearest_element_negative_values():
    """Test nearest_element negative."""
    arr = np.array([-10, -5, 0, 5, 10])
    val = -3
    # -5 is closest to -3, at index 1
    assert nearest_element(arr, val) == 1


def test_nearest_element_float_input():
    """Test nearest_element float."""
    arr = np.array([0.1, 0.5, 0.9])
    val = 0.4
    # 0.5 is closest to 0.4
    assert nearest_element(arr, val) == 1


def test_nearest_element_empty_array():
    """Test nearest_element empty."""
    arr = np.array([])
    val = 10
    try:
        nearest_element(arr, val)
        assert False, "Expected ValueError for empty array"
    except ValueError:
        pass  # expected


def test_freq2den_scalar():
    """Test freq2den scalar."""
    freq = 1  # MHz
    expected = 1.24e10  # m^-3
    assert freq2den(freq) == expected


def test_freq2den_array():
    """Test freq2den array."""
    freq = np.array([1, 2, 3])
    expected = 1.24e10 * freq**2
    result = freq2den(freq)
    np.testing.assert_allclose(result, expected)


def test_freq2den_zero():
    """Test freq2den zero."""
    freq = 0
    expected = 0.0
    assert freq2den(freq) == expected


def test_freq2den_negative_input():
    """Test freq2den negative."""
    freq = -2
    expected = 1.24e10 * freq**2  # Should still return positive density
    assert freq2den(freq) == expected


# Test core for Jason TEC code
def test_extract_first_date_valid():
    url = ["https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycle000/JA2_GPN_2PdP000_074_20080704_234513_20080705_004126.nc"]
    expected = datetime.datetime(2008, 7, 4, 23, 45, 13)
    assert extract_first_date(url) == expected


def test_extract_first_date_invalid():
    url = ["https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycle000/JA2_GPN_2PdP000_074_20080704_234513_20080705_004126.nc"]
    expected = datetime.datetime.max
    assert extract_first_date(url) == expected


def test_extract_cycle_number_valid():
    url = ["https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycle000/JA2_GPN_2PdP000_074_20080704_234513_20080705_004126.nc"]
    expected = 0
    assert extract_cycle_number(url) == expected


def test_extract_cycle_number_missing():
    url = ["https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycleINV/JA2_GPN_2PdP000_074_20080704_234513_20080705_004126.nc"]
    expected = -1
    assert extract_cycle_number(url) == expected
