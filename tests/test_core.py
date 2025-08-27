#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyVALION.library functions."""

import datetime
import numpy as np

from PyVALION.library import apply_median_filter
from PyVALION.library import compute_jason_tec
from PyVALION.library import compute_lanczos_kernel
from PyVALION.library import concat_data_dicts
from PyVALION.library import downsample_dict
from PyVALION.library import extract_cycle_number
from PyVALION.library import extract_first_date
from PyVALION.library import freq2den
from PyVALION.library import lanczos_filter
from PyVALION.library import make_empty_dict_data_jason
from PyVALION.library import nearest_element
from PyVALION.library import robust_iterative_filter
from PyVALION.library import round_and_stringify


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
    """Test extract_first_date valid."""
    url = ("https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycle000/JA2_GPN_2PdP000_074_20080704_234513_20080705_004126.nc")
    expected = datetime.datetime(2008, 7, 4, 23, 45, 13)
    assert extract_first_date(url) == expected


def test_extract_first_date_invalid():
    """Test extract_first_date invalid."""
    url = ("https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycle000/JA2_GPN_2PdP000_074.nc")
    expected = datetime.datetime.max
    assert extract_first_date(url) == expected


def test_extract_cycle_number_valid():
    """Test extract_cycle_number valid."""
    url = ("https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycle000/JA2_GPN_2PdP000_074_20080704_234513_20080705_004126.nc")
    expected = 0
    assert extract_cycle_number(url) == expected


def test_extract_cycle_number_missing():
    """Test extract_cycle_number missing."""
    url = ("https://www.ncei.noaa.gov/thredds-ocean/dodsC/jason2/gdr/gdr/"
           + "cycleINV/JA2_GPN_2PdP000_074_20080704_234513_20080705_004126.nc")
    expected = -1
    assert extract_cycle_number(url) == expected


def test_compute_jason_tec_zero():
    """Test compute_jason_tec zero."""
    iono_ku = 0.0
    expected_tec = 0
    assert compute_jason_tec(iono_ku) == expected_tec


def test_compute_jason_tec_positive_negative():
    """Test compute_jason_tec postive and negative."""
    assert compute_jason_tec(1.0) < 0
    assert compute_jason_tec(-1.0) > 0


def test_make_empty_dict_data_jason():
    """Test make_empty_dict_data_jason zero."""
    d = make_empty_dict_data_jason()
    assert set(d.keys()) == {"dtime", "TEC", "lon", "lat", "name"}
    assert d["TEC"].size == 0


def test_concat_data_dicts_success():
    """Test concat_data_dicts success."""
    A = {"dtime": np.array([1]), "TEC": np.array([2.0]),
         "lon": np.array([3.0]), "lat": np.array([4.0]),
         "name": np.array(["JA2"])}
    B = {"dtime": np.array([5]), "TEC": np.array([6.0]),
         "lon": np.array([7.0]), "lat": np.array([8.0]),
         "name": np.array(["JA3"])}
    C = concat_data_dicts(A, B)
    assert C["TEC"].tolist() == [2.0, 6.0]


def test_concat_data_dicts_dtype_mismatch():
    """Test concat_data_dicts dtype mismatch."""
    A = {"TEC": np.array([1.0])}
    B = {"TEC": np.array([1], dtype=int)}
    try:
        concat_data_dicts(A, B)
        assert False, "Expected TypeError for mismatched types"
    except TypeError:
        pass  # expected


def test_concat_data_dicts_value_error():
    """Test concat_data_dicts value error."""
    A = {"x": np.array([1, 2, 3])}
    B = {"x": [4, 5, 6]}  # not a numpy array
    try:
        concat_data_dicts(A, B)
        assert False, "Expected ValueError because not a numpy array"
    except ValueError:
        pass  # expected


def test_compute_lanczos_kernel_properties():
    """Test compute_lanczos_kernel properties."""
    k = compute_lanczos_kernel(5, 3)
    assert np.isclose(k.sum(), 1.0)
    assert len(k) == 11


def test_lanczos_filter_nan_handling():
    """Test lanczos_filter nan handling."""
    x = np.array([1, np.nan, 3, 4, 5], dtype=float)
    y = lanczos_filter(x, 2, 3)
    assert not np.all(np.isnan(y))


def test_robust_iterative_filter_outliers():
    """Test robust_iterative_filter outlier handling."""
    data = np.array([1, 1, 1, 100, 1, 1], dtype=float)
    filtered, mask = robust_iterative_filter(data, INIT_SIGMA=2)
    assert mask[3]  # big outlier flagged


def test_apply_median_filter_preserves_nans():
    """Test apply_median_filter nan preserving."""
    data = np.array([1, np.nan, 3], dtype=float)
    result = apply_median_filter(data, 1, 1)
    assert np.isnan(result[1])


def test_round_and_stringify():
    """Test round_and_stringify success."""
    lat = np.array([12.34, -12.34])
    lon = np.array([45.67, -45.67])
    coor_str, lat_r, lon_r = round_and_stringify(lat, lon, 0.5)
    assert coor_str.shape == lat.shape
    assert np.allclose(lat_r % 0.5, 0)


def test_downsample_dict():
    """Test downsample_dict success."""
    d = {"a": np.arange(10), "b": np.arange(10)}
    res = downsample_dict(d, 3)
    assert (res["a"] == [0, 3, 6, 9]).all()
