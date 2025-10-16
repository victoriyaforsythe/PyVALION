#!/usr/bin/env python
# --------------------------------------------------------
"""Unit tests for PyVALION.library functions."""

import datetime
import numpy as np
import pandas as pd

from PyVALION.library import apply_median_filter
from PyVALION.library import compute_jason_tec
from PyVALION.library import compute_lanczos_kernel
from PyVALION.library import compute_solzen
from PyVALION.library import concat_data_dicts
from PyVALION.library import downsample_dict
from PyVALION.library import extract_cycle_number
from PyVALION.library import extract_first_date
from PyVALION.library import fill_small_gaps
from PyVALION.library import find_Jason_G_and_y
from PyVALION.library import find_Jason_residuals
from PyVALION.library import freq2den
from PyVALION.library import lanczos_filter
from PyVALION.library import make_empty_dict_data_jason
from PyVALION.library import mask_dict
from PyVALION.library import nearest_element
from PyVALION.library import robust_iterative_filter
from PyVALION.library import round_and_stringify
from PyVALION.library import sza_data_space


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


def test_mask_dict():
    """Test mask_dict filters dict by boolean mask."""
    d = {"x": np.array([1, 2, 3]), "y": np.array([10, 20, 30])}
    mask = np.array([True, False, True])
    out = mask_dict(d, mask)
    assert np.all(out["x"] == np.array([1, 3]))
    assert np.all(out["y"] == np.array([10, 30]))


def test_downsample_dict():
    """Test downsample_dict success."""
    d = {"a": np.arange(10), "b": np.arange(10)}
    res = downsample_dict(d, 3)
    assert (res["a"] == [0, 3, 6, 9]).all()


def test_fill_small_gaps():
    """Test fill_small_gaps fills NaN gaps."""
    data = np.array([1.0, np.nan, 3.0])
    result = fill_small_gaps(data, 1)
    assert np.isfinite(result[1])
    assert result[0] == 1.0
    assert result[-1] == 3.0


def test_find_Jason_G_and_y_basic():
    """Test find_Jason_G_and_y builds G and y."""
    adtime = np.array([datetime.datetime(2020, 1, 1, 0, 0),
                       datetime.datetime(2020, 1, 1, 1, 0)])
    alon = np.array([0.0, 10.0])
    alat = np.array([0.0, 10.0])
    data = {
        "dtime": np.array([datetime.datetime(2020, 1, 1, 0, 30)]),
        "lat": np.array([5.0]),
        "lon": np.array([5.0]),
        "TEC": np.array([10.0]),
        "name": np.array(["JA2"])
    }
    y, units, G = find_Jason_G_and_y(adtime, alon, alat, data)
    assert "TEC" in y
    assert "TEC" in units
    assert G.ndim == 4


def test_find_Jason_residuals():
    """Test find_Jason_residuals computes residuals."""
    model = {"TEC": np.ones((1, 2, 2))}
    G = np.zeros((3, 1, 2, 2))
    obs_data = {"TEC": np.array([1.0, 2.0, 3.0])}
    units = {"TEC": "TECU"}
    _, residuals, model_units = find_Jason_residuals(model, G, obs_data, units)
    assert isinstance(residuals["TEC"], np.ndarray)
    assert model_units["TEC"] == "TECU"


def test_compute_solzen_single_day():
    """Test compute_solzen outputs valid solar zenith angles."""
    time_start = datetime.datetime(2020, 1, 1)
    adtime = np.array([datetime.datetime(2020, 1, 1, 0, 0),
                       datetime.datetime(2020, 1, 1, 12, 0)])
    adtime = pd.to_datetime(adtime)
    alon = np.array([0.0, 10.0])
    alat = np.array([0.0, 10.0])
    solzen = compute_solzen(time_start, 90, adtime, alon, alat)
    assert solzen.shape == (len(adtime),)
    assert np.all(np.isfinite(solzen))


def test_sza_data_space_mismatched_lengths():
    """Test sza_data_space with mismatched input lengths."""
    dtime = np.array([pd.Timestamp("2025-01-01 00:00")])
    alon = np.array([0, 10])
    alat = np.array([45, 55])
    try:
        sza_data_space(dtime, alon, alat)
        assert False, "Expected ValueError due to mismatched input lengths"
    except ValueError:
        assert True


def test_sza_data_space_known_results():
    """Test sza_data_space with known valid inputs."""
    dtime = np.reshape(np.array([datetime.datetime(2023, 11, 7, 21, 0, 0),
                                 datetime.datetime(2023, 11, 7, 21, 15, 0)]),
                       (1, 2))
    alon = np.reshape(np.array([166.65, -83.56]), (1, 2))
    alat = np.reshape(np.array([19.29, 45.07]), (1, 2))
    result = sza_data_space(dtime, alon, alat)

    expected = np.array([[64.17772638, 81.58068788]])
    assert np.allclose(result, expected, atol=1e-6), (
        f"Expected {expected}, got {result}")
