import pytest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from lightkurve.utils import KeplerQualityFlags, TessQualityFlags
from lightkurve.utils import module_output_to_channel, channel_to_module_output
from lightkurve.utils import LightkurveWarning
from lightkurve.utils import running_mean, validate_method
from lightkurve.utils import bkjd_to_astropy_time, btjd_to_astropy_time
from lightkurve.utils import centroid_quadratic
from lightkurve.utils import show_citation_instructions
from lightkurve.lightcurve import LightCurve


def test_channel_to_module_output():
    assert channel_to_module_output(1) == (2, 1)
    assert channel_to_module_output(42) == (13, 2)
    assert channel_to_module_output(84) == (24, 4)
    assert channel_to_module_output(33) == (11, 1)
    with pytest.raises(ValueError):
        channel_to_module_output(0)  # Invalid channel


def test_module_output_to_channel():
    assert module_output_to_channel(2, 1) == 1
    assert module_output_to_channel(13, 2) == 42
    assert module_output_to_channel(24, 4) == 84
    assert module_output_to_channel(11, 1) == 33
    with pytest.raises(ValueError):
        module_output_to_channel(0, 1)  # Invalid module
    with pytest.raises(ValueError):
        module_output_to_channel(2, 0)  # Invalid output


def test_running_mean():
    assert_almost_equal(running_mean([1, 2, 3], window_size=1), [1, 2, 3])
    assert_almost_equal(running_mean([1, 2, 3], window_size=2), [1.5, 2.5])
    assert_almost_equal(running_mean([2, 2, 2], window_size=3), [2])
    assert_almost_equal(running_mean([3, 4, 5], window_size=20), [4])


def test_quality_flag_decoding_kepler():
    """Can the QUALITY flags be parsed correctly?"""
    flags = list(KeplerQualityFlags.STRINGS.items())
    for key, value in flags:
        assert KeplerQualityFlags.decode(key)[0] == value
    # Can we recover combinations of flags?
    assert KeplerQualityFlags.decode(flags[5][0] + flags[7][0]) == [
        flags[5][1],
        flags[7][1],
    ]
    assert KeplerQualityFlags.decode(flags[3][0] + flags[4][0] + flags[5][0]) == [
        flags[3][1],
        flags[4][1],
        flags[5][1],
    ]


def test_quality_flag_decoding_tess():
    """Can the QUALITY flags be parsed correctly?"""
    flags = list(TessQualityFlags.STRINGS.items())
    for key, value in flags:
        assert TessQualityFlags.decode(key)[0] == value
    # Can we recover combinations of flags?
    assert TessQualityFlags.decode(flags[5][0] + flags[7][0]) == [
        flags[5][1],
        flags[7][1],
    ]
    assert TessQualityFlags.decode(flags[3][0] + flags[4][0] + flags[5][0]) == [
        flags[3][1],
        flags[4][1],
        flags[5][1],
    ]


def test_quality_flag_decoding_quantity_object():
    """Can a QUALITY flag that is a astropy quantity object be parsed correctly?

    This is a regression test for https://github.com/lightkurve/lightkurve/issues/804
    """
    from astropy.units.quantity import Quantity

    flags = list(TessQualityFlags.STRINGS.items())
    for key, value in flags:
        assert TessQualityFlags.decode(Quantity(key, dtype="int32"))[0] == value
    # Can we recover combinations of flags?
    assert TessQualityFlags.decode(
        Quantity(flags[5][0], dtype="int32") + Quantity(flags[7][0], dtype="int32")
    ) == [flags[5][1], flags[7][1]]
    assert TessQualityFlags.decode(
        Quantity(flags[3][0], dtype="int32")
        + Quantity(flags[4][0], dtype="int32")
        + Quantity(flags[5][0], dtype="int32")
    ) == [flags[3][1], flags[4][1], flags[5][1]]


def test_quality_mask():
    """Can we create a quality mask using KeplerQualityFlags?"""
    quality = np.array([0, 0, 1])
    assert np.all(KeplerQualityFlags.create_quality_mask(quality, bitmask=0))
    assert np.all(KeplerQualityFlags.create_quality_mask(quality, bitmask=None))
    assert np.all(KeplerQualityFlags.create_quality_mask(quality, bitmask="none"))
    assert (KeplerQualityFlags.create_quality_mask(quality, bitmask=1)).sum() == 2
    assert (
        KeplerQualityFlags.create_quality_mask(quality, bitmask="hardest")
    ).sum() == 2
    # Do we see a ValueError if an invalid bitmask is passed?
    with pytest.raises(ValueError) as err:
        KeplerQualityFlags.create_quality_mask(quality, bitmask="invalidoption")
    assert "not supported" in err.value.args[0]


@pytest.mark.xfail  # Lightkurve v2.x no longer support NaNs in time values
def test_lightkurve_warning():
    """Can we ignore Lightkurve warnings?"""
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("ignore", LightkurveWarning)
        time = np.array([1, 2, 3, np.nan])
        flux = np.array([1, 2, 3, 4])
        lc = LightCurve(time=time, flux=flux)
        assert len(warns) == 0


def test_validate_method():
    assert validate_method("foo", ["foo", "bar"]) == "foo"
    assert validate_method("FOO", ["foo", "bar"]) == "foo"
    with pytest.raises(ValueError):
        validate_method("foo", ["bar"])


def test_import():
    """Regression test for #605; `lk.utils` resolved to `lk.seismology.utils`"""
    from lightkurve import utils

    assert hasattr(utils, "btjd_to_astropy_time")


def test_btjd_bkjd_input():
    """Regression test for #607: are the bkjd/btjd functions tolerant?"""
    # Kepler
    assert bkjd_to_astropy_time(0).jd[0] == 2454833.0
    for user_input in [[0], np.array([0])]:
        assert_array_equal(bkjd_to_astropy_time(user_input).jd, np.array([2454833.0]))
    # TESS
    assert btjd_to_astropy_time(0).jd[0] == 2457000.0
    for user_input in [[0], np.array([0])]:
        assert_array_equal(btjd_to_astropy_time(user_input).jd, np.array([2457000.0]))


def test_centroid_quadratic():
    """Test basic operation of the quadratic centroiding function."""
    # Single bright pixel in the center
    data = np.ones((9, 9))
    data[2, 5] = 10
    col, row = centroid_quadratic(data)
    assert np.isclose(row, 2) & np.isclose(col, 5)

    # Two equally-bright pixels side by side
    data = np.zeros((9, 9))
    data[5, 1] = 5
    data[5, 2] = 5
    col, row = centroid_quadratic(data)
    assert np.isclose(row, 5) & np.isclose(col, 1.5)


# create a mask that'd mask out first 2 rows
a_mask = np.full((5, 5), True, dtype=bool)
a_mask[0:2, :] = False

@pytest.mark.parametrize("data_dtype, mask", [
    (float, None),
    (float, a_mask),
    (int, None),
    (int, a_mask),
    ])
def test_centroid_quadratic_robustness(data_dtype, mask):
    """Test quadratic centroids in edge cases; regression test for #610, etc."""
    # Brightest pixel in upper left
    data = np.zeros((5, 5), dtype=data_dtype)
    data[0, 0] = 1
    centroid_quadratic(data, mask=mask)
    col, row = centroid_quadratic(data, mask=mask)
    if mask is None:
        assert np.isfinite(col) & np.isfinite(row)
    else:
        # the mask (of top 2 rows) would make the eligible pixels to be uniformly at 0
        # no centroid can be calculated in such case.
        assert np.isnan(col) & np.isnan(row)

    # Brightest pixel in bottom right
    data = np.zeros((5, 5), dtype=data_dtype)
    data[-1, -1] = 1
    col, row = centroid_quadratic(data, mask=mask)
    assert np.isfinite(col) & np.isfinite(row)

    # Data contains a NaN
    if data_dtype is float:  # NaN only applicable to float
        data = np.zeros((5, 5), dtype=data_dtype)
        data[0, 0] = np.nan
        data[-1, -1] = 10
        col, row = centroid_quadratic(data, mask=mask)
        assert np.isfinite(col) & np.isfinite(row)

    # has NaN in the identified 3x3 patch
    if data_dtype is float:  # NaN only applicable to float
        data = np.zeros((5, 5), dtype=data_dtype)
        data[3, 2] = 10
        data[3, 3] = np.nan
        col, row = centroid_quadratic(data, mask=mask)
        assert np.isfinite(col) & np.isfinite(row)

    # Data contains are all negative (#1401 case `mask` is specified)
    # e.g., in a difference image
    data = np.full((5, 5), -9, dtype=data_dtype)
    data[3, 2] = -5
    col, row = centroid_quadratic(data, mask=mask)
    assert np.isfinite(col) & np.isfinite(row)

    # case the 3x3 patch contains masked pixels
    if mask is not None:
        data = np.zeros((5, 5), dtype=data_dtype)
        data[2, 1] = 10  # the row above is masked
        col, row = centroid_quadratic(data, mask=mask)
        assert np.isfinite(col) & np.isfinite(row)


def test_show_citation_instructions():
    show_citation_instructions()
