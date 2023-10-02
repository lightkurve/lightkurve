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

from lightkurve.utils import query_skycatalog

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


def test_centroid_quadratic_robustness():
    """Test quadratic centroids in edge cases; regression test for #610."""
    # Brightest pixel in upper left
    data = np.zeros((5, 5))
    data[0, 0] = 1
    centroid_quadratic(data)

    # Brightest pixel in bottom right
    data = np.zeros((5, 5))
    data[-1, -1] = 1
    centroid_quadratic(data)

    # Data contains a NaN
    data = np.zeros((5, 5))
    data[0, 0] = np.nan
    data[-1, -1] = 10
    col, row = centroid_quadratic(data)
    assert np.isfinite(col) & np.isfinite(row)


def test_show_citation_instructions():
    show_citation_instructions()

@pytest.mark.remote_data
def test_query_skycatalog():
    # Tests the region around TIC 228760807 which should return a catalog containing 4 objects. 
    c = SkyCoord(194.10141041659, -27.3905828803397, unit="deg")
    epoch  = Time(1569.4424277786259, scale='tdb', format='btjd')
    
    catalog = query_skycatalog(c, epoch, 'tic', 80, 18)
    assert len(catalog['ID']) == 4
    
    # Checks that an astropy Table is returned
    assert isinstance(catalog, Table)

     # Test that the proper motion works
    correct_ra = 194.10768406619445
    correct_dec = -27.41051178249595
 
    assert np.isclose(catalog['RA_CORRECTED'][0], correct_ra, atol=1e-6)
    assert np.isclose(catalog['DEC_CORRECTED'][0], correct_dec, atol=1e-6)

    # Test different epochs
    catalog_new = query_skycatalog(c, Time(2461041.500, scale='tt', format='jd'),'tic',80,18)

    correct_ra_new = 194.10764826027054
    correct_dec_new = -27.41050446197694

    assert np.isclose(catalog['RA_CORRECTED'][0], correct_ra_new, atol=1e-6)
    assert np.isclose(catalog['DEC_CORRECTED'][0], correct_dec_new, atol=1e-6)

    # test the catalog type i.e., simbad is not included in our catalog list.
    # Look at other tests to see if this is correct syntax
    with pytest.raises(ValueError, match="Can not parse catalog name 'simbad'"):
        query_skycatalog(c, epoch, 'simbad', 80, 18)
        
    # Test each other catalog
    # Gaia
    catalog_gaia = query_skycatalog(c, Time(1569.4424277786259, scale='tdb', format='btjd'),'gaiadr3',80,18)

    assert len(catalog_gaia['ID']) == 2

    #Kepler
    catalog_kepler = query_skycatalog(SkyCoord(285.679391, 50.2413, unit="deg"),
                                         Time(120.5391465105713, scale="tdb", format="bkjd"),
                                         'kic',20,18)

    assert len(catalog_kepler['ID']) == 5

    #K2
    catalog_k2 = query_skycatalog(SkyCoord(172.560465, 7.588391, unit="deg"),
                                         Time(1975.1781333280233, scale="tdb", format="bkjd"),
                                         'epic',20,18)

    assert len(catalog_k2['ID']) == 1
    
    

@pytest.mark.remote_data
def test_query_skycatalog_tpf():

    # Test that you get an answer for each TPF type
    TESS_tpf = lk.search_targetpixelfile("TIC 228760807")[0].download()
    catalog_tess = TESS_tpf.skycatalog
    assert isinstance(catalog_tess, Table)
    
    Kepler_tpf = lk.search_targetpixelfile("Kepler-10")[0].download()
    catalog_kepler = Kepler_tpf.skycatalog
    assert isinstance(catalog_kepler, Table)

    K2_tpf = lk.search_targetpixelfile("K2-18")[0].download()
    catalog_k2 = K2_tpf.skycatalog
    assert isinstance(catalog_k2, Table)

    # assert that the target is inside the TPF
    target_ra = TESS_tpf.ra
    target_dec = TESS_tpf.dec

    index = np.where(catalog_tess['ID']==228760807)
    cat_ra = catalog_tess['RA_CORRECTED'][index]
    cat_dec = catalog_tess['DEC_CORRECTED'][index]

    c1 = SkyCoord(target_ra, target_dec)
    c2 = SkyCoord(cat_ra, cat_dec)
    sep = c1.separation(c2)
    assert sep.arcsecond < 0.1 
    
    # Check that it has pixel positions and that they are within reason
    correct_row = 1806.0942842433474
    correct_col = 825.0328770805637
 
    assert np.isclose(catalog_tess['Row'][0], correct_row, atol=1e-6)
    assert np.isclose(catalog_tess['Column'][0], correct_col, atol=1e-6)
    
    # Check that there are magnitudes
    correct_mag = 9.459
    assert np.isclose(catalog_tess['Mag'][0], correct_mag, atol=1e-3)
    

    

