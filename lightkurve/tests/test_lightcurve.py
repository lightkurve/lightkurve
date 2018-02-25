from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)
from astropy.io import fits as pyfits
from ..lightcurve import (LightCurve, KeplerLightCurve, iterative_box_period_search)
from ..lightcurvefile import KeplerLightCurveFile, TessLightCurveFile

# 8th Quarter of Tabby's star
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")
TABBY_TPF = ("https://archive.stsci.edu/missions/kepler/target_pixel_files"
             "/0084/008462852/kplr008462852-2011073133259_lpd-targ.fits.gz")
K2_C08 = ("https://archive.stsci.edu/missions/k2/lightcurves/c8/"
          "220100000/39000/ktwo220139473-c08_llc.fits")
KEPLER10 = ("https://archive.stsci.edu/missions/kepler/lightcurves/"
            "0119/011904151/kplr011904151-2010009091648_llc.fits")
TESS_SIM = ("https://archive.stsci.edu/missions/tess/ete-6/tid/00/000/"
            "004/104/tess2019128220341-0000000410458113-0016-s_lc.fits")


def test_LightCurve():
    err_string = ("Input arrays have different lengths."
                  " len(time)=5, len(flux)=4, len(flux_err)=4")
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError) as err:
        lc = LightCurve(time=time, flux=flux)
    assert err_string == err.value.args[0]


def test_math_operators():
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5))
    lc_add = lc + 1
    lc_sub = lc - 1
    lc_mul = lc * 2
    lc_div = lc / 2
    assert_array_equal(lc_add.flux, lc.flux + 1)
    assert_array_equal(lc_sub.flux, lc.flux - 1)
    assert_array_equal(lc_mul.flux, lc.flux * 2)
    assert_array_equal(lc_div.flux, lc.flux / 2)


def test_rmath_operators():
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5))
    lc_add = 1 + lc
    lc_sub = 1 - lc
    lc_mul = 2 * lc
    lc_div = 2 / lc
    assert_array_equal(lc_add.flux, lc.flux + 1)
    assert_array_equal(lc_sub.flux, 1 - lc.flux)
    assert_array_equal(lc_mul.flux, lc.flux * 2)
    assert_array_equal(lc_div.flux, 2 / lc.flux)


@pytest.mark.remote_data
@pytest.mark.parametrize("path, mission", [(TABBY_Q8, "Kepler"), (K2_C08, "K2")])
def test_KeplerLightCurveFile(path, mission):
    lcf = KeplerLightCurveFile(path, quality_bitmask=None)
    hdu = pyfits.open(path)
    kplc = lcf.get_lightcurve('SAP_FLUX')

    assert kplc.channel == lcf.channel
    assert kplc.mission.lower() == mission.lower()
    if kplc.mission.lower() == 'kepler':
        assert kplc.campaign is None
        assert kplc.quarter == 8
    elif kplc.mission.lower() == 'k2':
        assert kplc.campaign == 8
        assert kplc.quarter is None

    assert_array_equal(kplc.time, hdu[1].data['TIME'])
    assert_array_equal(kplc.flux, hdu[1].data['SAP_FLUX'])


@pytest.mark.remote_data
@pytest.mark.parametrize("quality_bitmask",
                         ['hardest', 'hard', 'default', None,
                          1, 100, 2096639])
def test_TessLightCurveFile(quality_bitmask):
    tess_file = TessLightCurveFile(TESS_SIM, quality_bitmask=quality_bitmask)
    hdu = pyfits.open(TESS_SIM)
    tlc = tess_file.SAP_FLUX

    assert tlc.mission.lower() == 'tess'
    assert_array_equal(tlc.time, hdu[1].data['TIME'])
    assert_array_equal(tlc.flux, hdu[1].data['SAP_FLUX'])


@pytest.mark.remote_data
@pytest.mark.parametrize("quality_bitmask, answer", [('hardest', 2661),
    ('hard', 2706), ('default', 2917), (None, 3279),
    (1, 3279), (100, 3252), (2096639, 2661)])
def test_bitmasking(quality_bitmask, answer):
    """Test whether the bitmasking behaves like it should"""
    lcf = KeplerLightCurveFile(TABBY_Q8, quality_bitmask=quality_bitmask)
    flux = lcf.get_lightcurve('SAP_FLUX').flux
    assert len(flux) == answer


def test_lightcurve_fold():
    """Test the ``LightCurve.fold()`` method."""
    lc = LightCurve(time=np.linspace(0, 10, 100), flux=np.zeros(100)+1)
    fold = lc.fold(period=1)
    assert_almost_equal(fold.time[0], -0.5, 2)
    assert_almost_equal(np.min(fold.time), -0.5, 2)
    assert_almost_equal(np.max(fold.time), 0.5, 2)
    fold = lc.fold(period=1, phase=-0.1)
    assert_almost_equal(fold.time[0], -0.5, 2)
    assert_almost_equal(np.min(fold.time), -0.5, 2)
    assert_almost_equal(np.max(fold.time), 0.5, 2)


def test_lightcurve_stitch():
    """Test ``LightCurve.stitch()``."""
    lc = LightCurve(time=[1, 2, 3], flux=[1, .5, 1])
    lc = lc.stitch(lc)
    assert_array_equal(lc.flux, 2*[1, .5, 1])
    assert_array_equal(lc.time, 2*[1, 2, 3])


def test_lightcurve_stitch_multiple():
    """Test ``LightCurve.stitch()`` for multiple lightcurves at once."""
    lc = LightCurve(time=[1, 2, 3], flux=[1, .5, 1])
    lc = lc.stitch([lc, lc, lc])
    assert_array_equal(lc.flux, 4*[1, .5, 1])
    assert_array_equal(lc.time, 4*[1, 2, 3])


@pytest.mark.remote_data
def test_lightcurve_plot():
    """Sanity check to verify that lightcurve plotting works"""
    lcf = KeplerLightCurveFile(TABBY_Q8)
    lcf.plot()
    lcf.SAP_FLUX.plot()


def test_cdpp():
    """Test the basics of the CDPP noise metric."""
    # A flat lightcurve should have a CDPP close to zero
    assert_almost_equal(LightCurve(np.arange(200), np.ones(200)).cdpp(), 0)
    # An artificial lightcurve with sigma=100ppm should have cdpp=100ppm
    lc = LightCurve(np.arange(10000), np.random.normal(loc=1, scale=100e-6, size=10000))
    assert_almost_equal(lc.cdpp(transit_duration=1), 100, decimal=-0.5)


@pytest.mark.remote_data
def test_cdpp_tabby():
    """Compare the cdpp noise metric against the pipeline value."""
    lcf = KeplerLightCurveFile(TABBY_Q8)
    # Tabby's star shows dips after cadence 1000 which increase the cdpp
    lc = LightCurve(lcf.PDCSAP_FLUX.time[:1000], lcf.PDCSAP_FLUX.flux[:1000])
    assert(np.abs(lc.cdpp() - lcf.header(ext=1)['CDPP6_0']) < 30)


def test_bin():
    lc = LightCurve(time=np.arange(10), flux=2*np.ones(10),
                    flux_err=2**.5*np.ones(10))
    binned_lc = lc.bin(binsize=2)
    assert_allclose(binned_lc.flux, 2*np.ones(5))
    assert_allclose(binned_lc.flux_err, np.ones(5))
    assert len(binned_lc.time) == 5


def test_normalize():
    """Does the `LightCurve.normalize()` method normalize the flux?"""
    lc = LightCurve(time=np.arange(10), flux=5*np.ones(10))
    assert_allclose(np.median(lc.normalize().flux), 1)


@pytest.mark.remote_data
def test_iterative_box_period_search():
    """Can we recover the orbital period of Kepler-10b?"""
    answer = 0.837495  # wikipedia
    klc = KeplerLightCurveFile(KEPLER10)
    pdc = klc.PDCSAP_FLUX
    flat, trend = pdc.flatten(return_trend=True)

    _, _, kepler10b_period = iterative_box_period_search(flat, min_period=.5, max_period=1,
                                                         nperiods=101, period_scale='log')
    assert abs(kepler10b_period - answer) < 1e-2


def test_to_pandas():
    """Test the `LightCurve.to_pandas()` method."""
    time, flux, flux_err = range(3), np.ones(3), np.zeros(3)
    lc = LightCurve(time, flux, flux_err)
    try:
        df = lc.to_pandas()
        assert_allclose(df.index, time)
        assert_allclose(df.flux, flux)
        assert_allclose(df.flux_err, flux_err)
    except ImportError:
        # pandas is an optional dependency
        pass


def test_to_table():
    """Test the `LightCurve.to_table()` method."""
    time, flux, flux_err = range(3), np.ones(3), np.zeros(3)
    lc = LightCurve(time, flux, flux_err)
    tbl = lc.to_table()
    assert_allclose(tbl['time'], time)
    assert_allclose(tbl['flux'], flux)
    assert_allclose(tbl['flux_err'], flux_err)


def test_to_csv():
    """Test the `LightCurve.to_csv()` method."""
    time, flux, flux_err = range(3), np.ones(3), np.zeros(3)
    try:
        lc = LightCurve(time, flux, flux_err)
        assert(lc.to_csv() == 'time,flux,flux_err\n0,1.0,0.0\n1,1.0,0.0\n2,1.0,0.0\n')
    except ImportError:
        # pandas is an optional dependency
        pass


def test_date():
    '''Test the lc.date() function'''
    lcf = KeplerLightCurveFile(TABBY_Q8)
    date = lcf.timeobj.iso
    assert len(date) == len(lcf.time)
    assert date[0] == '2011-01-06 20:45:08.811'
    assert date[-1] == '2011-03-14 20:18:16.734'
