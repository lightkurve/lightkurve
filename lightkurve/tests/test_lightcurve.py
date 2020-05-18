from __future__ import division, print_function

from astropy.io import fits as pyfits
from astropy.utils.data import get_pkg_data_filename
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)
import pytest
import tempfile
import warnings

from ..lightcurve import LightCurve, KeplerLightCurve, TessLightCurve
from ..lightcurvefile import LightCurveFile, KeplerLightCurveFile, TessLightCurveFile
from ..targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from ..utils import LightkurveWarning
from .test_targetpixelfile import TABBY_TPF


# 8th Quarter of Tabby's star
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")
K2_C08 = ("https://archive.stsci.edu/missions/k2/lightcurves/c8/"
          "220100000/39000/ktwo220139473-c08_llc.fits")
KEPLER10 = ("https://archive.stsci.edu/missions/kepler/lightcurves/"
            "0119/011904151/kplr011904151-2010009091648_llc.fits")
TESS_SIM = ("https://archive.stsci.edu/missions/tess/ete-6/tid/00/000/"
            "004/104/tess2019128220341-0000000410458113-0016-s_lc.fits")
filename_tess = get_pkg_data_filename("data/tess25155310-s01-first-cadences.fits.gz")
filename_tess_custom = get_pkg_data_filename("data/test_TESS_interact_generated_custom-lc.fits")
filename_K2_custom = get_pkg_data_filename("data/test_K2_interact_generated_custom-lc.fits")


# `asteroid_test.fits` is a single cadence of TESS FFI data which contains a known solar system object
asteroid_TPF = get_pkg_data_filename("data/asteroid_test.fits")

def test_invalid_lightcurve():
    """Invalid LightCurves should not be allowed."""
    err_string = ("Input arrays have different lengths."
                  " len(time)=5, len(flux)=4")
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError) as err:
        LightCurve(time=time, flux=flux)
    assert err_string == err.value.args[0]


def test_empty_lightcurve():
    """LightCurves with no data should not be allowed."""
    err_string = ("either time or flux must be given")
    with pytest.raises(ValueError) as err:
        LightCurve()
    assert err_string == err.value.args[0]


def test_lc_nan_time():
    time = np.array([1, 2, 3, np.nan])
    flux = np.array([1, 2, 3, 4])
    with pytest.warns(LightkurveWarning, match='contains NaN times'):
        LightCurve(time=time, flux=flux)


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


def test_math_operators_on_objects():
    lc1 = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5))
    lc2 = LightCurve(time=np.arange(1, 5), flux=np.arange(11, 15), flux_err=np.arange(1, 5))
    assert_array_equal((lc1 + lc2).flux, lc1.flux + lc2.flux)
    assert_array_equal((lc1 - lc2).flux, lc1.flux - lc2.flux)
    assert_array_equal((lc1 * lc2).flux, lc1.flux * lc2.flux)
    assert_array_equal((lc1 / lc2).flux, lc1.flux / lc2.flux)
    # Change order
    assert_array_equal((lc2 + lc1).flux, lc2.flux + lc1.flux)
    assert_array_equal((lc2 - lc1).flux, lc2.flux - lc1.flux)
    assert_array_equal((lc2 * lc1).flux, lc2.flux * lc1.flux)
    assert_array_equal((lc2 / lc1).flux, lc2.flux / lc1.flux)
    # LightCurve objects can only be added or multiplied if they have equal length
    with pytest.raises(ValueError):
        lc = lc1 + lc1[0:-5]
    with pytest.raises(ValueError):
        lc = lc1 * lc1[0:-5]


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
    assert lcf.obsmode == 'long cadence'
    assert len(lcf.pos_corr1) == len(lcf.pos_corr2)

    # The liberal bitmask will cause the lightcurve to contain NaN times
    with pytest.warns(LightkurveWarning, match='NaN times'):
        lc = lcf.get_lightcurve('SAP_FLUX')

    assert lc.channel == lcf.channel
    assert lc.mission.lower() == mission.lower()
    if lc.mission.lower() == 'kepler':
        assert lc.campaign is None
        assert lc.quarter == 8
    elif lc.mission.lower() == 'k2':
        assert lc.campaign == 8
        assert lc.quarter is None
    assert lc.time_format == 'bkjd'
    assert lc.time_scale == 'tdb'
    assert lc.astropy_time.scale == 'tdb'
    assert lc.flux_unit == u.electron / u.second

    # Does the data match what one would obtain using pyfits.open?
    hdu = pyfits.open(path)
    assert lc.label == hdu[0].header['OBJECT']
    assert_array_equal(lc.time, hdu[1].data['TIME'])
    assert_array_equal(lc.flux, hdu[1].data['SAP_FLUX'] / ((hdu[1].header['CROWDSAP'] * hdu[1].header['FLFRCSAP'])))

    with pytest.raises(KeyError):
        lcf.get_lightcurve('BLABLA')


@pytest.mark.remote_data
@pytest.mark.parametrize("quality_bitmask",
                         ['hardest', 'hard', 'default', None,
                          1, 100, 2096639])
def test_TessLightCurveFile(quality_bitmask):
    tess_file = TessLightCurveFile(TESS_SIM, quality_bitmask=quality_bitmask)
    hdu = pyfits.open(TESS_SIM)
    lc = tess_file.SAP_FLUX

    assert lc.mission == 'TESS'
    assert lc.label == hdu[0].header['OBJECT']
    assert lc.time_format == 'btjd'
    assert lc.time_scale == 'tdb'
    assert lc.flux_unit == u.electron / u.second
    assert lc.sector == hdu[0].header['SECTOR']
    assert lc.camera == hdu[0].header['CAMERA']
    assert lc.ccd == hdu[0].header['CCD']
    assert lc.ra == hdu[0].header['RA_OBJ']
    assert lc.dec == hdu[0].header['DEC_OBJ']

    assert_array_equal(lc.time[0:10], hdu[1].data['TIME'][0:10])
    assert_array_equal(lc.flux[0:10], hdu[1].data['SAP_FLUX'][0:10])

    # Regression test for https://github.com/KeplerGO/lightkurve/pull/236
    assert np.isnan(lc.time).sum() == 0

    with pytest.raises(KeyError):
        tess_file.get_lightcurve('DOESNOTEXIST')


@pytest.mark.remote_data
@pytest.mark.parametrize("quality_bitmask, answer", [('hardest', 2661),
                                                     ('hard', 2706), ('default', 3113), (None, 3279),
                                                     (1, 3279), (100, 3252), (2096639, 2661)])
def test_bitmasking(quality_bitmask, answer):
    """Test whether the bitmasking behaves like it should"""
    lcf = KeplerLightCurveFile(TABBY_Q8, quality_bitmask=quality_bitmask)
    with warnings.catch_warnings():
        # Ignore "LightCurve contains NaN times" warnings triggered by liberal masks
        warnings.simplefilter("ignore", LightkurveWarning)
        flux = lcf.get_lightcurve('SAP_FLUX').flux
    assert len(flux) == answer


def test_lightcurve_fold():
    """Test the ``LightCurve.fold()`` method."""
    lc = LightCurve(time=np.linspace(0, 10, 100), flux=np.zeros(100)+1,
                    targetid=999, label='mystar', meta={'ccd': 2}, time_format='bkjd')
    fold = lc.fold(period=1)
    assert_almost_equal(fold.phase[0], -0.5, 2)
    assert_almost_equal(np.min(fold.phase), -0.5, 2)
    assert_almost_equal(np.max(fold.phase), 0.5, 2)
    assert fold.targetid == lc.targetid
    assert fold.label == lc.label
    assert fold.meta == lc.meta
    assert_array_equal(np.sort(fold.time_original), lc.time)
    assert len(fold.time_original) == len(lc.time)
    fold = lc.fold(period=1, t0=-0.1)
    assert_almost_equal(fold.time[0], -0.5, 2)
    assert_almost_equal(np.min(fold.phase), -0.5, 2)
    assert_almost_equal(np.max(fold.phase), 0.5, 2)
    with warnings.catch_warnings():
        # `transit_midpoint` is deprecated and its use will emit a warning
        warnings.simplefilter("ignore", LightkurveWarning)
        fold = lc.fold(period=1, transit_midpoint=-0.1)
    assert_almost_equal(fold.time[0], -0.5, 2)
    ax = fold.plot()
    assert (ax.get_xlabel() == 'Phase')
    ax = fold.scatter()
    assert (ax.get_xlabel() == 'Phase')
    ax = fold.errorbar()
    assert (ax.get_xlabel() == 'Phase')
    plt.close('all')


    odd = fold.odd_mask
    even = fold.even_mask
    assert len(odd) == len(fold.time)
    assert np.all(odd == ~even)
    assert np.sum(odd) == np.sum(even)
    # bad transit midpoint should give a warning
    # if user tries a t0 in JD but time is in BKJD
    with pytest.warns(LightkurveWarning, match='appears to be given in JD'):
        lc.fold(10, 2456600)

def test_lightcurve_fold_issue520():
    """Regression test for #520; accept quantities in `fold()`."""
    lc = LightCurve(time=np.linspace(0, 10, 100), flux=np.zeros(100)+1)
    lc.fold(period=1*u.day, t0=5*u.day)

def test_lightcurve_append():
    """Test ``LightCurve.append()``."""
    lc = LightCurve(time=[1, 2, 3], flux=[1, .5, 1], flux_err=[0.1, 0.2, 0.3])
    lc = lc.append(lc)
    assert_array_equal(lc.time, 2*[1, 2, 3])
    assert_array_equal(lc.flux, 2*[1, .5, 1])
    assert_array_equal(lc.flux_err, 2*[0.1, 0.2, 0.3])
    # KeplerLightCurve has extra data
    lc = KeplerLightCurve(time=[1, 2, 3], flux=[1, .5, 1],
                          centroid_col=[4, 5, 6], centroid_row=[7, 8, 9],
                          cadenceno=[10, 11, 12], quality=[10, 20, 30])
    lc = lc.append(lc)
    assert_array_equal(lc.time, 2*[1, 2, 3])
    assert_array_equal(lc.flux, 2*[1, .5, 1])
    assert_array_equal(lc.centroid_col, 2*[4, 5, 6])
    assert_array_equal(lc.centroid_row, 2*[7, 8, 9])
    assert_array_equal(lc.cadenceno, 2*[10, 11, 12])
    assert_array_equal(lc.quality, 2*[10, 20, 30])


def test_lightcurve_append_multiple():
    """Test ``LightCurve.append()`` for multiple lightcurves at once."""
    lc = LightCurve(time=[1, 2, 3], flux=[1, .5, 1])
    lc = lc.append([lc, lc, lc])
    assert_array_equal(lc.flux, 4*[1, .5, 1])
    assert_array_equal(lc.time, 4*[1, 2, 3])


def test_lightcurve_append_inconsistent_columns():
    """Test ``LightCurve.append()`` for different sub-classes.

    Compared to the base `LightCurve`, `KeplerLightCurve` has extra columns
    such as `centroid_col` and `cadenceno`.  This test checks whether
    appending such two objects raises a warning.
    """
    lc1 = KeplerLightCurve(time=[1, 2, 3], flux=[1, .5, 1])
    lc2 = LightCurve(time=[1, 2, 3], flux=[1, .5, 1])
    with pytest.warns(LightkurveWarning, match='extra_columns'):
        lc = lc1.append(lc2)


def test_lightcurve_copy():
    """Test ``LightCurve.copy()``."""
    time = np.array([1, 2, 3, 4])
    flux = np.array([1, 2, 3, 4])
    error = np.array([0.1, 0.2, 0.3, 0.4])
    lc = LightCurve(time=time, flux=flux, flux_err=error)

    nlc = lc.copy()
    assert_array_equal(lc.time, nlc.time)
    assert_array_equal(lc.flux, nlc.flux)
    assert_array_equal(lc.flux_err, nlc.flux_err)

    nlc.time[1] = 5
    nlc.flux[1] = 6
    nlc.flux_err[1] = 7

    # By changing 1 of the 4 data points in the new lightcurve's array-like
    # attributes, we expect assert_array_equal to raise an AssertionError
    # indicating a mismatch of 1/4 (or 25%).
    with pytest.raises(AssertionError, match=r'ismatch.*25'):
        assert_array_equal(lc.time, nlc.time)
    with pytest.raises(AssertionError, match=r'ismatch.*25'):
        assert_array_equal(lc.flux, nlc.flux)
    with pytest.raises(AssertionError, match=r'ismatch.*25'):
        assert_array_equal(lc.flux_err, nlc.flux_err)

    # KeplerLightCurve has extra data
    lc = KeplerLightCurve(time=[1, 2, 3], flux=[1, .5, 1],
                          centroid_col=[4, 5, 6], centroid_row=[7, 8, 9],
                          cadenceno=[10, 11, 12], quality=[10, 20, 30])
    nlc = lc.copy()
    assert_array_equal(lc.time, nlc.time)
    assert_array_equal(lc.flux, nlc.flux)
    assert_array_equal(lc.centroid_col, nlc.centroid_col)
    assert_array_equal(lc.centroid_row, nlc.centroid_row)
    assert_array_equal(lc.cadenceno, nlc.cadenceno)
    assert_array_equal(lc.quality, nlc.quality)

    nlc.time[1] = 6
    nlc.flux[1] = 7
    nlc.centroid_col[1] = 8
    nlc.centroid_row[1] = 9
    nlc.cadenceno[1] = 10
    nlc.quality[1] = 11

    # As before, by changing 1/3 data points, we expect a mismatch of 33.3%
    # with a repeating decimal. However, float precision for python 2.7 is 10
    # decimal digits, while python 3.6's is 13 decimal digits. Therefore,
    # a regular expression is needed for both versions.
    with pytest.raises(AssertionError, match=r'ismatch.*33\.3+'):
        assert_array_equal(lc.time, nlc.time)
    with pytest.raises(AssertionError, match=r'ismatch.*33\.3+'):
        assert_array_equal(lc.flux, nlc.flux)
    with pytest.raises(AssertionError, match=r'ismatch.*33\.3+'):
        assert_array_equal(lc.centroid_col, nlc.centroid_col)
    with pytest.raises(AssertionError, match=r'ismatch.*33\.3+'):
        assert_array_equal(lc.centroid_row, nlc.centroid_row)
    with pytest.raises(AssertionError, match=r'ismatch.*33\.3+'):
        assert_array_equal(lc.cadenceno, nlc.cadenceno)
    with pytest.raises(AssertionError, match=r'ismatch.*33\.3+'):
        assert_array_equal(lc.quality, nlc.quality)

@pytest.mark.parametrize("path, mission", [(filename_tess_custom, "TESS"),
                                           (filename_K2_custom, "K2")])
def test_custom_lightcurve_file(path, mission):
    """Test whether we can read in custom interact()-produced lightcurvefiles"""
    if mission == "K2":
        lcf_custom = KeplerLightCurveFile(path)
    elif mission == "TESS":
        with pytest.warns(LightkurveWarning):
            lcf_custom = TessLightCurveFile(path)
    assert lcf_custom.hdu[2].name == 'APERTURE'
    assert lcf_custom.cadenceno[0] >= 0
    assert lcf_custom.dec == lcf_custom.dec
    assert lcf_custom.time[-1] > lcf_custom.time[0]
    # .interact() files currently define FLUX, and not SAP_FLUX nor PDCSAP_FLUX
    lc = lcf_custom.get_lightcurve('FLUX')
    assert len(lc.flux) > 0
    with pytest.raises(KeyError):
        lcf_custom.get_lightcurve('BLABLA')
    with pytest.raises(KeyError):
        lcf_custom.SAP_FLUX
    with pytest.raises(KeyError):
        lcf_custom.PDCSAP_FLUX

    assert lc.mission.lower() == mission.lower()
    # Does the data match what one would obtain using pyfits.open?
    hdu = pyfits.open(path)
    assert lc.label == hdu[0].header['OBJECT']
    assert_array_equal(lc.time, hdu[1].data['TIME'])
    assert_array_equal(lc.flux, hdu[1].data['FLUX'])

    # TESS has QUALITY while Kepler/K2 has SAP_QUALITY:
    if mission == "TESS":
        assert "QUALITY" in lcf_custom.hdu[1].columns.names
        assert_array_equal(lc.quality, hdu[1].data['QUALITY'])
    if mission in ["K2", "Kepler"]:
        assert "SAP_QUALITY" in lcf_custom.hdu[1].columns.names
        assert_array_equal(lc.quality, hdu[1].data['SAP_QUALITY'])



@pytest.mark.remote_data
def test_lightcurve_plots():
    """Sanity check to verify that lightcurve plotting works"""
    for lcf in [KeplerLightCurveFile(TABBY_Q8), TessLightCurveFile(TESS_SIM)]:
        lcf.plot()
        lcf.plot(flux_types=['SAP_FLUX', 'PDCSAP_FLUX'])
        lcf.scatter()
        lcf.errorbar()
        lcf.SAP_FLUX.plot()
        lcf.SAP_FLUX.plot(normalize=False, title="Not the default")
        lcf.SAP_FLUX.scatter()
        lcf.SAP_FLUX.scatter(c='C3')
        lcf.SAP_FLUX.scatter(c=lcf.SAP_FLUX.time, show_colorbar=True, colorbar_label='Time')
        lcf.SAP_FLUX.errorbar()
        plt.close('all')


@pytest.mark.remote_data
def test_lightcurve_scatter():
    """Sanity check to verify that lightcurve scatter plotting works"""
    lcf = KeplerLightCurveFile(KEPLER10)
    lc = lcf.PDCSAP_FLUX.flatten()

    # get an array of original times, in the same order as the folded lightcurve
    foldkw = dict(period=0.837491)
    originaltime = LightCurve(lc.time, lc.time)
    foldedtimeinorder = originaltime.fold(**foldkw).flux

    # plot a grid of phase-folded and not, with colors
    fi, ax = plt.subplots(2, 2, figsize=(10,6), sharey=True, sharex='col')
    scatterkw = dict( s=5, cmap='winter')
    lc.scatter(ax=ax[0,0])
    lc.fold(**foldkw).scatter(ax=ax[0,1])
    lc.scatter(ax=ax[1,0], c=lc.time, **scatterkw)
    lc.fold(**foldkw).scatter(ax=ax[1,1], c=foldedtimeinorder, **scatterkw)
    plt.ylim(0.999, 1.001)


def test_cdpp():
    """Test the basics of the CDPP noise metric."""
    # A flat lightcurve should have a CDPP close to zero
    assert_almost_equal(LightCurve(np.arange(200), np.ones(200)).estimate_cdpp(), 0)
    # An artificial lightcurve with sigma=100ppm should have cdpp=100ppm
    lc = LightCurve(np.arange(10000), np.random.normal(loc=1, scale=100e-6, size=10000))
    assert_almost_equal(lc.estimate_cdpp(transit_duration=1), 100, decimal=-0.5)
    # Transit_duration must be an integer (cadences)
    with pytest.raises(ValueError):
        lc.estimate_cdpp(transit_duration=6.5)


@pytest.mark.remote_data
def test_cdpp_tabby():
    """Compare the cdpp noise metric against the pipeline value."""
    lcf = KeplerLightCurveFile(TABBY_Q8)
    # Tabby's star shows dips after cadence 1000 which increase the cdpp
    lc = LightCurve(lcf.PDCSAP_FLUX.time[:1000], lcf.PDCSAP_FLUX.flux[:1000])
    assert(np.abs(lc.estimate_cdpp() - lcf.get_header(ext=1)['CDPP6_0']) < 30)


# TEMPORARILY SKIP, cf. https://github.com/KeplerGO/lightkurve/issues/663
@pytest.mark.xfail
def test_bin():
    """Does binning work?"""
    lc = LightCurve(time=np.arange(10),
                    flux=2*np.ones(10),
                    flux_err=2**.5*np.ones(10))
    binned_lc = lc.bin(binsize=2)
    assert_allclose(binned_lc.flux, 2*np.ones(5))
    assert_allclose(binned_lc.flux_err, np.ones(5))
    assert len(binned_lc.time) == 5
    with pytest.raises(ValueError):
        lc.bin(method='doesnotexist')
    # If `flux_err` is missing, the errors on the bins should be the stddev
    lc = LightCurve(time=np.arange(10),
                    flux=2*np.ones(10))
    binned_lc = lc.bin(binsize=2)
    assert_allclose(binned_lc.flux_err, np.zeros(5))
    # Regression test for #377
    lc = KeplerLightCurve(time=np.arange(10),
                          flux=2*np.ones(10))
    lc.bin(5).remove_outliers()
    # Second regression test for #377
    lc = KeplerLightCurve(time=np.arange(1000) * 0.02,
                          flux=1*np.ones(1000) + np.random.normal(0, 1e-6, 1000),
                          cadenceno=np.arange(1000))
    assert np.isclose(lc.bin(2).estimate_cdpp(), 1, rtol=1)
    # Regression test for #500
    lc = LightCurve(time=np.arange(2000),
                    flux=np.random.normal(loc=42, scale=0.01, size=2000))
    assert np.round(lc.bin(2000).flux_err[0], 2) == 0.01


def test_bins_kwarg():
    """Does binning work with user-defined bin placement?"""
    # The bins feature requires astropy >3.1 or >2.10;
    # so we'll ignore this test if those versions are not available.
    # We should remove this check once we upgrade the minimum requirements.
    try:
        from astropy.stats import calculate_bin_edges
    except ImportError:
        return

    n_times = 3800
    time_points = np.sort(np.random.uniform(low=0.0, high=80.0, size=n_times))
    lc = LightCurve(time=time_points, flux=1.0+np.random.normal(0, 0.1, n_times),
                    flux_err=0.1*np.ones(n_times))
    # Do the shapes of binned lightcurves make sense?
    binned_lc = lc.bin(binsize=10)
    assert len(binned_lc) == n_times // 10
    binned_lc = lc.bin(binsize=11)
    # Resulting length with binsize may depend on implementation:
    #   Allowing for under-filled bins at boundary conditions
    assert ((len(binned_lc) >= (n_times // 11) ) &
            (len(binned_lc) <= (n_times // 11 +1) ) )
    #   Not allowing for under-filled bins at boundary conditions
    assert len(binned_lc) == (n_times // 11)
    # Resulting length with `bins=N` yields exactly N bins every time
    binned_lc = lc.bin(bins=38)
    assert len(binned_lc) == 38
    # Can't provide both a binsize= and a bins=; pick only one
    with pytest.raises(ValueError):
        binned_lc = lc.bin(binsize=10, bins=38)
    with pytest.raises(ValueError):
        binned_lc = lc.bin(10, 38, 'mean')
    # The `bins=`` kwarg can support a list or array
    time_bin_edges = [0,10,20,30,40,50,60,70,80]
    binned_lc = lc.bin(bins=time_bin_edges)
    # You get N-1 bins when you enter N fenceposts
    assert len(binned_lc) == (len(time_bin_edges) - 1 )
    time_bin_edges = np.arange(0,81,1)
    binned_lc = lc.bin(bins=time_bin_edges)
    assert len(binned_lc) == (len(time_bin_edges) - 1 )
    # Bins outside of the range get stuck in the last bin
    time_bin_edges = np.arange(0,61,1)
    binned_lc = lc.bin(bins=time_bin_edges)
    assert len(binned_lc) == (len(time_bin_edges) - 1 )
    # The `bins=`` kwarg can support a list or array
    for special_bins in ['blocks', 'knuth', 'scott', 'freedman']:
        binned_lc = lc.bin(bins=special_bins)
    with pytest.raises(ValueError):
        binned_lc = lc.bin(bins='junk_input!')
    # In dense bins, flux error should go down as root-N for N number of bins
    binned_lc = lc.bin(binsize=100) # Exactly 100 samples per bin
    assert np.isclose(lc.flux_err.mean()/np.sqrt(100),
                      binned_lc.flux_err.mean(), rtol=0.3)
    binned_lc = lc.bin(bins=38) # Roughly 100 samples per bin
    assert np.isclose(lc.flux_err.mean()/np.sqrt(100),
                  binned_lc.flux_err.mean(), rtol=0.3)
    # The bins parameter must be integer not a float
    with pytest.raises(TypeError):
        binned_lc = lc.bin(bins=381.0)
    # Binned lightcurve can have *more* bins than input lightcurve
    binned_lc = lc.bin(bins=10000)
    assert len(binned_lc) == 10000

    # To-do: Check for unusual edge cases that are now possible:
    #   - Binned lightcurve has NaN fluxes in empty bins
    #   - Binned lightcurve has a single bin (e.g. in Knuth)
    #   - Bins = 310.0


# TEMPORARILY SKIP, cf. https://github.com/KeplerGO/lightkurve/issues/663
@pytest.mark.xfail
def test_bin_quality():
    """Binning must also revise the quality and centroid columns."""
    lc = KeplerLightCurve(time=[1, 2, 3, 4],
                          flux=[1, 1, 1, 1],
                          quality=[0, 1, 2, 3],
                          centroid_col=[0, 1, 0, 1],
                          centroid_row=[0, 2, 0, 2])
    binned_lc = lc.bin(binsize=2)
    assert_allclose(binned_lc.quality, [1, 3])  # Expect bitwise or
    assert_allclose(binned_lc.centroid_col, [0.5, 0.5])  # Expect mean
    assert_allclose(binned_lc.centroid_row, [1, 1])  # Expect mean


def test_normalize():
    """Does the `LightCurve.normalize()` method normalize the flux?"""
    lc = LightCurve(time=np.arange(10), flux=5*np.ones(10), flux_err=0.05*np.ones(10))
    assert_allclose(np.median(lc.normalize().flux), 1)
    assert_allclose(np.median(lc.normalize().flux_err), 0.05/5)


def test_invalid_normalize():
    """Normalization makes no sense if the light curve is negative,
    zero-centered, or already in relative units."""
    # zero-centered light curve
    lc = LightCurve(time=np.arange(10), flux=np.zeros(10))
    with pytest.warns(LightkurveWarning, match='zero-centered'):
        lc.normalize()

    # zero-centered light curve with flux errors
    lc = LightCurve(time=np.arange(10), flux=np.zeros(10), flux_err=0.05*np.ones(10))
    with pytest.warns(LightkurveWarning, match='zero-centered'):
        lc.normalize()

    # negative light curve
    lc = LightCurve(time=np.arange(10), flux=-np.ones(10), flux_err=0.05*np.ones(10))
    with pytest.warns(LightkurveWarning, match='negative'):
        lc.normalize()

    # already in relative units
    lc = LightCurve(time=np.arange(10), flux=np.ones(10))
    with pytest.warns(LightkurveWarning, match='relative'):
        lc.normalize().normalize()

def test_to_pandas():
    """Test the `LightCurve.to_pandas()` method."""
    time, flux, flux_err = range(3), np.ones(3), np.zeros(3)
    lc = LightCurve(time, flux, flux_err)
    try:
        df = lc.to_pandas()
        assert_allclose(df.index, time)
        assert_allclose(df.flux, flux)
        assert_allclose(df.flux_err, flux_err)
        df.describe() # Will fail if for Endianness bugs
    except ImportError:
        # pandas is an optional dependency
        pass


def test_to_pandas_kepler():
    """When to_pandas() is executed on a KeplerLightCurve, it should include
    extra columns such as `quality`."""
    time, flux, quality = range(3), np.ones(3), np.zeros(3)
    lc = KeplerLightCurve(time, flux, quality=quality)
    try:
        df = lc.to_pandas()
        assert_allclose(df.quality, quality)
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
        assert(lc.to_csv(index=False, line_terminator='\n') == 'time,flux,flux_err\n0,1.0,0.0\n1,1.0,0.0\n2,1.0,0.0\n')
    except ImportError:
        # pandas is an optional dependency
        pass


@pytest.mark.remote_data
def test_to_fits():
    """Test the KeplerLightCurve.to_fits() method"""
    lcf = KeplerLightCurveFile(TABBY_Q8)
    hdu = lcf.PDCSAP_FLUX.to_fits()
    KeplerLightCurveFile(hdu)  # Regression test for #233
    assert type(hdu).__name__ is 'HDUList'
    assert len(hdu) == 2
    assert hdu[0].header['EXTNAME'] == 'PRIMARY'
    assert hdu[1].header['EXTNAME'] == 'LIGHTCURVE'
    assert hdu[1].header['TTYPE1'] == 'TIME'
    assert hdu[1].header['TTYPE2'] == 'FLUX'
    assert hdu[1].header['TTYPE3'] == 'FLUX_ERR'
    assert hdu[1].header['TTYPE4'] == 'CADENCENO'
    hdu = LightCurve([0, 1, 2, 3, 4], [1, 1, 1, 1, 1]).to_fits()

    # Test "round-tripping": can we read-in what we write
    lcf_new = LightCurveFile(hdu)  # Regression test for #233
    assert hdu[0].header['EXTNAME'] == 'PRIMARY'
    assert hdu[1].header['EXTNAME'] == 'LIGHTCURVE'
    assert hdu[1].header['TTYPE1'] == 'TIME'
    assert hdu[1].header['TTYPE2'] == 'FLUX'

    # Test aperture mask support in to_fits
    for tpf in [KeplerTargetPixelFile(TABBY_TPF), TessTargetPixelFile(filename_tess)]:
        random_mask = np.random.randint(0, 2, size=tpf.flux[0].shape, dtype=bool)
        thresh_mask = tpf.create_threshold_mask(threshold=3)

        lc = tpf.to_lightcurve(aperture_mask=random_mask)
        lc.to_fits(path=tempfile.NamedTemporaryFile().name, aperture_mask=random_mask)

        lc.to_fits(path=tempfile.NamedTemporaryFile().name, overwrite=True,
                  flux_column_name='SAP_FLUX')

        lc = tpf[0:2].to_lightcurve(aperture_mask=thresh_mask)
        lc.to_fits(aperture_mask=thresh_mask, path=tempfile.NamedTemporaryFile().name)

        # Test the extra data kwargs
        bkg_mask = ~tpf.create_threshold_mask(threshold=0.1)
        bkg_lc = tpf.to_lightcurve(aperture_mask=bkg_mask)
        lc = tpf.to_lightcurve(aperture_mask=tpf.hdu['APERTURE'].data)
        lc = tpf.to_lightcurve(aperture_mask=None)
        lc = tpf.to_lightcurve(aperture_mask=thresh_mask)
        lc_out = lc - bkg_lc.flux * (thresh_mask.sum()/bkg_mask.sum())
        lc_out.to_fits(aperture_mask=thresh_mask, path=tempfile.NamedTemporaryFile().name,
                       overwrite=True, extra_data={'BKG': bkg_lc.flux})


@pytest.mark.remote_data
def test_astropy_time():
    '''Test the `astropy_time` property'''
    lcf = KeplerLightCurveFile(TABBY_Q8)
    astropy_time = lcf.astropy_time
    iso = astropy_time.iso
    assert astropy_time.scale == 'tdb'
    assert len(iso) == len(lcf.time)
    #assert iso[0] == '2011-01-06 20:45:08.811'
    #assert iso[-1] == '2011-03-14 20:18:16.734'


def test_astropy_time_bkjd():
    """Does `LightCurve.astropy_time` support bkjd?"""
    bkjd = np.array([100, 200])
    lc = LightCurve(time=[100, 200], time_format='bkjd')
    assert_allclose(lc.astropy_time.jd, bkjd + 2454833.)


def test_lightcurve_repr():
    """Do __str__ and __repr__ work?"""
    time, flux = range(3), np.ones(3)
    str(LightCurve(time, flux))
    str(KeplerLightCurve(time, flux))
    str(TessLightCurve(time, flux))
    repr(LightCurve(time, flux))
    repr(KeplerLightCurve(time, flux))
    repr(TessLightCurve(time, flux))


@pytest.mark.remote_data
def test_lightcurvefile_repr():
    """Do __str__ and __repr__ work?"""
    lcf = KeplerLightCurveFile(TABBY_Q8)
    str(lcf)
    repr(lcf)
    lcf = TessLightCurveFile(TESS_SIM)
    str(lcf)
    repr(lcf)


def test_slicing():
    """Does LightCurve.__getitem__() allow slicing?"""
    time = np.linspace(0, 10, 10)
    flux = np.linspace(100, 200, 10)
    flux_err = np.linspace(5, 50, 10)
    lc = LightCurve(time, flux, flux_err)
    assert_array_equal(lc[0:5].time, time[0:5])
    assert_array_equal(lc[2::2].flux, flux[2::2])
    assert_array_equal(lc[5:9:-1].flux_err, flux_err[5:9:-1])

    # KeplerLightCurves contain additional data arrays that need to be sliced
    centroid_col = np.linspace(40, 50, 10)
    centroid_row = np.linspace(50, 60, 10)
    quality = np.linspace(70, 80, 10)
    cadenceno = np.linspace(90, 100, 10)
    lc = KeplerLightCurve(time, flux, flux_err,
                          centroid_col=centroid_col,
                          centroid_row=centroid_row,
                          cadenceno=cadenceno,
                          quality=quality)
    assert_array_equal(lc[::3].centroid_col, centroid_col[::3])
    assert_array_equal(lc[4:].centroid_row, centroid_row[4:])
    assert_array_equal(lc[10:2].quality, quality[10:2])
    assert_array_equal(lc[3:6].cadenceno, cadenceno[3:6])

    # The same is true for TessLightCurve
    lc = TessLightCurve(time, flux, flux_err,
                        centroid_col=centroid_col,
                        centroid_row=centroid_row,
                        cadenceno=cadenceno,
                        quality=quality)
    assert_array_equal(lc[::4].centroid_col, centroid_col[::4])
    assert_array_equal(lc[5:].centroid_row, centroid_row[5:])
    assert_array_equal(lc[10:3].quality, quality[10:3])
    assert_array_equal(lc[4:6].cadenceno, cadenceno[4:6])


def test_boolean_masking():
    lc = KeplerLightCurve(time=[1, 2, 3], flux=[1, 1, 10],
                          quality=[0, 0, 200], cadenceno=[5, 6, 7])
    assert_array_equal(lc[lc.flux < 5].time, [1, 2])
    assert_array_equal(lc[lc.flux < 5].flux, [1, 1])
    assert_array_equal(lc[lc.flux < 5].quality, [0, 0])
    assert_array_equal(lc[lc.flux < 5].cadenceno, [5, 6])


def test_remove_nans():
    """Does LightCurve.__getitem__() allow slicing?"""
    time, flux = [1, 2, 3, 4], [100, np.nan, 102, np.nan]
    lc_clean = LightCurve(time, flux).remove_nans()
    assert_array_equal(lc_clean.time, [1, 3])
    assert_array_equal(lc_clean.flux, [100, 102])


def test_remove_outliers():
    # Does `remove_outliers()` remove outliers?
    lc = LightCurve([1, 2, 3, 4], [1, 1, 1000, 1])
    lc_clean = lc.remove_outliers(sigma=1)
    assert_array_equal(lc_clean.time, [1, 2, 4])
    assert_array_equal(lc_clean.flux, [1, 1, 1])
    # It should also be possible to return the outlier mask
    lc_clean, outlier_mask = lc.remove_outliers(sigma=1, return_mask=True)
    assert(len(outlier_mask) == len(lc.flux))
    assert(outlier_mask.sum() == 1)
    # Can we set sigma_lower and sigma_upper?
    lc = LightCurve(time=[1, 2, 3, 4, 5], flux=[1, 1000, 1, -1000, 1])
    lc_clean = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper=1)
    assert_array_equal(lc_clean.time, [1, 3, 4, 5])
    assert_array_equal(lc_clean.flux, [1, 1, -1000, 1])


@pytest.mark.remote_data
def test_properties(capfd):
    '''Test if the describe function produces an output.
    The output is 624 characters at the moment, but we might add more properties.'''
    lcf = KeplerLightCurveFile(TABBY_Q8)
    kplc = lcf.get_lightcurve('SAP_FLUX')
    kplc.show_properties()
    out, _ = capfd.readouterr()
    assert len(out) > 500


def test_flatten_with_nans():
    """Flatten should not remove NaNs."""
    lc = LightCurve(time=[1, 2, 3, 4, 5],
                    flux=[np.nan, 1.1, 1.2, np.nan, 1.4],
                    flux_err=[1.0, np.nan, 1.2, 1.3, np.nan])
    flat_lc = lc.flatten(window_length=3)
    assert(len(flat_lc.time) == 5)
    assert(np.isfinite(flat_lc.flux).sum() == 3)
    assert(np.isfinite(flat_lc.flux_err).sum() == 3)


def test_flatten_robustness():
    """Test various special cases for flatten()."""
    # flatten should work with integer fluxes
    lc = LightCurve([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    expected_result = np.array([1.,  1.,  1.,  1.,  1., 1.])
    flat_lc = lc.flatten(window_length=3, polyorder=1)
    assert_allclose(flat_lc.flux, expected_result)
    # flatten should work even if `window_length > len(flux)`
    flat_lc = lc.flatten(window_length=7, polyorder=1)
    assert_allclose(flat_lc.flux, flat_lc.flux / np.median(flat_lc.flux))
    # flatten should work even if `polyorder >= window_length`
    flat_lc = lc.flatten(window_length=3, polyorder=3)
    assert_allclose(flat_lc.flux, expected_result)
    flat_lc = lc.flatten(window_length=3, polyorder=5)
    assert_allclose(flat_lc.flux, expected_result)
    # flatten should work even if `break_tolerance = None`
    flat_lc = lc.flatten(window_length=3, break_tolerance=None)
    assert_allclose(flat_lc.flux, expected_result)
    flat_lc, trend_lc = lc.flatten(return_trend=True)
    assert_allclose(flat_lc.time, trend_lc.time)
    assert_allclose(lc.flux, flat_lc.flux * trend_lc.flux)


def test_iterative_flatten():
    '''Test the iterative sigma clipping in flatten '''
    # Test a light curve with a single, buried outlier.
    x = np.arange(2000)
    y = np.sin(x/200)/100 + 1
    y[250] -= 0.01
    lc = LightCurve(x, y)
    # Flatten it
    c, f = lc.flatten(window_length=25, niters=2, sigma=3, return_trend=True)
    # Only one outlier should remain.
    assert np.isclose(c.flux, 1, rtol=0.00001).sum() == 1999
    mask = np.zeros(2000, dtype=bool)
    mask[250] = True
    # Flatten it using a mask to remove the bad data point.
    c, f = lc.flatten(window_length=25, niters=1, sigma=3, mask=mask,
                      return_trend=True)
    # Only one outlier should remain.
    assert np.isclose(c.flux, 1, rtol=0.00001).sum() == 1999


def test_fill_gaps():
    lc = LightCurve([1,2,3,4,6,7,8], [1,1,1,1,1,1,1])
    nlc = lc.fill_gaps()
    assert(len(lc.time) < len(nlc.time))
    assert(np.any(nlc.time == 5))
    assert(np.all(nlc.flux == 1))

    lc = LightCurve([1,2,3,4,6,7,8], [1,1,np.nan,1,1,1,1])
    nlc = lc.fill_gaps()
    assert(len(lc.time) < len(nlc.time))
    assert(np.any(nlc.time == 5))
    assert(np.all(nlc.flux == 1))
    assert(np.all(np.isfinite(nlc.flux)))

    # Because fill_gaps() uses pandas, check that it works regardless of endianness
    # For details see https://github.com/KeplerGO/lightkurve/issues/188
    lc = LightCurve(np.array([1, 2, 3, 4, 6, 7, 8], dtype='>f8'),
                    np.array([1, 1, 1, np.nan, np.nan, 1, 1], dtype='>f8'))
    lc.fill_gaps()
    lc = LightCurve(np.array([1, 2, 3, 4, 6, 7, 8], dtype='<f8'),
                    np.array([1, 1, 1, np.nan, np.nan, 1, 1], dtype='<f8'))
    lc.fill_gaps()


def test_targetid():
    """Is a generic targetid available on each type of LighCurve object?"""
    lc = LightCurve(time=[], targetid=5)
    assert lc.targetid == 5
    # Can we assign a new value?
    lc.targetid = 99
    assert lc.targetid == 99
    # Does it work for Kepler?
    lc = KeplerLightCurve(time=[], targetid=10)
    assert lc.targetid == 10
    # Can we assign a new value?
    lc.targetid = 99
    assert lc.targetid == 99
    # Does it work for TESS?
    lc = TessLightCurve(time=[], targetid=20)
    assert lc.targetid == 20


def test_regression_346():
    """Regression test for https://github.com/KeplerGO/lightkurve/issues/346"""
    # This previously triggered an IndexError:
    KeplerLightCurveFile(K2_C08).PDCSAP_FLUX.remove_nans().to_corrector().correct().estimate_cdpp()


def test_to_timeseries():
    """Test the `LightCurve.to_timeseries()` method."""
    time, flux, flux_err = np.arange(3)+2457576.4, np.ones(3), np.ones(3)*0.01
    lc = LightCurve(time, flux, flux_err, time_format="jd")
    try:
        ts = lc.to_timeseries()
        assert_allclose(ts['time'].value, time)
        assert_allclose(ts['flux'], flux)
        assert_allclose(ts['flux_err'], flux_err)
    except ImportError:
        # Requires AstroPy v3.2 or later
        pass


def test_flux_unit():
    """Checks the use of lc.flux_unit and lc.flux_quantity."""
    unit_obj = u.Unit("electron/second")
    # Can we set flux units using a Unit object?
    time, flux = range(3), np.ones(3)
    lc = LightCurve(time, flux, flux_unit=unit_obj)
    assert lc.flux_unit == unit_obj
    # Can we set flux units using a string?
    lc = LightCurve(time, flux, flux_unit="electron/second")
    assert lc.flux_unit == unit_obj
    # Can we pass a quantity to flux?
    lc = LightCurve(time, flux*unit_obj)
    assert lc.flux_unit == unit_obj
    # Can we retrieve correct flux quantities?
    assert lc.flux_quantity.unit ==unit_obj
    assert_array_equal(lc.flux_quantity.value, flux)
    # Is invalid user input validated?
    with pytest.raises(ValueError) as err:
        lc = LightCurve(time, flux, flux_unit="blablabla")
    assert "invalid `flux_unit`" in err.value.args[0]


def test_astropy_time_initialization():
    """Does the `LightCurve` constructor accept Astropy time objects?"""
    time = [1, 2, 3]
    lc = LightCurve(time=Time(time, format='jd', scale='utc'))
    assert lc.time_format == 'jd'
    assert lc.time_scale == 'utc'
    assert lc.astropy_time.format == 'jd'
    assert lc.astropy_time.scale == 'utc'
    lc = LightCurve(time=time, time_format='jd', time_scale='utc')
    assert lc.time_format == 'jd'
    assert lc.time_scale == 'utc'
    assert lc.astropy_time.format == 'jd'
    assert lc.astropy_time.scale == 'utc'


def test_normalize_unit():
    """Can the units of a normalized light curve be set?"""
    lc = LightCurve(flux=[1, 2, 3])
    for unit in ['percent', 'ppt', 'ppm']:
        assert lc.normalize(unit=unit).flux_unit.name == unit


def test_to_stingray():
    """Test the `LightCurve.to_stingray()` method."""
    time, flux, flux_err = range(3), np.ones(3), np.zeros(3)
    lc = LightCurve(time, flux, flux_err, time_format="jd")
    try:
        with warnings.catch_warnings():
            # Ignore "UserWarning: Numba not installed" raised by stingray.
            warnings.simplefilter("ignore", UserWarning)
            sr = lc.to_stingray()
        assert_allclose(sr.time, time)
        assert_allclose(sr.counts, flux)
        assert_allclose(sr.counts_err, flux_err)
    except ImportError:
        # Requires Stingray
        pass


def test_from_stingray():
    """Test the `LightCurve.from_stingray()` method."""
    try:
        from stingray import sampledata
        sr = sampledata.sample_data()
        lc = LightCurve.from_stingray(sr)
        assert_allclose(sr.time, lc.time)
        assert_allclose(sr.counts, lc.flux)
        assert_allclose(sr.counts_err, lc.flux_err)
    except ImportError:
        pass  # stingray is not a required dependency


def test_river():
    lc = LightCurve(time=np.arange(100), flux=np.random.normal(1, 0.01, 100))
    lc.plot_river(10, 1)
    plt.close()
    folded_lc = lc.fold(10, 1)
    folded_lc.plot_river()
    plt.close()
    folded_lc.plot_river(minimum_phase=-0.1, maximum_phase=0.2)
    plt.close()
    folded_lc.plot_river(method='median', bin_points=5)
    plt.close()
    folded_lc.plot_river(method='sigma', bin_points=5)
    plt.close()
    with pytest.warns(LightkurveWarning, match='`bin_points` is too high to plot'):
        folded_lc.plot_river(method='median', bin_points=6)
        plt.close()


# TEMPORARILY SKIP, cf. https://github.com/KeplerGO/lightkurve/issues/663
@pytest.mark.xfail
def test_bin_issue705():
    """Regression test for #705: binning failed."""
    lc = TessLightCurve(time=np.arange(50), flux=np.ones(50), quality=np.zeros(50))
    lc.bin(binsize=15)


@pytest.mark.remote_data
def test_SSOs():
    # TESS test
    lc = TessTargetPixelFile(asteroid_TPF).to_lightcurve(aperture_mask='all')
    result = lc.query_solar_system_objects(cadence_mask='all', cache=False)
    assert(len(result) == 1)
    result = lc.query_solar_system_objects(cadence_mask=np.asarray([True]), cache=False)
    assert(len(result) == 1)
    result, mask = lc.query_solar_system_objects(cadence_mask=np.asarray([True]), cache=True, return_mask=True)
    assert(len(mask) == len(lc.flux))


def test_get_header():
    """Test the basic functionality of ``tpf.get_header()``"""
    lcf = TessLightCurveFile(filename_tess_custom)
    assert lcf.get_header()['CREATOR'] == lcf.get_keyword("CREATOR")
    assert lcf.get_header(ext=2)['EXTNAME'] == "APERTURE"
    # ``tpf.header`` is deprecated
    with pytest.warns(LightkurveWarning, match='deprecated'):
        lcf.header()
