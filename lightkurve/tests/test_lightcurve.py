from __future__ import division, print_function

from astropy.io import fits as pyfits
from astropy.utils.data import get_pkg_data_filename
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
filename_tess = get_pkg_data_filename("data/tess25155310-s01-first-cadences.fits.gz")


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
        lc = LightCurve(time=time, flux=flux)


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

    # Does the data match what one would obtain using pyfits.open?
    hdu = pyfits.open(path)
    assert lc.label == hdu[0].header['OBJECT']
    assert_array_equal(lc.time, hdu[1].data['TIME'])
    assert_array_equal(lc.flux, hdu[1].data['SAP_FLUX'])

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

    # bad transit midpoint should give a warning
    # if user tries a t0 in JD but time is in BKJD
    with pytest.warns(LightkurveWarning, match='appears to be given in JD'):
        lc.fold(10, 2456600)


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
    with pytest.raises(AssertionError, match=r'ismatch.* 25'):
        assert_array_equal(lc.time, nlc.time)
    with pytest.raises(AssertionError, match=r'ismatch.* 25'):
        assert_array_equal(lc.flux, nlc.flux)
    with pytest.raises(AssertionError, match=r'ismatch.* 25'):
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
    with pytest.raises(AssertionError, match=r'ismatch.* 33\.3+'):
        assert_array_equal(lc.time, nlc.time)
    with pytest.raises(AssertionError, match=r'ismatch.* 33\.3+'):
        assert_array_equal(lc.flux, nlc.flux)
    with pytest.raises(AssertionError, match=r'ismatch.* 33\.3+'):
        assert_array_equal(lc.centroid_col, nlc.centroid_col)
    with pytest.raises(AssertionError, match=r'ismatch.* 33\.3+'):
        assert_array_equal(lc.centroid_row, nlc.centroid_row)
    with pytest.raises(AssertionError, match=r'ismatch.* 33\.3+'):
        assert_array_equal(lc.cadenceno, nlc.cadenceno)
    with pytest.raises(AssertionError, match=r'ismatch.* 33\.3+'):
        assert_array_equal(lc.quality, nlc.quality)


@pytest.mark.remote_data
def test_lightcurve_plots():
    """Sanity check to verify that lightcurve plotting works"""
    for lcf in [KeplerLightCurveFile(TABBY_Q8), TessLightCurveFile(TESS_SIM)]:
        lcf.plot()
        lcf.plot(flux_types=['SAP_FLUX', 'PDCSAP_FLUX'])
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
    assert(np.abs(lc.estimate_cdpp() - lcf.header(ext=1)['CDPP6_0']) < 30)


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
    lcf_new = KeplerLightCurveFile(hdu)  # Regression test for #233
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
    out, err = capfd.readouterr()
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
    KeplerLightCurveFile(K2_C08).PDCSAP_FLUX.to_corrector().correct().estimate_cdpp()


def test_new_corrector_api():
    """This test can be remove after we remove the deprecated `LightCurve.correct()` method"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LightkurveWarning)  # Deprecation warning
        lc1 = KeplerLightCurveFile(K2_C08).PDCSAP_FLUX.correct()
    lc2 = KeplerLightCurveFile(K2_C08).PDCSAP_FLUX.to_corrector().correct()
    assert_allclose(lc1.flux, lc2.flux)
