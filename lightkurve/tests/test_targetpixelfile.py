from __future__ import division, print_function

import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from astropy.utils.data import get_pkg_data_filename
from astropy.io.fits.verify import VerifyWarning
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import wcs
from astropy.io.fits.card import UNDEFINED
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

from ..targetpixelfile import KeplerTargetPixelFile, KeplerTargetPixelFileFactory
from ..targetpixelfile import TessTargetPixelFile
from ..lightcurve import TessLightCurve
from ..utils import LightkurveWarning
from ..search import open as lkopen

from .test_synthetic_data import filename_synthetic_flat

filename_tpf_all_zeros = get_pkg_data_filename("data/test-tpf-all-zeros.fits")
filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")
filename_tess = get_pkg_data_filename("data/tess25155310-s01-first-cadences.fits.gz")

TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")
TABBY_TPF = ("https://archive.stsci.edu/missions/kepler/target_pixel_files"
             "/0084/008462852/kplr008462852-2011073133259_lpd-targ.fits.gz")
TESS_SIM = ("https://archive.stsci.edu/missions/tess/ete-6/tid/00/000"
            "/004/176/tess2019128220341-0000000417699452-0016-s_tp.fits")
asteroid_TPF = get_pkg_data_filename("data/asteroid_test.fits")


@pytest.mark.remote_data
def test_load_bad_file():
    '''Test if a light curve can be opened without exception.'''
    with pytest.raises(ValueError) as exc:
        KeplerTargetPixelFile(TABBY_Q8)
    assert('is this a target pixel file?' in exc.value.args[0])
    with pytest.raises(ValueError) as exc:
        TessTargetPixelFile(TABBY_Q8)
    assert('is this a target pixel file?' in exc.value.args[0])


def test_tpf_shapes():
    """Are the data array shapes of the TargetPixelFile object consistent?"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]
    for tpf in tpfs:
        assert tpf.quality_mask.shape == tpf.hdu[1].data['TIME'].shape
        assert tpf.flux.shape == tpf.flux_err.shape


def test_tpf_plot():
    """Sanity check to verify that tpf plotting works"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tpf_one_center)]
    for tpf in tpfs:
        tpf.plot()
        tpf.plot(aperture_mask=tpf.pipeline_mask)
        tpf.plot(aperture_mask='all')
        tpf.plot(frame=3)
        with pytest.raises(ValueError):
            tpf.plot(frame=999999)
        tpf.plot(cadenceno=125250)
        with pytest.raises(ValueError):
            tpf.plot(cadenceno=999)
        tpf.plot(bkg=True)
        tpf.plot(scale="sqrt")
        tpf.plot(scale="log")
        with pytest.raises(ValueError):
            tpf.plot(scale="blabla")
        plt.close('all')


def test_tpf_zeros():
    """Does the LightCurve of a zero-flux TPF make sense?"""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask=None)
    with warnings.catch_warnings():
        # Ignore "LightCurve contains NaN times" warnings triggered by the liberal mask
        warnings.simplefilter("ignore", LightkurveWarning)
        lc = tpf.to_lightcurve()
    # If you don't mask out bad data, time contains NaNs
    assert np.any(lc.time != tpf.time)  # Using the property that NaN does not equal NaN
    # When you do mask out bad data everything should work.
    assert (tpf.astropy_time.jd == 0).any()
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask='hard')
    lc = tpf.to_lightcurve(aperture_mask="all")
    assert len(lc.time) == len(lc.flux)
    assert np.all(lc.time == tpf.time)
    assert np.all(lc.flux == 0)
    # The default QUALITY bitmask should have removed all NaNs in the TIME
    assert ~np.any(np.isnan(tpf.time))

@pytest.mark.parametrize("centroid_method", [("moments"), ("quadratic")])
def test_tpf_ones(centroid_method):
    """Does the LightCurve of a one-flux TPF make sense?"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tpf_one_center)]
    for tpf in tpfs:
        lc = tpf.to_lightcurve(aperture_mask='all', centroid_method=centroid_method)
        assert np.all(lc.flux == 1)
        assert np.all((lc.centroid_col < tpf.column+tpf.shape[1]).all()
                      * (lc.centroid_col > tpf.column).all())
        assert np.all((lc.centroid_row < tpf.row+tpf.shape[2]).all()
                      * (lc.centroid_row > tpf.row).all())


@pytest.mark.parametrize("quality_bitmask,answer", [(None, 1290), ('none', 1290),
                                                    ('default', 1233),
                                                    ('hard', 1101), ('hardest', 1101),
                                                    (1, 1290), (100, 1278), (2096639, 1101)])
def test_bitmasking(quality_bitmask, answer):
    """Test whether the bitmasking behaves like it should"""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center, quality_bitmask=quality_bitmask)
    with warnings.catch_warnings():
        # Ignore "LightCurve contains NaN times" warnings triggered by liberal masks
        warnings.simplefilter("ignore", LightkurveWarning)
        lc = tpf.to_lightcurve()
    assert len(lc.flux) == answer


def test_wcs():
    """Test the wcs property."""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tess)]:
        w = tpf.wcs
        ra, dec = tpf.get_coordinates()
        assert ra.shape == tpf.shape
        assert dec.shape == tpf.shape
        assert type(w).__name__ == 'WCS'


@pytest.mark.parametrize("method", [("moments"), ("quadratic")])
def test_wcs_tabby(method):
    '''Test the centroids from Tabby's star against simbad values'''
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    tpf.wcs
    ra, dec = tpf.get_coordinates(0)
    col, row = tpf.estimate_centroids(method=method)
    col -= tpf.column
    row -= tpf.row
    y, x = int(np.round(col[0])), int(np.round(row[1]))
    # Compare with RA and Dec from Simbad
    assert np.isclose(ra[x, y], 301.5643971, 1e-4)
    assert np.isclose(dec[x, y], 44.4568869, 1e-4)


def test_centroid_methods_consistency():
    """Are the centroid methods consistent for a well behaved target?"""
    pixels = lkopen(filename_synthetic_flat)
    centr_moments = pixels.estimate_centroids(method='moments')
    centr_quadratic = pixels.estimate_centroids(method='quadratic')
    # check that the maximum relative difference doesnt exceed 1%
    assert np.max(np.abs(centr_moments[0] - centr_quadratic[0]) / centr_moments[0]) < 1e-2
    assert np.max(np.abs(centr_moments[1] - centr_quadratic[1]) / centr_moments[1]) < 1e-2


def test_astropy_time():
    '''Test the lc.date() function'''
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        astropy_time = tpf.astropy_time
        assert astropy_time.scale == 'tdb'
        assert len(astropy_time.iso) == len(tpf.time)


def test_properties():
    """Test the short-hand properties."""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    assert(tpf.channel == tpf.hdu[0].header['CHANNEL'])
    assert(tpf.module == tpf.hdu[0].header['MODULE'])
    assert(tpf.output == tpf.hdu[0].header['OUTPUT'])
    assert(tpf.ra == tpf.hdu[0].header['RA_OBJ'])
    assert(tpf.dec == tpf.hdu[0].header['DEC_OBJ'])
    assert_array_equal(tpf.flux, tpf.hdu[1].data['FLUX'][tpf.quality_mask])
    assert_array_equal(tpf.flux_err, tpf.hdu[1].data['FLUX_ERR'][tpf.quality_mask])
    assert_array_equal(tpf.flux_bkg, tpf.hdu[1].data['FLUX_BKG'][tpf.quality_mask])
    assert_array_equal(tpf.flux_bkg_err, tpf.hdu[1].data['FLUX_BKG_ERR'][tpf.quality_mask])
    assert_array_equal(tpf.quality, tpf.hdu[1].data['QUALITY'][tpf.quality_mask])
    assert(tpf.campaign == tpf.hdu[0].header['CAMPAIGN'])
    assert(tpf.quarter is None)


def test_repr():
    """Do __str__ and __repr__ work?"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        str(tpf)
        repr(tpf)


def test_to_lightcurve():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        tpf.to_lightcurve()
        tpf.to_lightcurve(aperture_mask=None)
        tpf.to_lightcurve(aperture_mask='all')
        lc = tpf.to_lightcurve(aperture_mask='pipeline')
        assert lc.astropy_time.scale == 'tdb'
        assert lc.label == tpf.hdu[0].header['OBJECT']


def test_bkg_lightcurve():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        lc = tpf.get_bkg_lightcurve()
        lc = tpf.get_bkg_lightcurve(aperture_mask=None)
        lc = tpf.get_bkg_lightcurve(aperture_mask='all')
        assert lc.astropy_time.scale == 'tdb'
        assert lc.flux.shape == lc.flux_err.shape
        assert len(lc.time) == len(lc.flux)


def test_aperture_photometry():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        tpf.extract_aperture_photometry()
        tpf.extract_aperture_photometry(aperture_mask=None)
        tpf.extract_aperture_photometry(aperture_mask='all')
        tpf.extract_aperture_photometry(aperture_mask='pipeline')


def test_tpf_to_fits():
    """Can we write a TPF back to a fits file?"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        # `delete=False` is necessary to enable writing to the file on Windows
        # but it means we have to clean up the tmp file ourselves
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            tpf.to_fits(tmp.name)
        finally:
            tmp.close()
            os.remove(tmp.name)


def test_tpf_factory():
    """Can we create TPFs using KeplerTargetPixelFileFactory?"""
    from lightkurve.targetpixelfile import FactoryError

    factory = KeplerTargetPixelFileFactory(n_cadences=10, n_rows=6, n_cols=8)
    flux_0 = np.ones((6, 8))
    factory.add_cadence(frameno=0, flux=flux_0,
                        header={'TSTART': 0, 'TSTOP': 10})
    flux_9 = 3 * np.ones((6, 8))
    factory.add_cadence(frameno=9, flux=flux_9,
                        header={'TSTART': 90, 'TSTOP': 100})

    # You shouldn't be able to build a TPF like this...because TPFs shouldn't
    # have extensions where time stamps are duplicated (here frames 1-8 will have)
    # time stamp zero
    with pytest.warns(LightkurveWarning, match='identical TIME values'):
        tpf = factory.get_tpf()
    [factory.add_cadence(frameno=i, flux=flux_0,
                         header={'TSTART': i*10, 'TSTOP': (i*10)+10})
     for i in np.arange(2, 9)]

    # This should fail because the time stamps of the images are not in order...
    with pytest.warns(LightkurveWarning, match='chronological order'):
        tpf = factory.get_tpf()

    [factory.add_cadence(frameno=i, flux=flux_0,
                         header={'TSTART': i*10, 'TSTOP': (i*10)+10})
     for i in np.arange(1, 9)]

    # This should pass
    tpf = factory.get_tpf()

    assert_array_equal(tpf.flux[0], flux_0)
    assert_array_equal(tpf.flux[9], flux_9)
    assert(tpf.time[0] == 5)
    assert(tpf.time[9] == 95)

    # Can you add the WRONG sized frame?
    flux_wrong = 3 * np.ones((6, 9))
    with pytest.raises(FactoryError):
        factory.add_cadence(frameno=2, flux=flux_wrong,
                            header={'TSTART': 90, 'TSTOP': 100})

    # Can you add the WRONG cadence?
    flux_wrong = 3 * np.ones((6, 8))
    with pytest.raises(FactoryError):
        factory.add_cadence(frameno=11, flux=flux_wrong,
                            header={'TSTART': 90, 'TSTOP': 100})

    # Can we add our own keywords?
    tpf = factory.get_tpf(hdu0_keywords={'creator': 'Christina TargetPixelFileWriter'})
    assert tpf.get_keyword('CREATOR') == 'Christina TargetPixelFileWriter'


def _create_image_array(header=None, shape=(5, 5)):
    """Helper function for tests below."""
    if header is None:
        header = fits.Header()
    images = []
    for i in range(5):
        header['TSTART'] = i
        header['TSTOP'] = i + 1
        images.append(fits.ImageHDU(data=np.ones(shape), header=header))
    return images

def test_tpf_from_images():
    """Basic tests of tpf.from_fits_images()"""
    # Not without a wcs...
    with pytest.raises(Exception):
        KeplerTargetPixelFile.from_fits_images(_create_image_array(), size=(3, 3),
                                               position=SkyCoord(-234.75, 8.3393, unit='deg'))

    # Make a fake WCS based on astropy.docs...
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [-234.75, 8.3393]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])
    pixcrd = np.array([[0, 0], [24, 38], [45, 98]], np.float_)
    header = w.to_header()
    header['CRVAL1P'] = 10
    header['CRVAL2P'] = 20
    ra, dec = 268.21686048, -73.66991904

    # Now this should work.
    images = _create_image_array(header=header)
    tpf = KeplerTargetPixelFile.from_fits_images(images, size=(3, 3),
                                                 position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))
    assert isinstance(tpf, KeplerTargetPixelFile)


    with warnings.catch_warnings():
        # Some cards are too long -- to be investigated.
        warnings.simplefilter("ignore", VerifyWarning)
        # Can we write the output to disk?
        # `delete=False` is necessary below to enable writing to the file on Windows
        # but it means we have to clean up the tmp file ourselves
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            tpf.to_fits(tmp.name)
        finally:
            tmp.close()
            os.remove(tmp.name)

        # Can we read in a list of file names or a list of HDUlists?
        hdus = []
        tmpfile_names = []
        for im in images:
            tmpfile = tempfile.NamedTemporaryFile(delete=False)
            tmpfile_names.append(tmpfile.name)
            hdu = fits.HDUList([fits.PrimaryHDU(), im])
            hdu.writeto(tmpfile.name)
            hdus.append(hdu)

        # Should be able to run with a list of file names
        tpf_tmpfiles = KeplerTargetPixelFile.from_fits_images(tmpfile_names,
                            size=(3, 3),
                            position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))

        # Should be able to run with a list of HDUlists
        tpf_hdus = KeplerTargetPixelFile.from_fits_images(hdus,
                            size=(3, 3),
                            position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))

        # Clean up the temporary files we created
        for filename in tmpfile_names:
            try:
                os.remove(filename)
            except PermissionError:
                pass  # This appears to happen on Windows


def test_tpf_wcs_from_images():
    """Test to see if tpf.from_fits_images() output a tpf with WCS in the header"""
    # Not without a wcs...
    with pytest.raises(Exception):
        KeplerTargetPixelFile.from_fits_images(_create_image_array(), size=(3, 3),
                                               position=SkyCoord(-234.75, 8.3393, unit='deg'))

    # Make a fake WCS based on astropy.docs...
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [0., 0.]
    w.wcs.cdelt = np.array([0.001111, 0.001111])
    w.wcs.crval = [23.2334, 45.2333]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    header['CRVAL1P'] = 10
    header['CRVAL2P'] = 20
    ra, dec = 23.2336, 45.235

    # Now this should work.
    tpf = KeplerTargetPixelFile.from_fits_images(_create_image_array(header=header), size=(3, 3),
                                                 position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))
    assert tpf.hdu[1].header['1CRPX5'] != UNDEFINED
    assert tpf.hdu[1].header['1CTYP5'] == 'RA---TAN'
    assert tpf.hdu[1].header['2CTYP5'] == 'DEC--TAN'
    assert tpf.hdu[1].header['1CRPX5'] != UNDEFINED
    assert tpf.hdu[1].header['2CRPX5'] != UNDEFINED
    assert tpf.hdu[1].header['1CUNI5'] == 'deg'
    assert tpf.hdu[1].header['2CUNI5'] == 'deg'
    with warnings.catch_warnings():
        # Ignore the warning: "PC1_1 = a floating-point value was expected."
        warnings.simplefilter("ignore", AstropyWarning)
        assert tpf.wcs.to_header()['CDELT1'] == w.wcs.cdelt[0]


def test_properties2(capfd):
    '''Test if the describe function produces an output.
    The output is 1870 characters at the moment, but we might add more properties.'''
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf.show_properties()
    out, err = capfd.readouterr()
    assert len(out) > 1000


def test_interact():
    """Test the Jupyter notebook interact() widget."""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tess)]:
        tpf.interact()


@pytest.mark.remote_data
def test_interact_sky():
    """Test the Jupyter notebook interact() widget."""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tess)]:
        tpf.interact_sky()


def test_get_models():
    """Can we obtain PRF and TPF models?"""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask=None)
    with warnings.catch_warnings():
        # Ignore "RuntimeWarning: All-NaN slice encountered"
        warnings.simplefilter("ignore", RuntimeWarning)
        tpf.get_model()
        tpf.get_prf_model()


@pytest.mark.remote_data
def test_tess_simulation():
    """Can we read simulated TESS data?"""
    tpf = TessTargetPixelFile(TESS_SIM)
    assert tpf.mission == 'TESS'
    assert tpf.astropy_time.scale == 'tdb'
    assert tpf.flux.shape == tpf.flux_err.shape
    tpf.wcs
    col, row = tpf.estimate_centroids()
    # Regression test for https://github.com/KeplerGO/lightkurve/pull/236
    assert np.isnan(tpf.time).sum() == 0


def test_threshold_aperture_mask():
    """Does the threshold mask work?"""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    tpf.plot(aperture_mask='threshold')
    lc = tpf.to_lightcurve(aperture_mask=tpf.create_threshold_mask(threshold=1))
    assert (lc.flux == 1).all()
    # The TESS file shows three pixel regions above a 2-sigma threshold;
    # let's make sure the `reference_pixel` argument allows them to be selected.
    tpf = TessTargetPixelFile(filename_tess)
    assert tpf.create_threshold_mask(threshold=2.).sum() == 25
    assert tpf.create_threshold_mask(threshold=2., reference_pixel='center').sum() == 25
    assert tpf.create_threshold_mask(threshold=2., reference_pixel=None).sum() == 28
    assert tpf.create_threshold_mask(threshold=2., reference_pixel=(5, 0)).sum() == 2
    # A mask which contains zero pixels should work without crashing
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    assert tpf.create_threshold_mask().sum() == 0


def test_tpf_tess():
    """Does a TESS Sector 1 TPF work?"""
    tpf = TessTargetPixelFile(filename_tess, quality_bitmask=None)
    assert tpf.mission == 'TESS'
    assert tpf.targetid == 25155310
    assert tpf.sector == 1
    assert tpf.camera == 4
    assert tpf.ccd == 1
    assert tpf.pipeline_mask.sum() == 9
    assert tpf.background_mask.sum() == 30
    lc = tpf.to_lightcurve()
    assert isinstance(lc, TessLightCurve)
    assert_array_equal(lc.time, tpf.time)
    assert tpf.astropy_time.scale == 'tdb'
    assert tpf.flux.shape == tpf.flux_err.shape
    tpf.wcs
    col, row = tpf.estimate_centroids()


def test_tpf_slicing():
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    assert tpf[0].time == tpf.time[0]
    assert tpf[-1].time == tpf.time[-1]
    assert tpf[5:10].shape == tpf.flux[5:10].shape
    assert tpf[0].targetid == tpf.targetid
    assert_array_equal(tpf[tpf.time < tpf.time[5]].time, tpf.time[0:5])


def test_endianness():
    """Regression test for https://github.com/KeplerGO/lightkurve/issues/188"""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    tpf.to_lightcurve().to_pandas().describe()


def test_get_keyword():
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    assert tpf.get_keyword("TELESCOP") == "Kepler"
    assert tpf.get_keyword("TTYPE1", hdu=1) == "TIME"
    assert tpf.get_keyword("DOESNOTEXIST", default=5) == 5


def test_cutout():
    """Test tpf.cutout() function."""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tess, quality_bitmask=None)]:
        ntpf = tpf.cutout(size=2)
        assert ntpf.flux[0].shape == (2, 2)
        assert ntpf.flux_err[0].shape == (2, 2)
        assert ntpf.flux_bkg[0].shape == (2, 2)
        ntpf = tpf.cutout((0, 0), size=3)
        ntpf = tpf.cutout(size=(1, 2))
        assert ntpf.flux.shape[1] == 2
        assert ntpf.flux.shape[2] == 1
        ntpf = tpf.cutout(SkyCoord(tpf.ra, tpf.dec, unit='deg'), size=2)
        ntpf = tpf.cutout(size=2)
        assert np.product(ntpf.flux.shape[1:]) == 4
        assert ntpf.targetid == '{}_CUTOUT'.format(tpf.targetid)


def test_aperture_photometry_nan():
    """Regression test for #648.

    When FLUX or FLUX_ERR is entirely NaN in a TPF, the resulting light curve
    should report NaNs in that cadence rather than zero."""
    tpf = lkopen(filename_tpf_one_center)
    tpf.hdu[1].data['FLUX'][2] = np.nan
    tpf.hdu[1].data['FLUX_ERR'][2] = np.nan
    lc = tpf.to_lightcurve(aperture_mask='all')
    assert ~np.isnan(lc.flux[1])
    assert ~np.isnan(lc.flux_err[1])
    assert np.isnan(lc.flux[2])
    assert np.isnan(lc.flux_err[2])

@pytest.mark.remote_data
def test_SSOs():
    # TESS test
    tpf = TessTargetPixelFile(asteroid_TPF)
    result = tpf.query_solar_system_objects(cadence_mask='all', cache=False)
    assert(len(result) == 1)
    result = tpf.query_solar_system_objects(cadence_mask=np.asarray([True]), cache=False)
    assert(len(result) == 1)
    result, mask = tpf.query_solar_system_objects(cadence_mask=np.asarray([True]), cache=True, return_mask=True)
    assert(len(mask) == len(tpf.flux))


def test_get_header():
    """Test the basic functionality of ``tpf.get_header()``"""
    tpf = lkopen(filename_tpf_one_center)
    assert tpf.get_header()['CHANNEL'] == tpf.get_keyword("CHANNEL")
    assert tpf.get_header(0)['MISSION'] == tpf.get_keyword("MISSION")
    assert tpf.get_header(ext=2)['EXTNAME'] == "APERTURE"
    # ``tpf.header`` is deprecated
    with pytest.warns(LightkurveWarning, match='deprecated'):
        tpf.header
