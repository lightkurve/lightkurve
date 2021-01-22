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

from ..targetpixelfile import KeplerTargetPixelFile, TargetPixelFileFactory
from ..targetpixelfile import TessTargetPixelFile, TargetPixelFile
from ..lightcurve import TessLightCurve
from ..utils import LightkurveWarning, LightkurveDeprecationWarning
from ..io import read
from ..search import search_tesscut

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


def test_tpf_math():
    """Can you add, subtract, multiply and divide?"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]

        # These should work
        for tpf in tpfs:
            for other in [1, np.ones(tpf.flux.shape[1:]), np.ones(tpf.shape)]:
                tpf + other
                tpf - other
                tpf * other
                tpf / other

                tpf += other
                tpf -= other
                tpf *= other
                tpf /= other

        # These should fail with a value error because their shape is wrong.
        for tpf in tpfs:
            for other in [np.asarray([1, 2]), np.arange(len(tpf.time) - 1),
                          np.ones([100, 1]), np.ones([1, 2, 3])]:
                with pytest.raises(ValueError):
                    tpf + other

        # Check the values are correct
        assert np.all(((tpf.flux.value + 2) == (tpf + 2).flux.value)[np.isfinite(tpf.flux)])
        assert np.all(((tpf.flux.value - 2) == (tpf - 2).flux.value)[np.isfinite(tpf.flux)])
        assert np.all(((tpf.flux.value * 2) == (tpf * 2).flux.value)[np.isfinite(tpf.flux)])
        assert np.all(((tpf.flux.value / 2) == (tpf / 2).flux.value)[np.isfinite(tpf.flux)])
        assert np.all(((tpf.flux_err.value * 2) == (tpf * 2).flux_err.value)[np.isfinite(tpf.flux)])
        assert np.all(((tpf.flux_err.value / 2) == (tpf / 2).flux_err.value)[np.isfinite(tpf.flux)])


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
        tpf.plot(column='FLUX')
        tpf.plot(column='FLUX_ERR')
        tpf.plot(column='FLUX_BKG')
        tpf.plot(column='FLUX_BKG_ERR')
        tpf.plot(column='RAW_CNTS')
        tpf.plot(column='COSMIC_RAYS')
        with pytest.raises(ValueError):
            tpf.plot(column='not a column')

        plt.close('all')


def test_tpf_zeros():
    """Does the LightCurve of a zero-flux TPF make sense?"""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask=None)
    with warnings.catch_warnings():
        # Ignore "LightCurve contains NaN times" warnings triggered by the liberal mask
        warnings.simplefilter("ignore", LightkurveWarning)
        lc = tpf.to_lightcurve()
    # If you don't mask out bad data, time contains NaNs
    assert np.any(lc.time.value != tpf.time)  # Using the property that NaN does not equal NaN
    # When you do mask out bad data everything should work.
    assert (tpf.time.value == 0).any()
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask='hard')
    lc = tpf.to_lightcurve(aperture_mask="all")
    assert len(lc.time) == len(lc.flux)
    assert np.all(lc.time == tpf.time)
    assert np.all(np.isnan(lc.flux)) # we expect all NaNs because of #874
    # The default QUALITY bitmask should have removed all NaNs in the TIME
    assert ~np.any(np.isnan(tpf.time.value))

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
        assert np.all(lc.flux.value == 1)
        assert np.all((lc.centroid_col.value < tpf.column+tpf.shape[1]).all()
                      * (lc.centroid_col.value > tpf.column).all())
        assert np.all((lc.centroid_row.value < tpf.row+tpf.shape[2]).all()
                      * (lc.centroid_row.value > tpf.row).all())


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


@pytest.mark.remote_data
@pytest.mark.parametrize("method", [("moments"), ("quadratic")])
def test_wcs_tabby(method):
    '''Test the centroids from Tabby's star against simbad values'''
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    tpf.wcs
    ra, dec = tpf.get_coordinates(0)
    col, row = tpf.estimate_centroids(method=method)
    col = col.value - tpf.column
    row = row.value - tpf.row
    y, x = int(np.round(col[0])), int(np.round(row[1]))
    # Compare with RA and Dec from Simbad
    assert np.isclose(ra[x, y], 301.5643971, 1e-4)
    assert np.isclose(dec[x, y], 44.4568869, 1e-4)


def test_centroid_methods_consistency():
    """Are the centroid methods consistent for a well behaved target?"""
    pixels = read(filename_synthetic_flat)
    centr_moments = pixels.estimate_centroids(method='moments')
    centr_quadratic = pixels.estimate_centroids(method='quadratic')
    # check that the maximum relative difference doesnt exceed 1%
    assert np.max(np.abs(centr_moments[0] - centr_quadratic[0]) / centr_moments[0]) < 1e-2
    assert np.max(np.abs(centr_moments[1] - centr_quadratic[1]) / centr_moments[1]) < 1e-2


def test_properties():
    """Test the short-hand properties."""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    assert(tpf.channel == tpf.hdu[0].header['CHANNEL'])
    assert(tpf.module == tpf.hdu[0].header['MODULE'])
    assert(tpf.output == tpf.hdu[0].header['OUTPUT'])
    assert(tpf.ra == tpf.hdu[0].header['RA_OBJ'])
    assert(tpf.dec == tpf.hdu[0].header['DEC_OBJ'])
    assert_array_equal(tpf.flux.value, tpf.hdu[1].data['FLUX'][tpf.quality_mask])
    assert_array_equal(tpf.flux_err.value, tpf.hdu[1].data['FLUX_ERR'][tpf.quality_mask])
    assert_array_equal(tpf.flux_bkg.value, tpf.hdu[1].data['FLUX_BKG'][tpf.quality_mask])
    assert_array_equal(tpf.flux_bkg_err.value, tpf.hdu[1].data['FLUX_BKG_ERR'][tpf.quality_mask])
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
        lc = tpf.to_lightcurve(aperture_mask='threshold')
        assert lc.time.scale == 'tdb'
        assert lc.label == tpf.hdu[0].header['OBJECT']
        if np.any(tpf.pipeline_mask):
            tpf.to_lightcurve(aperture_mask='pipeline')
        else:
            with pytest.raises(ValueError):
                tpf.to_lightcurve(aperture_mask='pipeline')


def test_bkg_lightcurve():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        lc = tpf.get_bkg_lightcurve()
        lc = tpf.get_bkg_lightcurve(aperture_mask=None)
        lc = tpf.get_bkg_lightcurve(aperture_mask='all')
        assert lc.time.scale == 'tdb'
        assert lc.flux.shape == lc.flux_err.shape
        assert len(lc.time) == len(lc.flux)


def test_aperture_photometry():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tess)]:
        tpf.extract_aperture_photometry()
        for mask in [None, 'all', 'default', 'threshold', 'background']:
            tpf.extract_aperture_photometry(aperture_mask=mask)
        if np.any(tpf.pipeline_mask):
            tpf.extract_aperture_photometry(aperture_mask='pipeline')
        else:
            with pytest.raises(ValueError):
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
    """Can we create TPFs using TargetPixelFileFactory?"""
    from lightkurve.targetpixelfile import FactoryError

    factory = TargetPixelFileFactory(n_cadences=10, n_rows=6, n_cols=8)
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
    tpf = factory.get_tpf(hdu0_keywords={'TELESCOP':'TESS'})

    assert_array_equal(tpf.flux[0].value, flux_0)
    assert_array_equal(tpf.flux[9].value, flux_9)

    tpf = factory.get_tpf(hdu0_keywords={'TELESCOP':'Kepler'})

    assert_array_equal(tpf.flux[0].value, flux_0)
    assert_array_equal(tpf.flux[9].value, flux_9)
    assert(tpf.time[0].value == 5)
    assert(tpf.time[9].value == 95)

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
    tpf = factory.get_tpf(hdu0_keywords={'creator': 'Christina TargetPixelFileWriter', 'TELESCOP':'TESS'})
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
        TargetPixelFile.from_fits_images(_create_image_array(), size=(3, 3),
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
    tpf = TargetPixelFile.from_fits_images(images, size=(3, 3),
                                                 position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))
    assert isinstance(tpf, TargetPixelFile)


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
        tpf_tmpfiles = TargetPixelFile.from_fits_images(tmpfile_names,
                            size=(3, 3),
                            position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))

        # Should be able to run with a list of HDUlists
        tpf_hdus = TargetPixelFile.from_fits_images(hdus,
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
        TargetPixelFile.from_fits_images(_create_image_array(), size=(3, 3),
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
    tpf = TargetPixelFile.from_fits_images(_create_image_array(header=header), size=(3, 3),
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
    assert tpf.time.scale == 'tdb'
    assert tpf.flux.shape == tpf.flux_err.shape
    tpf.wcs
    col, row = tpf.estimate_centroids()
    # Regression test for https://github.com/KeplerGO/lightkurve/pull/236
    assert (tpf.time.value == 0).sum() == 0


def test_threshold_aperture_mask():
    """Does the threshold mask work?"""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    tpf.plot(aperture_mask='threshold')
    lc = tpf.to_lightcurve(aperture_mask=tpf.create_threshold_mask(threshold=1))
    assert (lc.flux.value == 1).all()
    # The TESS file shows three pixel regions above a 2-sigma threshold;
    # let's make sure the `reference_pixel` argument allows them to be selected.
    tpf = TessTargetPixelFile(filename_tess)
    assert tpf.create_threshold_mask(threshold=2.).sum() == 25
    assert tpf.create_threshold_mask(threshold=2., reference_pixel='center').sum() == 25
    assert tpf.create_threshold_mask(threshold=2., reference_pixel=None).sum() == 28
    assert tpf.create_threshold_mask(threshold=2., reference_pixel=(5, 0)).sum() == 2
    # A mask which contains zero-flux pixels should work without crashing
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    assert tpf.create_threshold_mask().sum() == 9


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
    assert tpf.time.scale == 'tdb'
    assert tpf.flux.shape == tpf.flux_err.shape
    tpf.wcs
    col, row = tpf.estimate_centroids()


@pytest.mark.parametrize("tpf_type", [KeplerTargetPixelFile, TessTargetPixelFile])
def test_tpf_slicing(tpf_type):
    """Test indexing and slicing of TargetPixelFile objects."""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpf = tpf_type(filename_tpf_one_center)

    assert tpf[0].time == tpf.time[0]
    assert tpf[-1].time == tpf.time[-1]
    assert tpf[5:10].shape == tpf.flux[5:10].shape
    assert tpf[0].targetid == tpf.targetid
    assert_array_equal(tpf[tpf.time < tpf.time[5]].time, tpf.time[0:5])

    frame = tpf[5]
    assert frame.shape[0] == 1
    assert frame.shape[1:] == tpf.shape[1:]
    assert_array_equal(frame.time[0], tpf.time[5])
    assert_array_equal(frame.flux[0], tpf.flux[5])

    frames = tpf[100:200]
    assert frames.shape[0] == 100
    assert frames.shape[1:] == tpf.shape[1:]
    assert_array_equal(frames.time, tpf.time[100:200])
    assert_array_equal(frames.flux, tpf.flux[100:200])


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
    tpf = read(filename_tpf_one_center)
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
    result = tpf.query_solar_system_objects() # default cadence_mask = 'outliers'
    assert(result is None) # the TPF has only data for 1 epoch. The lone time is removed as outlier
    result = tpf.query_solar_system_objects(cadence_mask='all', cache=False)
    assert(len(result) == 1)
    result = tpf.query_solar_system_objects(cadence_mask=np.asarray([True]), cache=False)
    assert(len(result) == 1)
    result = tpf.query_solar_system_objects(cadence_mask=[True], cache=False)
    assert(len(result) == 1)
    result = tpf.query_solar_system_objects(cadence_mask=(True), cache=False)
    assert(len(result) == 1)
    result, mask = tpf.query_solar_system_objects(cadence_mask=np.asarray([True]), cache=True, return_mask=True)
    assert(len(mask) == len(tpf.flux))
    try:
        result = tpf.query_solar_system_objects(cadence_mask='str-not-supported', cache=False)
        pytest.fail("Unsupported cadence_mask should have thrown Error")
    except ValueError:
        pass


def test_get_header():
    """Test the basic functionality of ``tpf.get_header()``"""
    tpf = read(filename_tpf_one_center)
    assert tpf.get_header()['CHANNEL'] == tpf.get_keyword("CHANNEL")
    assert tpf.get_header(0)['MISSION'] == tpf.get_keyword("MISSION")
    assert tpf.get_header(ext=2)['EXTNAME'] == "APERTURE"
    # ``tpf.header`` is deprecated
    with pytest.warns(LightkurveDeprecationWarning, match='deprecated'):
        tpf.header


def test_plot_pixels():
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    tpf.plot_pixels()
    tpf.plot_pixels(normalize=True)
    tpf.plot_pixels(periodogram=True)
    tpf.plot_pixels(periodogram=True, nyquist_factor=0.5)
    tpf.plot_pixels(aperture_mask='all')
    tpf.plot_pixels(aperture_mask=tpf.pipeline_mask)
    tpf.plot_pixels(aperture_mask=tpf.create_threshold_mask())
    tpf.plot_pixels(show_flux=True)
    tpf.plot_pixels(corrector_func=lambda x:x)
    plt.close('all')


@pytest.mark.remote_data
def test_missing_pipeline_mask():
    """Regression test for #791.

    TPFs produced by TESSCut contain an empty pipeline mask.  When the pipeline
    mask is missing or empty, we want `to_lightcurve()` to fall back on the
    'threshold' mask by default, to avoid creating a light curve based on zero pixels."""
    tpf = search_tesscut("Proxima Cen", sector=12).download(cutout_size=1)
    lc = tpf.to_lightcurve()
    assert np.isfinite(lc.flux).any()
    assert lc.meta.get('APERTURE_MASK', None) == 'threshold'

    with pytest.raises(ValueError):
        # if aperture_mask is explicitly set as pipeline,
        # the logic will throw an error as it is missing in the TPF
        lc = tpf.to_lightcurve(aperture_mask='pipeline')


def test_cutout_quality_masking():
    """Regression test for #813: Does tpf.cutout() maintain the quality mask?"""
    tpf = read(filename_tpf_one_center, quality_bitmask=8192)
    tpfcut = tpf.cutout()
    assert(len(tpf) == len(tpfcut))


def test_parse_numeric_aperture_masks():
    """Regression test for #694: float or int aperture masks should be
    interpreted as boolean masks."""
    tpf = read(filename_tpf_one_center)
    mask = tpf._parse_aperture_mask(np.zeros(tpf.shape[1:], dtype=float))
    assert(mask.dtype == bool)
    mask = tpf._parse_aperture_mask(np.zeros(tpf.shape[1:], dtype=int))
    assert(mask.dtype == bool)


def test_tpf_meta():
    """Can we access meta data using tpf.meta?"""
    tpf = read(filename_tpf_one_center)
    assert tpf.meta.get('MISSION') == 'K2'
    assert tpf.meta['MISSION'] == 'K2'
    assert tpf.meta.get('mission', None) is None  # key is case in-sensitive
    assert tpf.meta.get('CHANNEL') == 45
    # ensure meta is read-only view of the underlying self.hdu[0].header
    with pytest.raises(TypeError):
        tpf.meta['CHANNEL'] = 44
    with pytest.raises(TypeError):
        tpf.meta['KEY-NEW'] = 44


def test_estimate_background():
    """Verifies tpf.estimate_background()."""
    # Create a TPF with 100 electron/second in every pixel
    tpf = read(filename_tpf_all_zeros) + 100.
    # The resulting background should be 100 e/s/pixel
    bg = tpf.estimate_background(aperture_mask='all')
    assert_array_equal(bg.flux.value, 100)
    assert bg.flux.unit == tpf.flux.unit / u.pixel


def test_fluxmode():
    """This should verify the median flux use in an aperture"""
    tpf = read(filename_tpf_one_center)
    lc_n = tpf.extract_aperture_photometry(aperture_mask="all")
    lc_sum = tpf.extract_aperture_photometry(aperture_mask="all", flux_method="sum")
    lc_med = tpf.extract_aperture_photometry(aperture_mask="all", flux_method="median")
    lc_mean = tpf.extract_aperture_photometry(aperture_mask="all", flux_method="mean")
    assert lc_n.flux.value[0] == np.nansum(tpf.flux.value[0])
    assert lc_sum.flux.value[0] == np.nansum(tpf.flux.value[0])
    assert lc_med.flux.value[0] == np.nanmedian(tpf.flux.value[0])
    assert lc_mean.flux.value[0] == np.nanmean(tpf.flux.value[0])
    
    
