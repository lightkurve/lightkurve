from __future__ import division, print_function

import os
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tempfile

from ..targetpixelfile import KeplerTargetPixelFile, KeplerTargetPixelFileFactory
from ..targetpixelfile import TessTargetPixelFile


filename_tpf_all_zeros = get_pkg_data_filename("data/test-tpf-all-zeros.fits")
filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")
TABBY_TPF = ("https://archive.stsci.edu/missions/kepler/target_pixel_files"
             "/0084/008462852/kplr008462852-2011073133259_lpd-targ.fits.gz")
TESS_SIM = ("https://archive.stsci.edu/missions/tess/ete-6/tid/00/000"
            "/004/176/tess2019128220341-0000000417699452-0016-s_tp.fits")

@pytest.mark.remote_data
def test_load_bad_file():
    '''Test if a light curve can be opened without exception.'''
    with pytest.raises(ValueError) as exc:
        tpf = KeplerTargetPixelFile(TABBY_Q8)
    assert('is this a target pixel file?' in exc.value.args[0])
    with pytest.raises(ValueError) as exc:
        tpf = TessTargetPixelFile(TABBY_Q8)
    assert('is this a target pixel file?' in exc.value.args[0])


def test_tpf_shapes():
    """Are the data array shapes of the TargetPixelFile object consistent?"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]:
        assert tpf.quality_mask.shape == tpf.hdu[1].data['TIME'].shape
        assert tpf.flux.shape == tpf.flux_err.shape


def test_tpf_plot():
    """Sanity check to verify that tpf plotting works"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tpf_one_center)]:
        ax = tpf.plot()
        tpf.plot(aperture_mask=tpf.pipeline_mask)
        tpf.plot(aperture_mask='all')
        tpf.plot(frame=5)
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
    lc = tpf.to_lightcurve()
    # If you don't mask out bad data, time contains NaNs
    assert np.any(lc.time != tpf.time)  # Using the property that NaN does not equal NaN
    # When you do mask out bad data everything should work.
    assert (tpf.astropy_time.jd == 0).any()
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask='hard')
    lc = tpf.to_lightcurve()
    assert len(lc.time) == len(lc.flux)
    assert np.all(lc.time == tpf.time)
    assert np.all(lc.flux == 0)
    # The default QUALITY bitmask should have removed all NaNs in the TIME
    assert ~np.any(np.isnan(tpf.time))


def test_tpf_ones():
    """Does the LightCurve of a one-flux TPF make sense?"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tpf_one_center)]:
        lc = tpf.to_lightcurve(aperture_mask='all')
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
    '''Test whether the bitmasking behaves like it should'''
    tpf = KeplerTargetPixelFile(filename_tpf_one_center, quality_bitmask=quality_bitmask)
    lc = tpf.to_lightcurve()
    flux = lc.flux
    assert len(flux) == answer


def test_wcs():
    """Test the wcs property."""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tpf_one_center)]:
        w = tpf.wcs
        ra, dec = tpf.get_coordinates()
        assert ra.shape == tpf.shape
        assert dec.shape == tpf.shape
        assert type(w).__name__ == 'WCS'


def test_wcs_tabby():
    '''Test the centroids from Tabby's star against simbad values'''
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    w = tpf.wcs
    ra, dec = tpf.get_coordinates(0)
    col, row = tpf.centroids()
    col -= tpf.column
    row -= tpf.row
    y, x = int(np.round(col[0])), int(np.round(row[1]))
    # Compare with RA and Dec from Simbad
    assert np.isclose(ra[x, y], 301.5643971, 1e-4)
    assert np.isclose(dec[x, y], 44.4568869, 1e-4)


def test_astropy_time():
    '''Test the lc.date() function'''
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]:
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
                TessTargetPixelFile(filename_tpf_all_zeros)]:
        str(tpf)
        repr(tpf)


def test_to_lightcurve():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]:
        tpf.to_lightcurve()
        tpf.to_lightcurve(aperture_mask=None)
        tpf.to_lightcurve(aperture_mask='all')
        lc = tpf.to_lightcurve(aperture_mask='pipeline')
        assert lc.astropy_time.scale == 'tdb'
        assert lc.label == tpf.hdu[0].header['OBJECT']


def test_bkg_lightcurve():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]:
        lc = tpf.get_bkg_lightcurve()
        lc = tpf.get_bkg_lightcurve(aperture_mask=None)
        lc = tpf.get_bkg_lightcurve(aperture_mask='all')
        assert lc.astropy_time.scale == 'tdb'
        assert lc.flux.shape == lc.flux_err.shape
        assert len(lc.time) == len(lc.flux)


def test_aperture_photometry():
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]:
        tpf.aperture_photometry()
        tpf.aperture_photometry(aperture_mask=None)
        tpf.aperture_photometry(aperture_mask='all')
        tpf.aperture_photometry(aperture_mask='pipeline')

def test_tpf_to_fits():
    """Can we write a TPF back to a fits file?"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]:
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
    with pytest.raises(FactoryError) as exc:
        tpf = factory.get_tpf()
    [factory.add_cadence(frameno=i, flux=flux_0,
                        header={'TSTART': i*10, 'TSTOP': (i*10)+10})  for i in np.arange(2, 9)]

    # This should fail because the time stamps of the images are not in order...
    with pytest.raises(FactoryError) as exc:
        tpf = factory.get_tpf()

    [factory.add_cadence(frameno=i, flux=flux_0,
                        header={'TSTART': i*10, 'TSTOP': (i*10)+10})  for i in np.arange(1, 9)]

    # This should pass
    tpf = factory.get_tpf()

    assert_array_equal(tpf.flux[0], flux_0)
    assert_array_equal(tpf.flux[9], flux_9)
    assert(tpf.time[0] == 5)
    assert(tpf.time[9] == 95)

    # Can you add the WRONG sized frame?
    flux_wrong = 3 * np.ones((6, 9))
    with pytest.raises(FactoryError) as exc:
        factory.add_cadence(frameno=2, flux=flux_wrong,
                            header={'TSTART': 90, 'TSTOP': 100})

    # Can you add the WRONG cadence?
    flux_wrong = 3 * np.ones((6, 8))
    with pytest.raises(FactoryError) as exc:
        factory.add_cadence(frameno=11, flux=flux_wrong,
                            header={'TSTART': 90, 'TSTOP': 100})

    # Can we add our own keywords?
    tpf = factory.get_tpf(hdu0_keywords = {'creator': 'Christina'})
    assert tpf.header()['CREATOR'] == 'Christina'


def test_tpf_from_images():
    """Basic tests of tpf.from_fits_images()"""
    from glob import glob

    from astropy.io import fits
    from astropy import wcs
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from lightkurve.targetpixelfile import FactoryError

    # Can we read in a load of images?
    header = fits.Header()
    images = []
    for i in range(5):
        header['TSTART'] = i
        header['TSTOP'] = i + 1
        images.append(fits.ImageHDU(data=np.ones((5, 5)), header=header))

    # Not without a wcs...
    with pytest.raises(FactoryError) as exc:
        tpf_list = KeplerTargetPixelFile.from_fits_images(images, size=(3, 3))

    # Make a fake WCS based on astropy.docs...
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [-234.75, 8.3393]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])
    pixcrd = np.array([[0, 0], [24, 38], [45, 98]], np.float_)
    header = w.to_header()
    ra, dec = 268.21686048, -73.66991904

    # Add that header to our images...
    images = []
    for i in range(5):
        header['TSTART'] = i
        header['TSTOP'] = i + 1
        images.append(fits.ImageHDU(data=np.ones((5, 5)), header=header))

    # Now this should work.
    tpf_list = KeplerTargetPixelFile.from_fits_images(images, size=(3, 3), position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))
    assert isinstance(tpf_list, KeplerTargetPixelFile)

    # Can we read in a list of file names or a list of HDUlists?
    hdus = []
    for idx, im in enumerate(images):
        hdu = fits.HDUList([fits.PrimaryHDU(), im])
        hdu.writeto(get_pkg_data_filename('data/test_factory{}.fits'.format(idx)), overwrite=True)
        hdus.append(hdu)

    fnames = [get_pkg_data_filename('data/test_factory{}.fits'.format(i)) for i in range(5)]

    # Should be able to run with a list of file names
    tpf_fnames = KeplerTargetPixelFile.from_fits_images(fnames,
                                                        size=(3, 3),
                                                        position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))

    # Should be able to run with a list of HDUlists
    tpf_hdus = KeplerTargetPixelFile.from_fits_images(hdus,
                                                        size=(3, 3),
                                                        position=SkyCoord(ra, dec, unit=(u.deg, u.deg)))


def test_properties2(capfd):
    '''Test if the describe function produces an output.
    The output is 1870 characters at the moment, but we might add more properties.'''
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf.properties()
    out, err = capfd.readouterr()
    assert len(out) > 1000


def test_interact():
    """Test the Jupyter notebook interact() widget."""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tpf_one_center)]:
        tpf.interact()
        tpf.interact(lc=tpf.to_lightcurve(aperture_mask='all'))


def test_from_archive_should_accept_path():
    """If a path is accidentally passed to `from_archive` it should still just work."""
    KeplerTargetPixelFile.from_archive(filename_tpf_all_zeros)


def test_from_fits():
    """Does the tpf.from_fits() method work like the constructor?"""
    tpf = KeplerTargetPixelFile.from_fits(filename_tpf_one_center)
    assert isinstance(tpf, KeplerTargetPixelFile)
    assert tpf.targetid == KeplerTargetPixelFile(filename_tpf_one_center).targetid
    # Execute the same test for TESS
    tpf = TessTargetPixelFile.from_fits(filename_tpf_one_center)
    assert isinstance(tpf, TessTargetPixelFile)
    assert tpf.targetid == TessTargetPixelFile(filename_tpf_one_center).targetid


def test_get_models():
    """Can we obtain PRF and TPF models?"""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask=None)
    tpf.get_model()
    tpf.get_prf_model()


#@pytest.mark.remote_data
def test_tess_simulation():
    """Can we read simulated TESS data?"""
    tpf = TessTargetPixelFile(TESS_SIM)
    assert tpf.mission == 'TESS'
    assert tpf.astropy_time.scale == 'tdb'
    assert tpf.flux.shape == tpf.flux_err.shape
    tpf.wcs
    col, row = tpf.centroids()
    # Regression test for https://github.com/KeplerGO/lightkurve/pull/236
    assert np.isnan(tpf.time).sum() == 0
