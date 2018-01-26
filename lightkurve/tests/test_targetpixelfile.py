import numpy as np
import pytest
from astropy.utils.data import get_pkg_data_filename
from ..targetpixelfile import KeplerTargetPixelFile, KeplerQualityFlags


filename_tpf_all_zeros = get_pkg_data_filename("data/test-tpf-all-zeros.fits")
filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")

def test_load_bad_file():
    '''Test if a light curve can be opened without exception.'''
    with pytest.raises(ValueError) as exc:
        tpf = KeplerTargetPixelFile(TABBY_Q8)
    assert('is this a target pixel file?' in exc.value.args[0])

def test_tpf_shapes():
    """Are the data array shapes of the TargetPixelFile object consistent?"""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    assert tpf.quality_mask.shape == tpf.hdu[1].data['TIME'].shape
    assert tpf.flux.shape == tpf.flux_err.shape


def test_tpf_plot():
    """Sanity check to verify that tpf plotting works"""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    tpf.plot()
    tpf.plot(aperture_mask=tpf.pipeline_mask)


def test_tpf_zeros():
    """Does the LightCurve of a zero-flux TPF make sense?"""
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask=None)
    lc = tpf.to_lightcurve()
    # If you don't mask out bad data, time contains NaNs
    assert np.any(lc.time != tpf.time)  # Using the property that NaN does not equal NaN
    #When you do mask out bad data everything should work.
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, quality_bitmask='hard')
    lc = tpf.to_lightcurve()
    assert len(lc.time) == len(lc.flux)
    assert np.all(lc.time == tpf.time)
    assert np.all(lc.flux == 0)
    # The default QUALITY bitmask should have removed all NaNs in the TIME
    assert ~np.any(np.isnan(tpf.time))


def test_tpf_ones():
    """Does the LightCurve of a one-flux TPF make sense?"""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    lc = tpf.to_lightcurve(aperture_mask='all')
    assert np.all(lc.flux == 1)
    assert np.all((lc.centroid_col < tpf.column+tpf.shape[1]).all()
                  * (lc.centroid_col > tpf.column).all())
    assert np.all((lc.centroid_row < tpf.row+tpf.shape[2]).all()
                  * (lc.centroid_row > tpf.row).all())


def test_quality_flag_decoding():
    """Can the QUALITY flags be parsed correctly?"""
    flags = list(KeplerQualityFlags.STRINGS.items())
    for key, value in flags:
        assert KeplerQualityFlags.decode(key)[0] == value
    # Can we recover combinations of flags?
    assert KeplerQualityFlags.decode(flags[5][0] + flags[7][0]) == [flags[5][1], flags[7][1]]
    assert KeplerQualityFlags.decode(flags[3][0] + flags[4][0] + flags[5][0]) \
        == [flags[3][1], flags[4][1], flags[5][1]]


@pytest.mark.parametrize("quality_bitmask,answer",[('hardest', 1101),
    ('hard', 1101), ('default', 1233), (None, 1290),
    (1, 1290), (100, 1278), (2096639, 1101)])
def test_bitmasking(quality_bitmask, answer):
    '''Test whether the bitmasking behaves like it should'''
    tpf = KeplerTargetPixelFile(filename_tpf_one_center, quality_bitmask=quality_bitmask)
    lc = tpf.to_lightcurve()
    flux = lc.flux
    assert len(flux) == answer
