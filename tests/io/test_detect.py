import os

from astropy.io import fits

from lightkurve import PACKAGEDIR
from lightkurve.io import detect_filetype

from .. import TESTDATA


def test_detect_filetype():
    """Can we detect the correct filetype?"""
    k2_path = os.path.join(TESTDATA, "test-tpf-star.fits")
    tess_path = os.path.join(TESTDATA, "tess25155310-s01-first-cadences.fits.gz")
    assert detect_filetype(fits.open(k2_path)) == "KeplerTargetPixelFile"
    assert detect_filetype(fits.open(tess_path)) == "TessTargetPixelFile"
