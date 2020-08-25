import os

from astropy.io import fits

from ... import PACKAGEDIR
from .. import detect_filetype


def test_detect_filetype():
    """Can we detect the correct filetype?"""
    k2_path = os.path.join(PACKAGEDIR, "tests", "data", "test-tpf-star.fits")
    tess_path = os.path.join(PACKAGEDIR, "tests", "data", "tess25155310-s01-first-cadences.fits.gz")
    assert detect_filetype(fits.open(k2_path)) == 'KeplerTargetPixelFile'
    assert detect_filetype(fits.open(tess_path)) == 'TessTargetPixelFile'
