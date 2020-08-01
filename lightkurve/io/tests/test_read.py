import os
import warnings

import pytest

from ...utils import LightkurveDeprecationWarning, LightkurveError
from ... import PACKAGEDIR, KeplerTargetPixelFile, TessTargetPixelFile
from .. import read


def test_read():
    # define paths to k2 and tess data
    k2_path = os.path.join(PACKAGEDIR, "tests", "data", "test-tpf-star.fits")
    tess_path = os.path.join(PACKAGEDIR, "tests", "data", "tess25155310-s01-first-cadences.fits.gz")
    # Ensure files are read in as the correct object
    k2tpf = read(k2_path)
    assert(isinstance(k2tpf, KeplerTargetPixelFile))
    tesstpf = read(tess_path)
    assert(isinstance(tesstpf, TessTargetPixelFile))
    # Open should fail if the filetype is not recognized
    try:
        read(os.path.join(PACKAGEDIR, "data", "lightkurve.mplstyle"))
    except LightkurveError:
        pass
    # Can you instantiate with a path?
    assert(isinstance(KeplerTargetPixelFile(k2_path), KeplerTargetPixelFile))
    assert(isinstance(TessTargetPixelFile(tess_path), TessTargetPixelFile))
    # Can open take a quality_bitmask argument?
    assert(read(k2_path, quality_bitmask='hard').quality_bitmask == 'hard')


def test_open():
    """Does the deprecated `open` function still work?"""
    from .. import open
    with warnings.catch_warnings():  # lk.open is deprecated
        warnings.simplefilter("ignore", LightkurveDeprecationWarning)
        # define paths to k2 and tess data
        k2_path = os.path.join(PACKAGEDIR, "tests", "data", "test-tpf-star.fits")
        tess_path = os.path.join(PACKAGEDIR, "tests", "data", "tess25155310-s01-first-cadences.fits.gz")
        # Ensure files are read in as the correct object
        k2tpf = open(k2_path)
        assert(isinstance(k2tpf, KeplerTargetPixelFile))
        tesstpf = open(tess_path)
        assert(isinstance(tesstpf, TessTargetPixelFile))
        # Open should fail if the filetype is not recognized
        try:
            open(os.path.join(PACKAGEDIR, "data", "lightkurve.mplstyle"))
        except LightkurveError:
            pass
        # Can you instantiate with a path?
        assert(isinstance(KeplerTargetPixelFile(k2_path), KeplerTargetPixelFile))
        assert(isinstance(TessTargetPixelFile(tess_path), TessTargetPixelFile))
        # Can open take a quality_bitmask argument?
        assert(open(k2_path, quality_bitmask='hard').quality_bitmask == 'hard')


def test_filenotfound():
    """Regression test for #540; ensure lk.read() yields `FileNotFoundError`."""
    with pytest.raises(FileNotFoundError):
        read("DOESNOTEXIST")
