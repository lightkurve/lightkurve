import os
import warnings
import tempfile

import pytest

from ...utils import LightkurveDeprecationWarning, LightkurveError
from ... import PACKAGEDIR, KeplerTargetPixelFile, TessTargetPixelFile, LightCurve
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


def test_basic_ascii_io():
    """Verify we do not break the basic ascii i/o functionality provided by AstroPy Table."""
    # Part I: Can we read a LightCurve from a CSV file?
    csvfile = tempfile.NamedTemporaryFile(delete=False)  # using delete=False to make tests pass on Windows
    try:
        csvfile.write(b"time,flux,flux_err,color\n1,2,3,red\n4,5,6,green\n7,8,9,blue")
        csvfile.flush()
        lc_csv = LightCurve.read(csvfile.name, format="ascii.csv")
        assert lc_csv.time[0].value == 1
        assert lc_csv.flux[1] == 5
        assert lc_csv.color[2] == "blue"
    finally:
        csvfile.close()
        os.remove(csvfile.name)

    # Part II: can we write the light curve to a tab-separated ascii file, and read it back in?
    tabfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        lc_csv.write(tabfile.name, format='ascii.tab', overwrite=True)
        lc_rst = LightCurve.read(tabfile.name, format='ascii.tab')
        assert lc_rst.color[2] == "blue"
        assert (lc_csv == lc_rst).all()
    finally:
        tabfile.close()
        os.remove(tabfile.name)
