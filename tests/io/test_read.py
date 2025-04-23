import gc
import os
import warnings
import tempfile
import tracemalloc

import pytest

from astropy.io import fits

from lightkurve.collections import LightCurveCollection, TargetPixelFileCollection
from lightkurve.utils import LightkurveDeprecationWarning, LightkurveError
from lightkurve import (
    PACKAGEDIR,
    KeplerTargetPixelFile,
    TessTargetPixelFile,
    LightCurve,
)
from lightkurve.io import read, read_lc_collection, read_tpf_collection
from lightkurve.io.generic import read_generic_lightcurve

from .. import TESTDATA
from ..test_lightcurve import TABBY_Q8
from ..test_targetpixelfile import TABBY_TPF

#
# For tests with pytest error::ResourceWarning
# and error::pytest.PytestUnraisableExceptionWarning
# they are to ensure all internal file handles are closed in read operations
# (ResourceWarning in case of unclosed file handles,
#  is wrapped by as PytestUnraisableExceptionWarning by pytest)
#

@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_read_lc():
    filename_lc = os.path.join(TESTDATA, "test-lc-tess-pimen-100-cadences.fits")
    lc = read(filename_lc)
    assert isinstance(lc, LightCurve)


@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_read_lc_in_hdu():
    filename_lc = os.path.join(TESTDATA, "test-lc-tess-pimen-100-cadences.fits")
    hdul = fits.open(filename_lc)
    # lk.read() does not support hdul as input
    lc = read_generic_lightcurve(hdul, flux_column="pdcsap_flux", time_format="btjd")
    hdul.close()
    assert len(lc.flux) > 0, "LC should be functional even the hdul is closed."


def test_read_lc_cloud():
    """Read a lightcurve file from AWS S3 cloud"""
    cloud_uri = 's3://stpubdata/tess/public/tid/s0015/0000/0003/7542/2201/tess2019226182529-s0015-0000000375422201-0151-s_lc.fits'
    lc = read(cloud_uri)
    assert isinstance(lc, LightCurve)


# tpf.hdu has open file handle, so they are not tested for unclosed file handles
def test_read_tpf():
    # define paths to k2 and tess data
    k2_path = os.path.join(TESTDATA, "test-tpf-star.fits")
    tess_path = os.path.join(TESTDATA, "tess25155310-s01-first-cadences.fits.gz")
    # Ensure files are read in as the correct object
    k2tpf = read(k2_path)
    assert isinstance(k2tpf, KeplerTargetPixelFile)
    tesstpf = read(tess_path)
    assert isinstance(tesstpf, TessTargetPixelFile)
    # Open should fail if the filetype is not recognized
    try:
        read(os.path.join(PACKAGEDIR, "data", "lightkurve.mplstyle"))
    except LightkurveError:
        pass
    # Can you instantiate with a path?
    assert isinstance(KeplerTargetPixelFile(k2_path), KeplerTargetPixelFile)
    assert isinstance(TessTargetPixelFile(tess_path), TessTargetPixelFile)
    # Can open take a quality_bitmask argument?
    assert read(k2_path, quality_bitmask="hard").quality_bitmask == "hard"


def test_read_tpf_cloud():
    """Read a TPF file from AWS S3 cloud"""
    cloud_uri = 's3://stpubdata/kepler/public/target_pixel_files/0082/008264588/kplr008264588-2009131105131_lpd-targ.fits.gz'
    tpf = read(cloud_uri)
    assert isinstance(tpf, KeplerTargetPixelFile)


def test_read_lc_collection():
    """Read multiple light curve files into a collection"""
    path_list = ['s3://stpubdata/kepler/public/lightcurves/0082/008264588/kplr008264588-2009131105131_llc.fits',
                 's3://stpubdata/kepler/public/lightcurves/0082/008264588/kplr008264588-2009166043257_llc.fits',
                 's3://stpubdata/kepler/public/lightcurves/0082/008264588/kplr008264588-2009259160929_llc.fits']
    collection = read_lc_collection(path_list)
    assert isinstance(collection, LightCurveCollection)

    # Check that stitching produces a single light curve
    stitched = read_lc_collection(path_list, stitch=True)
    assert isinstance(stitched, LightCurve)

    # Checking edge cases - path to a TPF file and an invalid path
    # Neither path should produce a light curve, and resulting collection should be empty
    path_empty = ['s3://stpubdata/kepler/public/target_pixel_files/0082/008264588/kplr008264588-2009131105131_lpd-targ.fits.gz',
                  's3://invalid']
    empty_collection = read_lc_collection(path_empty)
    assert isinstance(empty_collection, LightCurveCollection)
    assert not empty_collection.data


def test_read_tpf_collection():
    """Read multiple TPF files into a collection"""
    path_list = ['s3://stpubdata/tess/public/tid/s0015/0000/0003/7542/2201/tess2019226182529-s0015-0000000375422201-0151-s_tp.fits',
                 's3://stpubdata/tess/public/tid/s0016/0000/0003/7542/2201/tess2019253231442-s0016-0000000375422201-0152-s_tp.fits',
                 's3://stpubdata/tess/public/tid/s0017/0000/0003/7542/2201/tess2019279210107-s0017-0000000375422201-0161-s_tp.fits']

    collection = read_tpf_collection(path_list)
    assert isinstance(collection, TargetPixelFileCollection)


def test_open():
    """Does the deprecated `open` function still work?"""
    from lightkurve.io import open

    with warnings.catch_warnings():  # lk.open is deprecated
        warnings.simplefilter("ignore", LightkurveDeprecationWarning)
        # define paths to k2 and tess data
        k2_path = os.path.join(TESTDATA, "test-tpf-star.fits")
        tess_path = os.path.join(TESTDATA, "tess25155310-s01-first-cadences.fits.gz")
        # Ensure files are read in as the correct object
        k2tpf = open(k2_path)
        assert isinstance(k2tpf, KeplerTargetPixelFile)
        tesstpf = open(tess_path)
        assert isinstance(tesstpf, TessTargetPixelFile)
        # Open should fail if the filetype is not recognized
        try:
            open(os.path.join(PACKAGEDIR, "data", "lightkurve.mplstyle"))
        except LightkurveError:
            pass
        # Can you instantiate with a path?
        assert isinstance(KeplerTargetPixelFile(k2_path), KeplerTargetPixelFile)
        assert isinstance(TessTargetPixelFile(tess_path), TessTargetPixelFile)
        # Can open take a quality_bitmask argument?
        assert open(k2_path, quality_bitmask="hard").quality_bitmask == "hard"


def test_filenotfound():
    """Regression test for #540; ensure lk.read() yields `FileNotFoundError`."""
    filename = "some/path/DOESNOTEXIST"
    with pytest.raises(FileNotFoundError) as excinfo:
        read(filename)
    # ensure the filepath is in the exception
    assert filename in str(excinfo.value)


@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
# ignore AstropyUserWarning: File may have been truncated  (for corrupted at data section FITS)
@pytest.mark.filterwarnings("ignore:.*been truncated.*")
# ignore  VerifyWarning: Error validating header for HDU #1 ... (for test-lc-tess-pimen-corrupted-at-header.fits )
@pytest.mark.filterwarnings("ignore:.*Error validating header.*")
@pytest.mark.parametrize("fits_name", [
    # TPF, truncated somewhere in the data section, the error is in 'TypeError: buffer is too small for requested array'
    "test-tpf-kplr-tabby-corrupted.fits",
    # TPF, truncated somewhere in the header (BINTABLE), the error is 'IndexError: list index out of range'
    "test-tpf-kplr-tabby-corrupted-at-header.fits",
    # TPF, truncated somewhere in the header (PRIMARY),
    "test-tpf-kplr-tabby-corrupted-at-header2.fits",

    # TPF, TESS variants, some code paths are TESS / Kepler specific
    # so TESS TPFs are included to complete the coverage
    # source: mast:TESS/product/tess2023209231226-s0068-0000000261136679-0262-s_tp.fits
    #  (pi Men, sector 68, SPOC 2min cadence)
    "test-lc-tess-pimen-corrupted.fits",
    "test-lc-tess-pimen-corrupted-at-header.fits",
    "test-lc-tess-pimen-corrupted-at-header2.fits",

    # LC, truncated in data section ; source: mast:TESS/product/tess2018206045859-s0001-0000000261136679-0120-s_lc.fits
    "test-lc-tess-pimen-corrupted.fits",
    # LC, truncated in header (BINTABLE)
    "test-lc-tess-pimen-corrupted-at-header.fits",
    # LC, truncated in header (PRIMARY)
    "test-lc-tess-pimen-corrupted-at-header2.fits",
])
def test_file_corrupted(fits_name):
    """Regression test for #1184; ensure lk.read() yields an error that includes the filename."""
    filename_fits_corrupted = os.path.join(TESTDATA, fits_name)
    with pytest.raises(BaseException) as excinfo:
        read(filename_fits_corrupted)
    # ensure the filepath is in the exception
    assert filename_fits_corrupted in str(excinfo.value)


def test_basic_ascii_io():
    """Verify we do not break the basic ascii i/o functionality provided by AstroPy Table."""
    # Part I: Can we read a LightCurve from a CSV file?
    csvfile = tempfile.NamedTemporaryFile(
        delete=False
    )  # using delete=False to make tests pass on Windows
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
        lc_csv.write(tabfile.name, format="ascii.tab", overwrite=True)
        lc_rst = LightCurve.read(tabfile.name, format="ascii.tab")
        assert lc_rst.color[2] == "blue"
        assert (lc_csv == lc_rst).all()
    finally:
        tabfile.close()
        os.remove(tabfile.name)


@pytest.mark.memtest
@pytest.mark.remote_data
@pytest.mark.parametrize("fits_path, iterations_warmup, run_iterations", [
    (TABBY_Q8, 40, 60),
    (TABBY_TPF, 40, 60),
    ])
def test_read_memory_usage(fits_path, iterations_warmup, run_iterations):
    """Ensure reading LC/TPF has no memory leak. Regression test for #1388.
    The test uses real data rather than trimmed-down test data
    to better simulate real life scenarios.
    """
    def do_read():
        # do the actual read in a function,
        # to ensure object read is out-of-scope and to be freed up after it's done,
        # simulating the typical scenario
        obj_read = read(fits_path)
        return len(obj_read)

    tracemalloc.start()
    try:
        h_current, h_peak = [], []  # history of tracemalloc for error reporting
        for _ in range(iterations_warmup):
            do_read()
            current, peak = tracemalloc.get_traced_memory()
            h_current.append(current)
            h_peak.append(peak)
        gc.collect()  # run GC so that the number would be more consistent across runs
        current, peak = tracemalloc.get_traced_memory()
        post_warmup_mem, post_warmup_peak = current, peak
        h_current.append(current)
        h_peak.append(f"{peak} (post-warmup, after GC)")

        for _ in range(run_iterations):
            do_read()
            current, peak = tracemalloc.get_traced_memory()
            h_current.append(current)
            h_peak.append(peak)
        gc.collect()  # run GC so that the number would be more consistent across runs
        current, peak = tracemalloc.get_traced_memory()
        post_run_mem, post_run_peak = current, peak
        h_current.append(current)
        h_peak.append(f"{peak} (post-run, after GC)")

        # if the test fails, print out detailed history of the memory usage for diagnosis
        assert_err_msg = ("Memory usage should not keep increasing. "
                          f"After warmup ({iterations_warmup}): mem: {post_warmup_mem}, peak: {post_warmup_peak} . "
                          f"After run ({run_iterations} more): mem: {post_run_mem}, peak: {post_run_peak} . "
                          f"History of (current, peak):\n "
                          )
        assert_err_msg += "\n".join([str((c, p)) for c, p in zip(h_current, h_peak)])

        # leave plenty buffer (2X of post warm-up memory) for the memory leak detection
        # if there is a leak, the actual memory usage at the end is likely to be significantly higher.
        assert post_run_mem < 2 * post_warmup_mem, assert_err_msg
    finally:
        tracemalloc.stop()
