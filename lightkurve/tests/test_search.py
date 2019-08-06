"""Test features of lightkurve that interact with the data archive at MAST.

Note: if you have the `pytest-remotedata` package installed, then tests flagged
with the `@pytest.mark.remote_data` decorator below will only run if the
`--remote-data` argument is passed to py.test.  This allows tests to pass
if no internet connection is available.
"""
import os
import sys
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import tempfile
import warnings

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

from ..utils import LightkurveWarning
from ..search import search_lightcurvefile, search_targetpixelfile, \
                     search_tesscut, SearchResult, SearchError, open
from .. import KeplerLightCurveFile
from .. import KeplerTargetPixelFile, TessTargetPixelFile, TargetPixelFileCollection

from .. import PACKAGEDIR


@pytest.mark.remote_data
def test_search_targetpixelfile():
    # EPIC 210634047 was observed twice in long cadence
    assert(len(search_targetpixelfile(210634047).table) == 2)
    # ...including Campaign 4
    assert(len(search_targetpixelfile(210634047, campaign=4).table) == 1)
    # KIC 11904151 (Kepler-10) was observed in LC in 15 Quarters
    assert(len(search_targetpixelfile(11904151).table) == 15)
    # ...including quarter 11 but not 12:
    assert(len(search_targetpixelfile(11904151, quarter=11).unique_targets) == 1)
    assert(len(search_targetpixelfile(11904151, quarter=12).table) == 0)
    # should work for all split campaigns
    campaigns = [[91, 92, 9], [101, 102, 10], [111, 112, 11]]
    ids = [200068780, 200071712, 202975993]
    for c, idx in zip(campaigns, ids):
        ca = search_targetpixelfile(idx, campaign=c[0]).table
        cb = search_targetpixelfile(idx, campaign=c[1]).table
        assert(len(ca) == 1)
        assert(len(ca) == len(cb))
        assert(~np.any(ca['description'] == cb['description']))
        # If you specify the whole campaign, both split parts must be returned.
        cc = search_targetpixelfile(idx, campaign=c[2]).table
        assert(len(cc) == 2)
    search_targetpixelfile(11904151, quarter=11).download()
    # with mission='TESS', it should return TESS observations
    tic = 273985862  # Has been observed in multiple sectors including 1
    assert(len(search_targetpixelfile(tic, mission='TESS').table) > 1)
    assert(len(search_targetpixelfile(tic, mission='TESS', sector=1, radius=100).table) == 2)
    search_targetpixelfile(tic, mission='TESS', sector=1).download()
    assert(len(search_targetpixelfile("pi Mensae", sector=1).table) == 1)
    # Issue #445: indexing with -1 should return the last index of the search result
    assert(len(search_targetpixelfile("pi Men")[-1]) == 1)


@pytest.mark.remote_data
def test_search_lightcurvefile(caplog):
    # We should also be able to resolve it by its name instead of KIC ID
    assert(len(search_lightcurvefile('Kepler-10').table) == 15)
    # An invalid KIC/EPIC ID or target name should be dealt with gracefully
    search_lightcurvefile(-999)
    assert "Could not resolve" in caplog.text
    search_lightcurvefile("DOES_NOT_EXIST (UNIT TEST)")
    assert "Could not resolve" in caplog.text
    # If we ask for all cadence types, there should be four Kepler files given
    assert(len(search_lightcurvefile(4914423, quarter=6, cadence='any').table) == 4)
    # ...and only one should have long cadence
    assert(len(search_lightcurvefile(4914423, quarter=6, cadence='long').table) == 1)
    # Should be able to resolve an ra/dec
    assert(len(search_lightcurvefile('297.5835, 40.98339', quarter=6).table) == 1)
    # Should be able to resolve a SkyCoord
    c = SkyCoord('297.5835 40.98339', unit=(u.deg, u.deg))
    assert(len(search_lightcurvefile(c, quarter=6).table) == 1)
    search_lightcurvefile(c, quarter=6).download()
    # with mission='TESS', it should return TESS observations
    tic = 273985862
    assert(len(search_lightcurvefile(tic, mission='TESS').table) > 1)
    assert(len(search_lightcurvefile(tic, mission='TESS', sector=1, radius=100).table) == 2)
    search_lightcurvefile(tic, mission='TESS', sector=1).download()
    assert(len(search_lightcurvefile("pi Mensae", sector=1).table) == 1)


@pytest.mark.remote_data
def test_search_tesscut():
    # Cutout by target name
    assert(len(search_tesscut("pi Mensae", sector=1).table) == 1)
    assert(len(search_tesscut("pi Mensae").table) > 1)
    # Cutout by TIC ID
    assert(len(search_tesscut('TIC 206669860', sector=2).table) == 1)
    # Cutout by RA, dec string
    search_string = search_tesscut('30.578761, -83.210593')
    # Cutout by SkyCoord
    c = SkyCoord('30.578761 -83.210593', unit=(u.deg, u.deg))
    search_coords = search_tesscut(c)
    # These should be identical
    assert(len(search_string.table) == len(search_coords.table))
    # Test cutout at edge of FFI
    search_edge = search_tesscut('30.578761, 6.210593')
    assert(len(search_edge.table) == 1)
    try:
        # This is too near the FFI edge and should fail to download
        search_edge.download()
    except SearchError:
        pass


@pytest.mark.remote_data
def test_search_tesscut_download():
    """Can we download TESS cutouts via `search_cutout().download()?"""
    ra, dec = 30.578761, -83.210593
    search_string = search_tesscut('{}, {}'.format(ra, dec))
    # Make sure they can be downloaded with default size
    tpf = search_string.download()
    # Ensure the correct object has been read in
    assert(isinstance(tpf, TessTargetPixelFile))
    # Ensure default size is 5x5
    assert(tpf.flux[0].shape == (5, 5))
    # Ensure the tpf has a targetid (#473 regression test)
    assert(len(tpf.targetid) > 0)
    # Ensure the WCS is valid (#434 regression test)
    center_ra, center_dec = tpf.wcs.all_pix2world([[2.5, 2.5]], 1)[0]
    assert_almost_equal(ra, center_ra, decimal=1)
    assert_almost_equal(dec, center_dec, decimal=1)
    # Download with different dimensions
    tpfc = search_string.download_all(cutout_size=4, quality_bitmask='hard')
    assert(isinstance(tpfc, TargetPixelFileCollection))
    assert(tpfc[0].quality_bitmask == 'hard')  # Regression test for #494
    # Ensure correct dimensions
    assert(tpfc[0].flux[0].shape == (4, 4))
    # Download with rectangular dimennsions?
    rect_tpf = search_string.download(cutout_size=(3, 5))
    assert(rect_tpf.flux[0].shape == (3, 5))


@pytest.mark.remote_data
def test_search_with_skycoord():
    """Can we pass both names, SkyCoord objects, and coordinate strings?"""
    sr_name = search_targetpixelfile("Kepler-10")
    assert len(sr_name) == 15  # Kepler-10 as observed during 15 quarters in long cadence
    # Can we search using a SkyCoord objects?
    sr_skycoord = search_targetpixelfile(SkyCoord.from_name("Kepler_10"))
    assert_array_equal(sr_name.table['productFilename'], sr_skycoord.table['productFilename'])
    # Can we search using a string of "ra dec" decimals?
    sr_decimal = search_targetpixelfile("285.67942179 +50.24130576")
    assert_array_equal(sr_name.table['productFilename'], sr_decimal.table['productFilename'])
    # Can we search using a sexagesimal string?
    sr_sexagesimal = search_targetpixelfile("19:02:43.1 +50:14:28.7")
    assert_array_equal(sr_name.table['productFilename'], sr_sexagesimal.table['productFilename'])
    # Can we search using the KIC ID?
    sr_kic = search_targetpixelfile(11904151)
    assert_array_equal(sr_name.table['productFilename'], sr_kic.table['productFilename'])


@pytest.mark.remote_data
def test_searchresult():
    sr = search_lightcurvefile('Kepler-10')
    assert len(sr) == len(sr.table)  # Tests SearchResult.__len__
    assert len(sr[2:7]) == 5  # Tests SearchResult.__get__
    assert len(sr[2]) == 1
    assert "kplr" in str(sr)  # Tests SearchResult.__repr__


@pytest.mark.remote_data
def test_month():
    # In short cadence, if we specify both quarter and month
    sr = search_targetpixelfile('Kepler-10', quarter=11, month=1, cadence='short')
    assert(len(sr) == 1)
    sr = search_targetpixelfile('Kepler-10', quarter=11, month=[1, 3], cadence='short')
    assert(len(sr) == 2)


@pytest.mark.remote_data
def test_collections():
    # TargetPixelFileCollection class
    assert(len(search_targetpixelfile(205998445, radius=900).table) == 4)
    # LightCurveFileCollection class with set targetlimit
    assert(len(search_lightcurvefile(205998445, radius=900, limit=3).download_all()) == 3)
    # if fewer targets are found than targetlimit, should still download all available
    assert(len(search_targetpixelfile(205998445, radius=900, limit=6).table) == 4)
    # if download() is used when multiple files are available, should only download 1
    with pytest.warns(LightkurveWarning, match='4 files available to download'):
        assert(isinstance(search_targetpixelfile(205998445, radius=900).download(),
                          KeplerTargetPixelFile))


@pytest.mark.remote_data
def test_properties():
    c = SkyCoord('297.5835 40.98339', unit=(u.deg, u.deg))
    assert_almost_equal(search_targetpixelfile(c, quarter=6).ra, 297.5835)
    assert_almost_equal(search_targetpixelfile(c, quarter=6).dec, 40.98339)
    assert(len(search_targetpixelfile(c, quarter=6).target_name) == 1)
    assert(len(search_targetpixelfile(c, quarter=6).obsid) == 1)


@pytest.mark.remote_data
def test_source_confusion():
    # Regression test for issue #148.
    # When obtaining the TPF for target 6507433, @benmontet noticed that
    # a target 4 arcsec away was returned instead.
    # See https://github.com/KeplerGO/lightkurve/issues/148
    desired_target = 6507433
    tpf = search_targetpixelfile(desired_target, quarter=8).download()
    assert tpf.targetid == desired_target


def test_empty_searchresult():
    """Does an empty SearchResult behave gracefully?"""
    sr = SearchResult(Table())
    assert len(sr) == 0
    str(sr)
    with pytest.warns(LightkurveWarning, match='empty search'):
        sr.download()
    with pytest.warns(LightkurveWarning, match='empty search'):
        sr.download_all()


def test_open():
    # define paths to k2 and  tess data
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
    except (ValueError, IOError):
        pass
    # Can you instantiate with a path?
    assert(isinstance(KeplerTargetPixelFile(k2_path), KeplerTargetPixelFile))
    assert(isinstance(TessTargetPixelFile(tess_path), TessTargetPixelFile))
    # Can open take a quality_bitmask argument?
    assert(open(k2_path, quality_bitmask='hard').quality_bitmask == 'hard')


@pytest.mark.remote_data
def test_issue_472():
    """Regression test for https://github.com/KeplerGO/lightkurve/issues/472"""
    # The line below previously threw an exception because the target was not
    # observed in Sector 2; we're expecting an empty SearchResult instead.
    search = search_tesscut("TIC41336498", sector=2)
    assert isinstance(search, SearchResult)
    len(search) == 0


@pytest.mark.remote_data
def test_corrupt_download_handling():
    """When a corrupt file exists in the cache, make sure the user receives
    a helpful error message.

    This is a regression test for #511.
    """
    from builtins import open  # Because open is imported as lightkurve.open at the top
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Pretend a corrupt file exists at the expected cache location
        expected_dir = os.path.join(tmpdirname,
                                   "mastDownload",
                                   "Kepler",
                                   "kplr011904151_lc_Q111111110111011101")
        expected_fn = os.path.join(expected_dir, "kplr011904151-2010009091648_lpd-targ.fits.gz")
        os.makedirs(expected_dir)
        open(expected_fn, 'w').close()  # create "corrupt" i.e. empty file
        with pytest.raises(SearchError) as err:
            search_targetpixelfile("Kepler-10", quarter=4).download(download_dir=tmpdirname)
        assert "The file was likely only partially downloaded." in err.value.args[0]


def test_filenotfound():
    """Regression test for #540; ensure lk.open() yields `FileNotFoundError`."""
    # Python 2 uses IOError instead of FileNotFoundError;
    # the block below can be removed when we drop Python 2 support.
    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = IOError
    with pytest.raises(FileNotFoundError):
        open("DOESNOTEXIST")
