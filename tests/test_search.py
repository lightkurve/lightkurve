"""Test features of lightkurve that interact with the data archive at MAST.

Note: if you have the `pytest-remotedata` package installed, then tests flagged
with the `@pytest.mark.remote_data` decorator below will only run if the
`--remote-data` argument is passed to py.test.  This allows tests to pass
if no internet connection is available.
"""
import os
import pytest

from numpy.testing import assert_almost_equal, assert_array_equal
import tempfile
from requests import HTTPError

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

import lightkurve as lk

from lightkurve.utils import LightkurveWarning, LightkurveError
from lightkurve.search import (
    search_lightcurve,
    search_targetpixelfile,
    search_tesscut,
    SearchResult,
    SearchError,
    log,
)
from lightkurve import (
    KeplerTargetPixelFile,
    TessTargetPixelFile,
    TargetPixelFileCollection,
)

from .test_conf import use_custom_config_file, remove_custom_config


@pytest.mark.remote_data
def test_search_targetpixelfile():
    # EPIC 210634047 was observed twice in long cadence
    assert len(search_targetpixelfile("EPIC 210634047", mission="K2").table) == 2
    # ...including Campaign 4
    assert (
        len(search_targetpixelfile("EPIC 210634047", mission="K2", campaign=4).table)
        == 1
    )
    # KIC 11904151 (Kepler-10) was observed in LC in 15 Quarters
    assert (
        len(
            search_targetpixelfile(
                "KIC 11904151", mission="Kepler", cadence="long"
            ).table
        )
        == 15
    )
    # ...including quarter 11 but not 12:
    assert (
        len(
            search_targetpixelfile(
                "KIC 11904151", mission="Kepler", cadence="long", quarter=11
            ).unique_targets
        )
        == 1
    )
    assert (
        len(
            search_targetpixelfile(
                "KIC 11904151", mission="Kepler", cadence="long", quarter=12
            ).table
        )
        == 0
    )
    search_targetpixelfile("KIC 11904151", quarter=11, cadence="long").download()
    # with mission='TESS', it should return TESS observations
    tic = "TIC 273985862"  # Has been observed in multiple sectors including 1
    assert len(search_targetpixelfile(tic, mission="TESS").table) > 1
    assert (
        len(search_targetpixelfile(tic, author="SPOC", sector=1, radius=100).table)
        == 2
    )
    search_targetpixelfile(tic, author="SPOC", sector=1).download()
    assert len(search_targetpixelfile("pi Mensae", sector=1, author="SPOC").table) == 1
    # Issue #445: indexing with -1 should return the last index of the search result
    assert len(search_targetpixelfile("pi Men")[-1]) == 1


@pytest.mark.remote_data
def test_search_split_campaigns():
    """Searches should should work for split campaigns.

    K2 Campaigns 9, 10, and 11 were split into two halves for various technical
    reasons (C91=C9a, C92=C9b, C101=C10a, C102=C10b, C111=C11a, C112=C11b).
    We expect most targets from those campaigns to return two TPFs.
    """
    campaigns = [9, 10, 11]
    ids = ["EPIC 228162462", "EPIC 228726301", "EPIC 202975993"]
    for c, idx in zip(campaigns, ids):
        search = search_targetpixelfile(idx, campaign=c, cadence="long").table
        assert len(search) == 2


@pytest.mark.remote_data
def test_search_lightcurve(caplog):
    # We should also be able to resolve it by its name instead of KIC ID
    # The name Kepler-10 somehow no longer works on MAST. So we use 2MASS instead:
    #   https://simbad.cds.unistra.fr/simbad/sim-id?Ident=%405506010&Name=Kepler-10
    assert (
        len(search_lightcurve("2MASS J19024305+5014286", mission="Kepler", author="Kepler", cadence="long").table)
        == 15
    )
    # An invalid KIC/EPIC ID or target name should be dealt with gracefully
    search_lightcurve(-999)
    assert "disambiguate" in caplog.text
    search_lightcurve("DOES_NOT_EXIST (UNIT TEST)")
    assert "disambiguate" in caplog.text
    # If we ask for all cadence types, there should be four Kepler files given
    assert len(search_lightcurve("KIC 4914423", quarter=6, cadence="any", author="Kepler").table) == 4
    # ...and only one should have long cadence
    assert len(search_lightcurve("KIC 4914423", quarter=6, cadence="long", author="Kepler").table) == 1
    # Should be able to resolve an ra/dec
    assert len(search_lightcurve("297.5835, 40.98339", quarter=6, author="Kepler").table) == 1
    # Should be able to resolve a SkyCoord
    c = SkyCoord("297.5835 40.98339", unit=(u.deg, u.deg))
    search = search_lightcurve(c, quarter=6, author="Kepler")
    assert len(search.table) == 1
    assert len(search) == 1
    # We should be able to download a light curve
    search.download()
    # The second call to download should use the local cache
    caplog.clear()
    caplog.set_level("DEBUG")
    search.download()
    assert "found in local cache" in caplog.text
    # with mission='TESS', it should return TESS observations
    tic = "TIC 273985862"
    assert len(search_lightcurve(tic, mission="TESS").table) > 1
    assert (
        len(
            search_lightcurve(
                tic, mission="TESS", author="spoc", sector=1, radius=100
            ).table
        )
        == 2
    )
    search_lightcurve(tic, mission="TESS", author="SPOC", sector=1).download()
    assert len(search_lightcurve("pi Mensae", author="SPOC", sector=1).table) == 1


@pytest.mark.remote_data
def test_search_tesscut():
    # Cutout by target name
    assert len(search_tesscut("pi Mensae", sector=1).table) == 1
    assert len(search_tesscut("pi Mensae").table) > 1
    # Cutout by TIC ID
    assert len(search_tesscut("TIC 206669860", sector=28).table) == 1
    # Cutout by RA, dec string
    search_string = search_tesscut("30.578761, -83.210593")
    # Cutout by SkyCoord
    c = SkyCoord("30.578761 -83.210593", unit=(u.deg, u.deg))
    search_coords = search_tesscut(c)
    # These should be identical
    assert len(search_string.table) == len(search_coords.table)
    # The coordinates below are beyond the edge of the sector 4 (camera 1-4) FFI
    search_edge = search_tesscut("30.578761, 6.210593", sector=4)
    assert len(search_edge.table) == 0


@pytest.mark.remote_data
def test_search_tesscut_download(caplog):
    """Can we download TESS cutouts via `search_cutout().download()?"""
    try:
        ra, dec = 30.578761, -83.210593
        search_string = search_tesscut("{}, {}".format(ra, dec), sector=[1, 12])
        # Make sure they can be downloaded with default size
        tpf = search_string[1].download()
        # Ensure the correct object has been returned
        assert isinstance(tpf, TessTargetPixelFile)
        # Ensure default size is 5x5
        assert tpf.flux[0].shape == (5, 5)
        assert len(tpf.targetid) > 0  # Regression test #473
        assert tpf.sector == 12  # Regression test #696
        # Ensure the WCS is valid (#434 regression test)
        center_ra, center_dec = tpf.wcs.all_pix2world([[2.5, 2.5]], 1)[0]
        assert_almost_equal(ra, center_ra, decimal=1)
        assert_almost_equal(dec, center_dec, decimal=1)
        # Download with different dimensions
        tpfc = search_string.download_all(cutout_size=4, quality_bitmask="hard")
        assert isinstance(tpfc, TargetPixelFileCollection)
        assert tpfc[0].quality_bitmask == "hard"  # Regression test for #494
        assert tpfc[0].sector == 1  # Regression test #696
        assert tpfc[1].sector == 12  # Regression test #696
        # Ensure correct dimensions
        assert tpfc[0].flux[0].shape == (4, 4)
        # Download with rectangular dimennsions?
        rect_tpf = search_string[0].download(cutout_size=(3, 5))
        assert rect_tpf.flux[0].shape == (3, 5)
        # If we ask for the exact same cutout, do we get it from cache?
        caplog.clear()
        log.setLevel("DEBUG")
        tpf_cached = search_string[0].download(cutout_size=(3, 5))
        assert "Cached file found." in caplog.text
        # test #1063 - ensure when download_dir is specified, there is no error
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=".", prefix="temp_lk_cache_4test_") as download_dir:
            # ensure relative path works, the bug in #1063
            tpf_w_download_dir = search_string[0].download(cutout_size=(3, 5), download_dir=download_dir)
            assert tpf_w_download_dir.flux[0].shape == (3, 5)
            tpf_w_download_dir = None  # remove the tpf reference so that the underlying file can be deleted on Windows
    except HTTPError as exc:
        # TESSCut will occasionally return a "504 Gateway Timeout error" when
        # it is overloaded.  We don't want this to trigger a test failure.
        if "504" not in str(exc):
            raise exc


@pytest.mark.remote_data
def test_search_with_skycoord():
    """Can we pass both names, SkyCoord objects, and coordinate strings?"""
    sr_name = search_targetpixelfile("KIC 11904151", mission="Kepler", cadence="long")
    assert (
        len(sr_name) == 15
    )  # Kepler-10 as observed during 15 quarters in long cadence
    # Can we search using a SkyCoord objects?
    sr_skycoord = search_targetpixelfile(
        SkyCoord.from_name("KIC 11904151"), mission="Kepler", cadence="long"
    )
    assert_array_equal(
        sr_name.table["productFilename"], sr_skycoord.table["productFilename"]
    )
    # Can we search using a string of "ra dec" decimals?
    sr_decimal = search_targetpixelfile(
        "285.67942179 +50.24130576", mission="Kepler", cadence="long"
    )
    assert_array_equal(
        sr_name.table["productFilename"], sr_decimal.table["productFilename"]
    )
    # Can we search using a sexagesimal string?
    sr_sexagesimal = search_targetpixelfile(
        "19:02:43.1 +50:14:28.7", mission="Kepler", cadence="long"
    )
    assert_array_equal(
        sr_name.table["productFilename"], sr_sexagesimal.table["productFilename"]
    )
    # Can we search using the KIC ID?
    sr_kic = search_targetpixelfile("KIC 11904151", mission="Kepler", cadence="long")
    assert_array_equal(
        sr_name.table["productFilename"], sr_kic.table["productFilename"]
    )


@pytest.mark.remote_data
def test_searchresult():
    sr = search_lightcurve("KIC 11904151", mission="Kepler")
    assert len(sr) == len(sr.table)  # Tests SearchResult.__len__
    assert len(sr[2:7]) == 5  # Tests SearchResult.__get__
    assert len(sr[2]) == 1
    assert "kplr" in sr.__repr__()
    assert "kplr" in sr._repr_html_()


@pytest.mark.remote_data
def test_month():
    # In short cadence, if we specify both quarter and month
    sr = search_targetpixelfile("KIC 11904151", quarter=11, month=1, cadence="short")
    assert len(sr) == 1
    sr = search_targetpixelfile("KIC 11904151", quarter=11, month=[1, 3], cadence="short")
    assert len(sr) == 2


@pytest.mark.remote_data
def test_collections():
    # TargetPixelFileCollection class
    assert (
        len(search_targetpixelfile("EPIC 205998445", mission="K2", radius=900).table)
        == 4
    )
    # LightCurveFileCollection class with set targetlimit
    assert (
        len(
            search_lightcurve(
                "EPIC 205998445", mission="K2", radius=900, limit=3, author="K2"
            ).download_all()
        )
        == 3
    )
    # if fewer targets are found than targetlimit, should still download all available
    assert (
        len(
            search_targetpixelfile(
                "EPIC 205998445", mission="K2", radius=900, limit=6
            ).table
        )
        == 4
    )
    # if download() is used when multiple files are available, should only download 1
    with pytest.warns(LightkurveWarning, match="4 files available to download"):
        assert isinstance(
            search_targetpixelfile(
                "EPIC 205998445", mission="K2", radius=900, author="K2"
            ).download(),
            KeplerTargetPixelFile,
        )


@pytest.mark.remote_data
def test_properties():
    c = SkyCoord("297.5835 40.98339", unit=(u.deg, u.deg))
    assert_almost_equal(search_targetpixelfile(c, quarter=6).ra, 297.5835)
    assert_almost_equal(search_targetpixelfile(c, quarter=6).dec, 40.98339)
    assert len(search_targetpixelfile(c, quarter=6).target_name) == 1
    assert len(search_targetpixelfile(c, quarter=6).obsid) == 1


@pytest.mark.remote_data
def test_source_confusion():
    # Regression test for issue #148.
    # When obtaining the TPF for target 6507433, @benmontet noticed that
    # a target 4 arcsec away was returned instead.
    # See https://github.com/lightkurve/lightkurve/issues/148
    desired_target = "KIC 6507433"
    tpf = search_targetpixelfile(desired_target, quarter=8).download()
    assert tpf.targetid == 6507433


def test_empty_searchresult():
    """Does an empty SearchResult behave gracefully?"""
    sr = SearchResult(Table())
    assert len(sr) == 0
    str(sr)
    with pytest.warns(LightkurveWarning, match="empty search"):
        sr.download()
    with pytest.warns(LightkurveWarning, match="empty search"):
        sr.download_all()


@pytest.mark.remote_data
def test_issue_472():
    """Regression test for https://github.com/lightkurve/lightkurve/issues/472"""
    # The line below previously threw an exception because the target was not
    # observed in Sector 2; we're always expecting a SearchResult object (empty
    # or not) rather than an exception.
    # Whether or not this SearchResult is empty has changed over the years,
    # because the target is only ~15 pixels beyond the FFI edge and the accuracy
    # of the FFI footprint polygons at the MAST portal have changed at times.
    search = search_tesscut("TIC41336498", sector=2)
    assert isinstance(search, SearchResult)


@pytest.mark.remote_data
def test_corrupt_download_handling_case_empty():
    """When a corrupt file exists in the cache, make sure the user receives
    a helpful error message.

    This is a regression test for #511 and #1184.

    For case the file is truncated, see test_read.py::test_file_corrupted
    It cannot be done easily here because on Windows,
    a similar test would result in PermissionError when `tempfile`
    tries to do cleanup.
    Some low level codes (probably astropy.fits) still hold a file handle
    of the corrupted FIS file.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Pretend a corrupt file exists at the expected cache location
        expected_dir = os.path.join(
            tmpdirname, "mastDownload", "Kepler", "kplr011904151_lc_Q111111110111011101"
        )
        expected_fn = os.path.join(
            expected_dir, "kplr011904151-2010009091648_lpd-targ.fits.gz"
        )
        os.makedirs(expected_dir)
        open(expected_fn, "w").close()  # create "corrupt" i.e. empty file
        with pytest.raises(LightkurveError) as err:
            search_targetpixelfile("KIC 11904151", quarter=4, cadence="long").download(
                download_dir=tmpdirname
            )
        assert "may be corrupt" in err.value.args[0]
        assert expected_fn in err.value.args[0]


@pytest.mark.remote_data
def test_mast_http_error_handling(monkeypatch):
    """Regression test for #1211; ensure downloads yields an error when MAST download result in an error."""
    from astroquery.mast import Observations

    result = search_lightcurve("TIC 273985862", mission="TESS")
    remote_url = result.table[0]["dataURI"]

    def mock_http_error_response(*args, **kwargs):
        """Mock the `download_product()` response to simulate MAST returns HTTP error"""
        return Table(data={
            "Local Path": ["./mastDownload/acme_lc.fits"],
            "Status": ["ERROR"],
            "Message": ["HTTP Error 500: Internal Server Error"],
            "URL": [remote_url],
            })

    monkeypatch.setattr(Observations, "download_products", mock_http_error_response)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # ensure the we don't hit cache so that it'll always download from MAST
        with pytest.raises(LightkurveError) as excinfo:
            result[0].download(download_dir=tmpdirname)
        assert "HTTP Error 500" in str(excinfo.value)
        assert remote_url in str(excinfo.value)


@pytest.mark.remote_data
def test_indexerror_631():
    """Regression test for #631; avoid IndexError."""
    # This previously triggered an exception:
    result = search_lightcurve("KIC 8462852", sector=15, radius=1, author="spoc")
    assert len(result) == 1


@pytest.mark.skip(
    reason="TODO: issue re-appeared on 2020-01-11; needs to be revisited."
)
@pytest.mark.remote_data
def test_name_resolving_regression_764():
    """Due to a bug, MAST resolved "EPIC250105131" to a different position than
    "EPIC 250105131". This regression test helps us verify that the bug does
    not re-appear. Details: https://github.com/lightkurve/lightkurve/issues/764
    """
    from astroquery.mast import MastClass

    c1 = MastClass().resolve_object(objectname="EPIC250105131")
    c2 = MastClass().resolve_object(objectname="EPIC 250105131")
    assert c1.separation(c2).to("arcsec").value < 0.1


@pytest.mark.remote_data
def test_overlapping_targets_718():
    """Regression test for #718."""
    # Searching for the following targets without radius should only return
    # the requested targets, not their overlapping neighbors.
    targets = ["KIC 5112705", "KIC 10058374", "KIC 5385723"]
    for target in targets:
        search = search_lightcurve(target, quarter=11, author="Kepler")
        assert len(search) == 1
        assert search.target_name[0] == f"kplr{target[4:].zfill(9)}"

    # When using `radius=1` we should also retrieve the overlapping targets
    search = search_lightcurve("KIC 5112705", quarter=11, author="Kepler", radius=1 * u.arcsec)
    assert len(search) > 1

    # Searching by `target_name` should not preven a KIC identifier to work
    # in a TESS data search
    search = search_targetpixelfile(
        "KIC 8462852", mission="TESS", sector=15, author="spoc"
    )
    assert len(search) == 1


@pytest.mark.remote_data
def test_tesscut_795():
    """Regression test for #795: make sure the __repr__.of a TESSCut
    SearchResult works."""
    str(search_tesscut("KIC 8462852"))  # This raised a KeyError


@pytest.mark.remote_data
def test_download_flux_column():
    """Can we pass reader keyword arguments to the download method?"""
    lc = search_lightcurve("Pi Men", author="SPOC", sector=12).download(
        flux_column="sap_flux"
    )
    assert_array_equal(lc.flux, lc.sap_flux)


@pytest.mark.remote_data
def test_exptime_filtering():
    """Can we pass "fast", "short", exposure time to the cadence argument?"""
    # Try `cadence="fast"`
    res = search_lightcurve("AU Mic", sector=27, cadence="fast")
    assert len(res) == 1
    assert res.exptime[0].value == 20
    # Try `cadence="short"`
    res = search_lightcurve("AU Mic", sector=27, cadence="short")
    assert len(res) == 1
    assert res.table["t_exptime"][0] == 120
    # Try `cadence=20`
    res = search_lightcurve("AU Mic", sector=27, cadence=20)
    assert len(res) == 1
    assert res.table["t_exptime"][0] == 20
    assert "fast" in res.table["productFilename"][0]

    # Now do the same with the new exptime argument,
    # because `cadence` may be deprecated.
    # Try `exptime="fast"`
    res = search_lightcurve("AU Mic", sector=27, exptime="fast")
    assert len(res) == 1
    assert res.exptime[0].value == 20
    # Try `exptime="SHoRt"` -- mixed lower/uppercase is on purpose
    res = search_lightcurve("AU Mic", sector=27, exptime="SHoRt")
    assert len(res) == 1
    assert res.table["t_exptime"][0] == 120
    # Try `exptime=20`
    res = search_lightcurve("AU Mic", sector=27, exptime=20)
    assert len(res) == 1
    assert res.table["t_exptime"][0] == 20
    assert "fast" in res.table["productFilename"][0]


@pytest.mark.remote_data
def test_search_slicing_regression():
    # Regression test: slicing after calling __repr__ failed.
    res = search_lightcurve("AU Mic", exptime=20)
    res.__repr__()
    res[res.exptime.value < 100]


@pytest.mark.remote_data
def test_ffi_hlsp():
    """Can SPOC, QLP (FFI), and TESS-SPOC (FFI) light curves be accessed?"""
    search = search_lightcurve(
        "TrES-2b", mission="tess", author="any", sector=26
    )  # aka TOI 2140.01
    assert "QLP" in search.table["author"]
    assert "TESS-SPOC" in search.table["author"]
    assert "SPOC" in search.table["author"]
    # tess-spoc also products tpfs
    search = search_targetpixelfile("TrES-2b", mission="tess", author="any", sector=26)
    assert "TESS-SPOC" in search.table["author"]
    assert "SPOC" in search.table["author"]


@pytest.mark.remote_data
def test_qlp_ffi_lightcurve():
    """Can we search and download an MIT QLP FFI light curve?"""
    search = search_lightcurve("TrES-2b", sector=26, author="qlp")
    assert len(search) == 1
    assert search.author[0] == "QLP"
    assert search.exptime[0] == 30 * u.minute  # Sector 26 had 30-minute FFIs
    lc = search.download()
    all(lc.flux == lc.kspsap_flux)


@pytest.mark.remote_data
def test_spoc_ffi_lightcurve():
    """Can we search and download a SPOC FFI light curve?"""
    search = search_lightcurve("TrES-2b", sector=26, author="tess-spoc")
    assert len(search) == 1
    assert search.author[0] == "TESS-SPOC"
    assert search.exptime[0] == 30 * u.minute  # Sector 26 had 30-minute FFIs
    lc = search.download()
    all(lc.flux == lc.pdcsap_flux)


@pytest.mark.remote_data
def test_split_k2_campaigns():
    """Do split K2 campaign sections appear separately in search results?"""
    # Campaign 9
    search_c09 = search_targetpixelfile("EPIC 228162462", cadence="long", campaign=9)
    assert search_c09.table["mission"][0] == "K2 Campaign 09a"
    assert search_c09.table["mission"][1] == "K2 Campaign 09b"
    # Campaign 10
    search_c10 = search_targetpixelfile("EPIC 228725972", cadence="long", campaign=10)
    assert search_c10.table["mission"][0] == "K2 Campaign 10a"
    assert search_c10.table["mission"][1] == "K2 Campaign 10b"
    # Campaign 11
    search_c11 = search_targetpixelfile("EPIC 203830112", cadence="long", campaign=11)
    assert search_c11.table["mission"][0] == "K2 Campaign 11a"
    assert search_c11.table["mission"][1] == "K2 Campaign 11b"


@pytest.mark.remote_data
def test_customize_search_result_display():
    search = search_lightcurve("TIC390021728")
    # default display does not have proposal id
    assert 'proposal_id' not in search.__repr__()

    # custom config: has proposal_id in display
    try:
        use_custom_config_file("data/lightkurve_sr_cols_added.cfg")
        # Note: here a *different* TIC is used for search to avoid the complication
        # of caching.
        # if the same TIC is used, the cached result would be returned, without
        # consiering the customization specified.
        # the TIC used is in multiple sectors, with some rows having proposal_id and some rows
        # have none. So it's also a sanity test the for the actual proposal_id display logic.
        search = search_lightcurve("TIC298734307")
        assert 'proposal_id' in search.__repr__()
    finally:
        remove_custom_config()  # restore default to avoid side effects

    # test changing config at runtime
    try:
        lk.conf.search_result_display_extra_columns = ['sequence_number']

        search = search_lightcurve("TIC169175503")  # again use a different TIC to avoid caching complication
        assert 'sequence_number' in search.__repr__()
    finally:
        lk.conf.search_result_display_extra_columns = []  # restore default to avoid side effects

    # Test per-object customization
    search.display_extra_columns = []
    assert 'proposal_id' not in search.__repr__()
    search.display_extra_columns = ['sequence_number', 'proposal_id']  # also support multiple columns
    assert 'proposal_id' in search.__repr__()
    assert 'sequence_number' in search.__repr__()


@pytest.mark.remote_data
def test_customize_search_result_display_case_nonexistent_column():

    # Ensure that if an extra column specified are not in search result
    # the extra column will not be shown (and it does not generate error)
    #
    # One typical case is that some columns are in the result of
    # search_lightcurve() / search_targetpixelfile(), but not in those of search_tesscut()

    search = search_lightcurve("TIC390021728")
    search.display_extra_columns = ['foo_col']
    assert 'foo_col' not in search.__repr__()
