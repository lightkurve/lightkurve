from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from astropy.coordinates import SkyCoord
import astropy.units as u

from ..search import search_lightcurvefile, search_targetpixelfile, ArchiveError
from ..targetpixelfile import KeplerTargetPixelFile


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
        ca = search_targetpixelfile(idx, quarter=c[0]).table
        cb = search_targetpixelfile(idx, quarter=c[1]).table
        assert(len(ca) == 1)
        assert(len(ca) == len(cb))
        assert(~np.any(ca['description'] == cb['description']))
        # If you specify the whole campaign, both split parts must be returned.
        cc = search_targetpixelfile(idx, quarter=c[2]).table
        assert(len(cc) == 2)
    search_targetpixelfile(11904151, quarter=11).download()


@pytest.mark.remote_data
def test_search_lightcurvefile():
    # We should also be able to resolve it by its name instead of KIC ID
    assert(len(search_lightcurvefile('Kepler-10').table) == 15)
    # An invalid KIC/EPIC ID should be dealt with gracefully
    with pytest.raises(ArchiveError) as exc:
        search_lightcurvefile(-999)
    assert('Could not resolve' in str(exc))
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


@pytest.mark.remote_data
def test_search_with_skycoord():
    """Can we pass both names, SkyCoord objects, and coordinate strings?"""
    sr_name = search_targetpixelfile("Kepler-10")
    assert len(sr_name) == 15  # Kepler-10 as observed during 15 quarters in long cadence
    # Can we search using a SkyCoord objects?
    sr_skycoord = search_targetpixelfile(SkyCoord.from_name("Kepler_10"))
    assert_array_equal(sr_name.table['productFilename'], sr_skycoord.table['productFilename'])
    # Can we search using a string of "ra dec" decimals?
    sr_decimal = search_targetpixelfile("285.6794217927134 +50.2413057664939")
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
    assert(isinstance(search_targetpixelfile(205998445, radius=900).download(), KeplerTargetPixelFile))


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
