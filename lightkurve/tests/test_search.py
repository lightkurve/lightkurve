from __future__ import division, print_function

import logging
import numpy as np
import pytest

from astropy.coordinates import SkyCoord
import astropy.units as u

from ..search import search_lcf, search_tpf, ArchiveError
from .. import log

@pytest.mark.remote_data
def test_search_tpf():
    # EPIC 210634047 was observed twice in long cadence
    assert(len(search_tpf(210634047).products) == 2)
    # ...including Campaign 4
    assert(len(search_tpf(210634047, campaign=4).products) == 1)
    # KIC 11904151 (Kepler-10) was observed in LC in 15 Quarters
    assert(len(search_tpf(11904151).products) == 15)
    # ...including quarter 11 but not 12:
    assert(len(search_tpf(11904151, quarter=11).targets) == 1)
    assert(len(search_tpf(11904151, quarter=12).products) == 0) ### NOTE fails for .targets
    # should work for all split campaigns
    campaigns = [[91, 92, 9], [101, 102, 10], [111, 112, 11]]
    ids = [200068780, 200071712, 202975993]
    for c, idx in zip(campaigns, ids):
        ca = search_tpf(idx, quarter=c[0]).products
        cb = search_tpf(idx, quarter=c[1]).products
        assert(len(ca) == 1)
        assert(len(ca) == len(cb))
        assert(~np.any(ca['description'] == cb['description']))
        # If you specify the whole campaign, both split parts must be returned.
        cc = search_tpf(idx, quarter=c[2]).products
        assert(len(cc) == 2)
    search_tpf(11904151, quarter=11).download()

@pytest.mark.remote_data
def test_search_lcf():
    # We should also be able to resolve it by its name instead of KIC ID
    assert(len(search_lcf('Kepler-10').products) == 15)
    # An invalid KIC/EPIC ID should be dealt with gracefully
    with pytest.raises(ArchiveError) as exc:
        search_lcf(-999)
    assert('Could not resolve' in str(exc))
    # If we ask for all cadence types, there should be four Kepler files given
    assert(len(search_lcf(4914423, quarter=6, cadence='any').products) == 4)
    # ...and only one should have long cadence
    assert(len(search_lcf(4914423, quarter=6, cadence='long').products) == 1)
    # Should be able to resolve an ra/dec
    assert(len(search_lcf('297.5835, 40.98339', quarter=6).products) == 1)
    # Should be able to resolve a SkyCoord
    c = SkyCoord('297.5835 40.98339', unit=(u.deg, u.deg))
    assert(len(search_lcf(c, quarter=6).products) == 1)
    search_lcf(c, quarter=6).download()

@pytest.mark.remote_data
def test_month():
    # In short cadence, if we specify both quarter and month it should work:
    search_tpf('Kepler-10', quarter=11, month=1, cadence='short')

@pytest.mark.remote_data
def test_collections():
    # TargetPixelFileCollection class
    assert(len(search_tpf(205998445, radius=900).download_all()) == 4)
    # LightCurveFileCollection class
    assert(len(search_lcf(205998445, radius=900).download_all()) == 4)
