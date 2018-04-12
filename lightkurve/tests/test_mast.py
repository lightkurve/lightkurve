"""Test features of lightkurve that interact with the data archive at MAST.

Note: if you have the `pytest-remotedata` package installed, then tests flagged
with the `@pytest.mark.remote_data` decorator below will only run if the
`--remote-data` argument is passed to py.test.  This allows tests to pass
if no internet connection is available.
"""
from __future__ import division, print_function

import pytest

from ..mast import (search_kepler_tpf_products, search_kepler_lightcurve_products,
                    ArchiveError)
from .. import KeplerTargetPixelFile, KeplerLightCurveFile


@pytest.mark.remote_data
def test_search_kepler_tpf_products():
    """Tests `lightkurve.mast.search_kepler_tpf_products`."""
    # EPIC 210634047 was observed twice in long cadence
    assert(len(search_kepler_tpf_products(210634047)) == 2)
    # ...including Campaign 4
    assert(len(search_kepler_tpf_products(210634047, campaign=4)) == 1)
    # KIC 11904151 (Kepler-10) was observed in LC in 15 Quarters
    assert(len(search_kepler_tpf_products(11904151)) == 15)
    # ...including quarter 11 but not 12:
    assert(len(search_kepler_tpf_products(11904151, quarter=11)) == 1)
    assert(len(search_kepler_tpf_products(11904151, quarter=12)) == 0)
    # should work for 91/92
    assert(len(search_kepler_tpf_products(200068780, quarter=91)) == 1)
    assert(len(search_kepler_tpf_products(200068780, quarter=92)) == 1)
    assert(len(search_kepler_tpf_products(200071712, quarter=102)) == 1)
    # We should also be able to resolve it by its name instead of KIC ID
    assert(len(search_kepler_tpf_products('Kepler-10')) == 15)
    # An invalid KIC/EPIC ID should be dealt with gracefully
    with pytest.raises(ArchiveError) as exc:
        search_kepler_tpf_products(-999)
    assert('Could not resolve' in str(exc))


@pytest.mark.remote_data
def test_search_kepler_lightcurve_products():
    """Tests `lightkurve.mast.search_kepler_lightcurve_products`."""
    assert(len(search_kepler_lightcurve_products('Kepler-10')) == 15)
    assert(len(search_kepler_lightcurve_products(200071712, quarter=102)) == 1)

@pytest.mark.remote_data
@pytest.mark.filterwarnings('ignore:Query returned no results')
def test_kepler_tpf_from_archive():
    # Request an object name that does not exist
    with pytest.raises(ArchiveError) as exc:
        KeplerTargetPixelFile.from_archive("LightKurve_Unit_Test_Invalid_Target")
    assert('not resolve' in str(exc))
    # Request an EPIC ID that was not observed
    with pytest.raises(ArchiveError) as exc:
        KeplerTargetPixelFile.from_archive(246000000)
    assert('No Target Pixel File found' in str(exc))
    # Request a valid target that has multiple TPFs
    with pytest.raises(ArchiveError) as exc:
        KeplerTargetPixelFile.from_archive('Kepler-10')
    assert('Please specify quarter' in str(exc))
    # But, if we specify the quarter for Kepler-10 it should work:
    KeplerTargetPixelFile.from_archive('Kepler-10', quarter=11)
    # However, for short cadence there is one file per month in Kepler
    with pytest.raises(ArchiveError) as exc:
        KeplerTargetPixelFile.from_archive('Kepler-10', quarter=11, cadence='short')
    assert('month' in str(exc))
    # In short cadence, if we specify both quarter and month it should work:
    KeplerTargetPixelFile.from_archive('Kepler-10', quarter=11, month=1, cadence='short')


@pytest.mark.remote_data
@pytest.mark.filterwarnings('ignore:Query returned no results')
def test_kepler_lightcurve_from_archive():
    # Request an object name that does not exist
    with pytest.raises(ArchiveError) as exc:
        KeplerLightCurveFile.from_archive("LightKurve_Unit_Test_Invalid_Target")
    assert('not resolve' in str(exc))
    # Request an EPIC ID that was not observed
    with pytest.raises(ArchiveError) as exc:
        KeplerLightCurveFile.from_archive(246000000)
    assert('No lightcurve file found' in str(exc))
    # Request a valid target that has multiple TPFs
    with pytest.raises(ArchiveError) as exc:
        KeplerLightCurveFile.from_archive('Kepler-10')
    assert('Please specify quarter' in str(exc))
    # But, if we specify the quarter for Kepler-10 it should work:
    KeplerLightCurveFile.from_archive('Kepler-10', quarter=11)
    # However, for short cadence there is one file per month in Kepler
    with pytest.raises(ArchiveError) as exc:
        KeplerLightCurveFile.from_archive('Kepler-10', quarter=11, cadence='short')
    assert('month' in str(exc))
    # In short cadence, if we specify both quarter and month it should work:
    KeplerLightCurveFile.from_archive('Kepler-10', quarter=11, month=1, cadence='short')
