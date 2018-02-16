"""Test features of lightkurve that interact with the data archive at MAST.

Note: if you have the `pytest-remotedata` package installed, then tests flagged
with the `@pytest.mark.remote_data` decorator below will only run if the
`--remote-data` argument is passed to py.test.  This allows tests to pass
if no internet connection is available.
"""
import pytest

from ..mast import search_kepler_tpf_products, ArchiveError
from .. import KeplerTargetPixelFile


@pytest.mark.remote_data
def test_search_kepler_tpf_products():
    # EPIC 210634047 was observed twice in long cadence
    assert(len(search_kepler_tpf_products(210634047)) == 2)
    # ...including Campaign 4
    assert(len(search_kepler_tpf_products(210634047, campaign=4)) == 1)
    # KIC 11904151 (Kepler-10) was observed in LC in 15 Quarters
    assert(len(search_kepler_tpf_products(11904151)) == 15)
    # ...including quarter 11 but not 12:
    assert(len(search_kepler_tpf_products(11904151, quarter=11)) == 1)
    assert(len(search_kepler_tpf_products(11904151, quarter=12)) == 0)
    # We should also be able to resolve it by its name instead of KIC ID
    assert(len(search_kepler_tpf_products('Kepler-10')) == 15)


@pytest.mark.remote_data
def test_tpf_from_archive():
    # Request an object name that does not exist
    with pytest.raises(ArchiveError) as exc:
        KeplerTargetPixelFile.from_archive("InvalidTargetUnitTest")
    assert('not resolve' in str(exc))
    # Request an EPIC ID that was not observed
    with pytest.raises(ArchiveError) as exc:
        KeplerTargetPixelFile.from_archive(246000000)
    assert('not found' in str(exc))
    # Request a valid target that has multiple TPFs
    with pytest.raises(ArchiveError) as exc:
        KeplerTargetPixelFile.from_archive('Kepler-10')
    assert('multiple' in str(exc))
    # This should work!
    KeplerTargetPixelFile.from_archive('Kepler-10', quarter=11)
