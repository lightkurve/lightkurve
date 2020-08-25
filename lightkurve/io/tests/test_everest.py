import pytest

from ... import search_lightcurve


@pytest.mark.remote_data
def test_search_everest():
    """Can we search and download an EVEREST light curve?"""
    search = search_lightcurve("GJ 9827", author="EVEREST", campaign=12)
    assert len(search) == 1
    assert search.table["author"][0] == "EVEREST"
    lc = search.download()
    assert type(lc).__name__ == "KeplerLightCurve"
    assert lc.campaign == 12
