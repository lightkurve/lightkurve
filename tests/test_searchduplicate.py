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
from astroquery.mast import Catalogs

from lightkurve.utils import LightkurveWarning, LightkurveError
from lightkurve import SearchDuplicate


@pytest.mark.remote_data
def test_SearchDuplicate():
    """ TIC 158324245 was classified as a SPLIT in TIC v8.2
    This means that it itself is not a real star, but composed of several other stars
    The brightest real star is called the DUPLICATE and replaces the MAST paramters of 
    the original. 

    There is then an additional faint star. 
    This procedure should return the ID's of all associated stars. 
    """
    catalog_data = Catalogs.query_object("TIC 158324245", radius=0.001, catalog="TIC", version=8.2)
    table = catalog_data["ID", "ra", "dec", "Tmag", "disposition", "duplicate_id"]

    assert table['ID'][0] == '158324245'
    assert table['ID'][1] == '1717079071'
    assert table['ID'][2] == '1717079066'

    assert table['disposition'][0] == 'SPLIT'
    assert table['disposition'][1] == 'DUPLICATE'
    
