import numpy as np
import pytest
import warnings

from ..nexsci import *
from .. import nexsci


def test_retrieval():
    ''' Test the data can be retrieved '''
    nexsci._retrieve_online_data()

def test_freshness():
    ''' Test the fresh data works '''
    # This function should always run...
    nexsci._check_data_is_fresh()

def test_dataframe():
    ''' Test the dataframe can be read '''
    df = get_nexsci_data()
    assert(len(df) >= 0)

def test_trappist():
    ''' Test the planet mask works '''
    t = np.arange(0, 100, 0.001)
    m = find_planet_mask('TRAPPIST-1', t)
    v = (1. - (m.sum()/float(len(t))))
    # Trappist 1 should have a planet transiting 7.9% of the time
    assert np.isclose(v, 0.079, atol=0.001)

def test_unknown():
    ''' Test the unknown planet mask works '''
    t = np.arange(0, 100, 0.001)
    m = create_planet_mask(t, 10, 0, 5)
    # This should have a planet transiting 75% of the time
    v = (1. - (m.sum()/float(len(t))))
    assert np.isclose(v, 0.75, atol=0.01)
