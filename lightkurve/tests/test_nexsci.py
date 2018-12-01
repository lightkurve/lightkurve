import numpy as np
import pytest
import warnings
import os

from .. import PACKAGEDIR
from ..nexsci import *
from .. import nexsci
from ..utils import LightkurveWarning

@pytest.mark.remote_data
def test_retrieval():
    ''' Test the data can be retrieved '''
    nexsci._retrieve_online_data()
    # Get rid of it for next test
    os.remove('{}/data/planets.csv'.format(PACKAGEDIR))

@pytest.mark.remote_data
def test_freshness():
    ''' Test the fresh data works '''
    # This function should always run...

    # Will redownload the data
    nexsci._check_data_is_fresh()

    #Make it not fresh
    fname = '{}/data/planets.csv'.format(PACKAGEDIR)
    mtime = os.stat(fname).st_mtime
    os.utime(fname, (mtime - 86400*10, mtime - 86400*10))

    # Will redownload the data
    with pytest.warns(LightkurveWarning, match='NExScI Database out of date'):
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

def test_a_bad_planet():
    ''' Test the planet mask works '''
    t = np.arange(0, 100, 0.001)
    with pytest.raises(ValueError) as exc:
        m = find_planet_mask('Christina', t)
    with pytest.raises(ValueError) as exc:
        m = find_planet_mask('TRAPPIST-1', 1)
    with pytest.raises(ValueError) as exc:
        m = find_planet_mask('TRAPPIST-1', t, 1)
    with pytest.raises(ValueError) as exc:
        m = find_planet_mask('HD 209458', t, [1])   


def test_unknown():
    ''' Test the unknown planet mask works '''
    t = np.arange(0, 100, 0.001)
    m = create_planet_mask(t, 10, 0, 5)
    # This should have a planet transiting 75% of the time
    v = (1. - (m.sum()/float(len(t))))
    assert np.isclose(v, 0.75, atol=0.01)
