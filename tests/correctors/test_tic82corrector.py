"""Unit tests for the `TIC82Corrector` class."""
import pytest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from astroquery.mast import Catalogs

import lightkurve as lk
from lightkurve.correctors import TIC82Corrector
from lightkurve.correctors import search_tic


@pytest.mark.remote_data
def test_search_tic():
    """ TIC 158324245 was classified as a SPLIT in TIC v8.2
    This means that it itself is not a real star, but composed of several other stars
    The brightest real star is called the DUPLICATE and replaces the MAST paramters of 
    the original. 

    There is then an additional faint star. 
    This procedure should return the ID's of all associated stars. 
    """

    table = search_tic(tic="158324245")

    assert table['ID'][0] == '158324245'
    assert table['ID'][1] == '1717079071'
    assert table['ID'][2] == '1717079066'

    assert table['disposition'][0] == 'SPLIT'
    assert table['disposition'][1] == 'DUPLICATE'


def test_TIC82Corrector_priors():
    """This test will check that correction to the PDCSAP flux is being calculated
    corectly
    """

    #Example of a join
    catalog_data = Catalogs.query_object("TIC 408200371", radius=0.001, catalog="TIC", version=8.2)
    table_join = catalog_data["ID", "ra", "dec", "Tmag", "disposition", "duplicate_id"]
    #This table should contain information for two objects

    #For their object of interest they will need to download the light curve
    #You might have more than one sector and so will have to do each one at a time
    lcf = lk.search_lightcurve('TIC 408200371', mission='TESS', author="SPOC", sector=14).download()

    #Get the mag of your object from the lcf
    mag_obj = lcf.meta['TESSMAG']
     
    #Get the mag of the duplicate
    pos = np.where(table_join['disposition']=="DUPLICATE")
    mag_dup = table_join['Tmag'][pos][0]

    #Calculate the fluxes
    F2m = 10**(4-0.4*mag_obj)
    FGi = 10**(4-0.4*mag_dup)

    assert_almost_equal(F2m,2.5562305448313114)
    assert_almost_equal(FGi,2.4139040312021063)

    #Sum up flux values of all except split
    comb_flux = F2m + FGi
    assert_almost_equal(comb_flux,4.970134576033418)

    #Calculate the flux ratio
    ratio = F2m/comb_flux
    assert_almost_equal(ratio,0.514318175036499)

    flux_new = lcf.flux.value / ratio
    flux_err_new =  lcf.flux_err.value / ratio
    check1 = flux_new[1]

    assert_almost_equal(int(check1),26118)

    #Now lets use the actual corrector
    join = TIC82Corrector(table_join, lcf)
    corrected_lc = join.correct()

    check2 = corrected_lc.flux.value[1]

    assert_almost_equal(int(check2),int(check1))
     

    #Example of a SPLIT
    catalog_data2 = Catalogs.query_object("TIC 158324245", radius=0.001, catalog="TIC", version=8.2)
    table_split = catalog_data2["ID", "ra", "dec", "Tmag", "disposition", "duplicate_id"]
    #This table should contain information for three objects

    #For their object of interest they will need to download the light curve
    #You might have more than one sector and so will have to do each one at a time
    lcf_split = lk.search_lightcurve('TIC 158324245', mission='TESS', author="SPOC", sector=26).download()
    print(lcf_split.flux.value[0])
    
    #Get the mag of your object from the lcf
    mag_obj_split = lcf_split.meta['TESSMAG']
    #Get position of the SPLIT in the array
    pos_split = np.where(table_split['disposition']=="SPLIT")
    
    #Get the mag of the duplicate
    pos_dup = np.where(table_split['disposition']=="DUPLICATE")
    mag_dup2 = table_split['Tmag'][pos_dup][0]
    
    
    #Get the location of the faint objects in the array
    t = np.arange(0,len(table_split['disposition']),1,dtype=int)
    pos_faint = np.delete(t,[pos_split[0],pos_dup[0]])
    mag_faint = table_split['Tmag'][pos_faint][0]

    #Calculate the fluxes
    Fs = 10**(4-0.4*mag_obj_split)
    Fd = 10**(4-0.4*mag_dup2)
    Ff = 10**(4-0.4*mag_faint)

    assert_almost_equal(np.round(Fs,3),np.round(1.490046576557392,3))
    assert_almost_equal(np.round(Fd,3),np.round(0.8086486316572605,3))
    assert_almost_equal(np.round(Ff,3),np.round(0.6396170440617517,3))
    
    #Calculate the flux ratio
    ratio2 = ((Fs+Fd+Ff)/Fs)*(Fd/(Fd+Ff))
    
    flux_new2 = lcf_split.flux.value * ratio2
    flux_err_new2 =  lcf_split.flux_err.value * ratio2
    check3 = flux_new2[1]

    assert_almost_equal(int(check3),int(12377.408))
    
    #Now lets use the actual corrector
    split = TIC82Corrector(table_split, lcf_split)
    corrected_lc2 = split.correct()
    check4 = corrected_lc2.flux.value[1]

    assert_almost_equal(int(check4),int(check3))
     
