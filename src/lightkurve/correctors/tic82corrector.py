"""Defines TIC82Corrector
"""
import logging
import warnings

import numpy as np
from astropy.table import join, Table, Row
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy import units as u
from astropy.utils import deprecated
from astropy.time import Time
from astroquery.mast import Catalogs
import matplotlib
import lightkurve as lk

from .corrector import Corrector

from ..lightcurve import LightCurve, MPLSTYLE

__all__ = ['search_tic',
           'TIC82Corrector']

log = logging.getLogger(__name__)

def search_tic(name=None, ra=None, dec=None, tic=None):
    """Here the user can input the name, TIC-ID, or RA and Dec of an object and query MAST to 
    determine if it part of the <1% which have been been affected by the update of the TIC to
    version 8.2. In this version the Gaia catalog was used instead of the 2MASS. This has led 
    to the discovery that some stars are actually artificats, some have duplicate entries and 
    are in fact one star, and some were actually made up of multiple stars.
    See https://outerspace.stsci.edu/display/TESS/TIC+v8+and+CTL+v8.xx+Data+Release+Notes
    for more details on this issue 

    If an object has been affected by the TIC update then this code will inform the user what
    kind of issue it has via displaying the MAST "disposition" column. 

    Input needed from a user will be the object of interest name, TIC-ID, or co-ordinates.

    ----------
    Example
    >>> import lightkurve as lk
    >>> from lightkurve.correctors import search_tic
    >>> table = search_tic(tic=394177355)
    
    """
    if name!=None:
        catalog_data = Catalogs.query_object(name, radius=0.001, catalog="TIC", version=8.2)
        output = catalog_data["ID", "ra", "dec", "Tmag", "disposition", "duplicate_id"]

    if ra and dec !=None:
        cords = str(ra)+" "+str(dec)
        catalog_data = Catalogs.query_object(cords, radius=0.001, catalog="TIC", version=8.2)
        output = catalog_data["ID", "ra", "dec", "Tmag", "disposition", "duplicate_id"]

    if tic !=None:
        tname ="TIC "+str(tic)
        catalog_data = Catalogs.query_object(tname, radius=0.001, catalog="TIC", version=8.2)
        output = catalog_data["ID", "ra", "dec", "Tmag", "disposition", "duplicate_id"]

    return output



class TIC82Corrector(Corrector):
    """The TIC input catalog is a list of ~1.7X10^9 sources on the sky and is used to provide stellar 
    parameters for evaluation. The catalogue is designed to include stars much fainter than likely TESS targets. 
    The catalog is therefore vast and as such certain spurious entries can occur, which must be considered.
 
    To ensure completeness and reliability the TIC is maintained and updated with new information from 
    astronomical surveys. TIC-7 was based on 2MASS data and was updated with information from Gaia DS2 to form 
    TIC-8.  
 
    Upon the incorporation of this new data several problems were found with a small number of objects (<1%). 
    Several Gaia objects were not found in the 2MASS data, and other objects once thought to be individual stars 
    are actually multiple objects.  
 
    To resolve these issues all Gaia objects not found in 2MASS, or earlier versions of the TIC, were given new 
    TIC-IDs. The more problematic objects were identified, and their nature documented under an assigned 
    “Disposition”column in the catalogue. The kinds of problematic objects are defined below:

    - Artifacts: These are spurious sources from 2MASS generally caused by diffraction spikes around bright stars. 
	- For newly flagged ARTIFACT sources, target pixel files are produced, but no light curves or data 
          validation pipeline products.
	- Beware! The search for artifacts around target stars was limited to stars brighter than 13 Tmag.
 
    - Join: Two TIC objects of near-equal brightness are in fact the same one real star. They originate from 
      slight mismatches between different catalogs, usually a 2MASS star that failed to be matched with its 
      respective Gaia DR2 entry.
	- The 2MASS object retains its TIC-ID for backward compatibility.
	- The stellar parameters of the 2MASS TIC-ID are updated to the improved Gaia-derived values (unless it is 
          from a specially curated list, in which case these values are used).
	- The Gaia TIC-ID is given the “Disposition” argument of DUPLICATE, indicating it is not a real star.
	- If using Lightkurve the SPOC flux of your 2MASS TIC object will be wrong before Sector 38. For data 
          before sector 38 you can adjust the SAP flux using the parent Tmag, as provided on MAST, and the SPLIT 
          mag in the object header. This correction however is only an approximation. 
	- Note that such objects are not labeled as JOIN in the disposition column on MAST.
 
    - Split: A bright 2MASS object that is not real, but the combination (sum of flux) of two or more fainter 
      objects, which have been found by Gaia. 
	- The 2MASS object remains in the TIC with the original TIC-ID
		- The magnitudes and stellar parameters are updated with those of the brightest real object 
                  identified by Gaia.
		- The star is marked as SPLIT in the “Disposition” column.
		- The 2MASS JHK magnitude will be incorrect.
	- Two or more new sources (the real sources) are added to the TIC with new TIC IDs. 
		- In these cases, the new TIC objects will be marked as DUPLICATES and have the TIC ID of the
                  original object in the “duplicate_id” column.
	- There is no way to tell what TESS signal came from a given star in the unresolved system.
	- Note that the old 2MASS magnitude isn’t available in TIC v8.2, as such the user will have to obtain 
          the magnitude from the meta data, and then use a sum of all mags to correct the flux for all sectors 
          before 38.

    In this notebook we will fix the flux for any object identified as a split or join.
    Exaples
    ------
    You will first need to run your obect of interest through search_tic and from that you will get a table
    which will identify the kind of issue you are working with.

    >>> import lightkurve as lk
    >>> from lightkurve.correctors import TIC82Corrector, search_tic
    >>> table = search_tic(tic=394177355)
    >>> SPLIT= lk.search_lightcurve("TIC 394177355", mission="TESS", author="SPOC", sector=14)
    >>> SPLIT_lc = SPLIT[0].download()
    >>> corrected_split_lc = TIC82Corrector(table, SPLIT_lc)"""
    
    def __init__(self, table, lc):
        self.table = table
        self.ids = table['ID']
        self.ra = table['ra']
        self.dec = table['dec']
        self.mag = table['Tmag']
        self.disp = table['disposition']
        self.dupid = table['duplicate_id']
        self.lc = lc
        self.lc.mag = lc.meta['TESSMAG']
        self.lc.flux = lc.flux.value
        self.lc.flux_err = lc.flux_err.value
        self.lc.time = lc.time
        
    def correct(self):

        #From the table obtained via SearchDuplicate determine what kind of object you have
        if 'SPLIT' in self.disp:
            #This means the object is a split and so will need to be corrected
            
            dp = np.where(self.disp=="DUPLICATE")
            sp = np.where(self.disp=="SPLIT")

            t = np.arange(0,len(self.disp),1,dtype=int)
            fp = np.delete(t,[sp[0],dp[0]])

            #Get the duplicate mag
            mag_dup = self.mag[dp]

            #Get the mag of the faint sources - there might be multiple
            mag_faint= self.mag[fp]

            #Convert things to flux
            Fs = 10**(4-0.4*self.lc.mag) #The split
            Fd = 10**(4-0.4*mag_dup) #The duplicate

            Ff = []
            for b in range(len(mag_faint)):
                Ff.append(10**(4-0.4*mag_faint[b])) #The faint

            #Sum up flux values of all except split
            sumfaint = sum(Ff)

            #Calculate the flux ratio
            ratio = ((sumfaint + Fd + Fs)/Fs) * (Fd/(Fd+sumfaint))

            flux_new = self.lc.flux * ratio
            flux_err_new =  self.lc.flux_err * ratio

            #Make a copy of the original so we can make sure units are correct
            corrected_lc = self.lc.copy()
            corrected_lc.flux = flux_new
            corrected_lc.flux_err = flux_err_new

            return corrected_lc

        elif 'DUPLICATE' in self.disp:
            
            #This means it is a JOIN case.
            dp = np.where(self.disp=="DUPLICATE")
            
            #Get the duplicate mag
            mag_Gaia = self.mag[dp]

            #Get the mag of the 2mass sources
            mag_2mass = self.lc.mag
           
            #Convert things to flux
            F2m = 10**(4-0.4*self.lc.mag)
            FGi = 10**(4-0.4*mag_Gaia)

            #Sum up flux values of all except split
            comb_flux = F2m + FGi

            #Calculate the flux ratio
            ratio = F2m/comb_flux

            flux_new = self.lc.flux / ratio
            flux_err_new =  self.lc.flux_err / ratio

            #Make a copy of the original so we can make sure units are correct
            corrected_lc = self.lc.copy()
            corrected_lc.flux = flux_new
            corrected_lc.flux_err = flux_err_new
        
            return corrected_lc
            

    def diagnose(self):
        return None

    
