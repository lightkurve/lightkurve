"""
THIS IS A WORK IN PROGRESS

This tool will provide the user with a list of stars/objects within a defined region around a requested target.

The returned list contains the proper motion corrected R.A and Dec. postions of the objects in addition to several other parameters.

This tool can be applied to the following TESS products
- LightCurve Objects
- TargetPixelFile Objects
- TESSCut Objects

Eventual use case will be:
mytpf.get_skycatalog()
Returning a table


This file will likely be merged into the search.py code and structure 
It is placed here as an initial working document 
"""

import sys, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles

import astropy.units as u
from astropy.time import Time

from astroquery.mast import Observations
from astroquery.mast import Catalogs

import lightkurve as lk
from lightkurve import *

def get_skycatalog(tpf_or_lc, 
                search_radius_deg=None, 
                lim_mag=None, 
                catalog=None):
    
    #Need to add a check to make sure file type is correct 

    #Need to make this a box search
    #If a TPF supplied make the box search slightly larger than the size of the TPF
     
    
    
    #Get the time that the data was collected
    new_time = tpf_or_lc.time[0]
    #Get the R.A., Dec., and the equinox 
    ra = tpf_or_lc.ra
    dec = tpf_or_lc.dec
    #Format it so that it works with the catalog query
    cords = str(ra)+" "+str(dec)
    #Get the equinox from the header
    equinox = tpf_or_lc.meta["EQUINOX"]
    
    
    #This code is specific for a TESS object a TPF, TESSCut or a LC - TPF.starcat_all(rad,lim_mag) and it return the answer
    #We could modify it in the future so that you could just put in the object R.A and Dec.

    
    #Setting defaults for search radius, limiting mag, and catalog

    #We should set a default value if a radius is specified
    #If it is a TPF and the radius is not specified then get this info based on the size plus x amount of pixels
    
    if search_radius_deg==None:
        srd=0.1
    else:
        srd = search_radius_deg
        
    if lim_mag == None:
        lm = 18
    else: 
        lm = lim_mag

    #Want the catalog to have either the TIC or KIC inputs as default
    #Need to use tpf.mission to determine which mission it is from
    #Then set default
    #Problem is that there is no KIC option in catalog query - we could try something else?
    
    if catalog == None:
        ct = "TIC"
    else:
        ct = catalog
        
    #Allowed catalog values - want to add a check here
   
    cv = ["HSC","Galex","Gaia","TIC","CTL","Panstarrs", "DiskDetective", "Plato"]
    
    catalogTIC = Catalogs.query_object(str(cords), radius=srd, catalog=ct)
    mag_cut = np.where(catalogTIC["Tmag"]<float(lm))
    mag_lim = catalogTIC[mag_cut]
    
    ra_list = mag_lim["ra"]
    dec_list = mag_lim["dec"]
    pm_ra = mag_lim["pmRA"]
    pm_dec = mag_lim["pmDEC"]

    pm_unit = u.milliarcsecond / u.year
    
    new_ra = []
    new_dec = []

    #Below is then code pulled in/adapted from #1332 by Veselin
    #_get_nearby_gaia_objects
    #_get_corrected_coordinates
    for a in range(len(ra_list)):
    
        if ra is None or dec is None or pm_ra is None or pm_dec is None or (np.all(pm_ra == 0) and np.all(pm_dec == 0)) or equinox is None:
            return ra, dec, False

        c = SkyCoord(ra_list[a] * u.deg, 
                     dec_list[a] * u.deg, 
                     pm_ra_cosdec=pm_ra[a] * pm_unit, 
                     pm_dec=pm_dec[a] * pm_unit,
                     frame='icrs',
                     obstime=Time(equinox, format="decimalyear", scale="tt") + 0.5)
    
            
        new_c = c.apply_space_motion(new_obstime=new_time)
        
        new_ra.append(new_c.ra.to(u.deg).value)
        new_dec.append(new_c.dec.to(u.deg).value)
    
    #Will need to convert this into astropy to make lk compatable
    df = mag_lim.to_pandas()
    
    df.insert(3, 'ra_corrected', new_ra, True)
    df.insert(4, 'dec_corrected', new_dec, True)
        
    return df

