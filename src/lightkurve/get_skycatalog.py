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

import numpy as np
import astropy.units as u
from astropy.time import Time
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

#We are going to use Vizer instead of the mast query
from astroquery.vizier import Vizier

def get_skycatalog(coord: SkyCoord, 
                   radius: float, 
                   magnitude_limit: float=18., 
                   catalog: str,
                   equinox: float, 
                   columns: list = None):
    """
    Function that returns an astropy table of sources that meet the input criteria. Queries ---

    Parameters:
    -----------
        coord : astropy.coordinates.SkyCoord
            Coordinates around which to do a radius query
        radius : float
            Radius in arcseconds to query
        magnitude_limit : float
            A value to limit the results in based on Gaia Gmag. Default, 18.
        catalog: str
            The catalog to query, either 'KIC', 'EPIC', or 'TIC', 'GaiaDR3'

    """
    #Need to add a check to make sure file type is correct 
    #Need to make this a box search
        # Could do, or could cut it down I don't think it's a faster query
     
    
    #This code is specific for a TESS object a TPF, TESSCut or a LC - TPF.starcat_all(rad,lim_mag) and it return the answer
    #We could modify it in the future so that you could just put in the object R.A and Dec.

    
    #Setting defaults for search radius, limiting mag, and catalog

    #We should set a default value if a radius is specified
    #If it is a TPF and the radius is not specified then get this info based on the size plus x amount of pixels
    
    #Want the catalog to have either the TIC or KIC inputs as default
    #Need to use tpf.mission to determine which mission it is from
    #Then set default
    #Problem is that there is no KIC option in catalog query - we could try something else?
       
    #Allowed catalog values - want to add a check here

    
    
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
    
    # Just return RADec corrected as ra and dec
    df.insert(3, 'ra_corrected', new_ra, True)
    df.insert(4, 'dec_corrected', new_dec, True)

    # Table should have ra, dec, t/k/k2mag, 
        
    return df

