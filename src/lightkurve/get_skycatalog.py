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
                   epoch: float,
                   vizeR_cols: list = None):
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

    
    
    catalog_result = vizeR_cols.query_object(coord, catalog, Angle(radius, "arcsec"))[0]

    #This will need to be updates for each heading as cab be different for each one
    ra_list = catalog_result["RAJ2000"]
    dec_list = catalog_result["DEJ2000"]
    pm_ra = catalog_result["pmRA"]
    pm_dec = catalog_result["pmDEC"]

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
                     obstime=equinox)
    
            
        new_c = c.apply_space_motion(new_obstime=epoch)
        
        new_ra.append(new_c.ra.to(u.deg).value)
        new_dec.append(new_c.dec.to(u.deg).value)
    
    #Will need to convert this into astropy to make lk compatable
    df = mag_lim.to_pandas()
    
    # Just return RADec corrected as ra and dec
    df.insert(3, 'ra_corrected', new_ra, True)
    df.insert(4, 'dec_corrected', new_dec, True)

    # Table should have ra, dec, t/k/k2mag, 
        
    return df



_catalog_dict = {'kepler':'V/133', 'k2':'IV/34', 'tess', 'gaia'}

def query_skycatalog(catalog_name, magnitude_limit=18, epoch=2000, equinox=Reasonable Date, arcsec_radius=20):
        """Returns an astropy.table of the sources in the TPF"""
        epoch = self.time.jd.mean()
        equinox=Time(self.EQUINOX, format="decimalyear", scale="tt") + 0.5

        if catalog_name.lower() in ['kepler']:
            catalog_name = _catalog_dict['kepler']
            catalog = Vizier(columns=['KIC','RAJ2000', 'DEJ2000','pmRA','pmDE', 'kepmag'],
              column_filters={"kepmag":f"<{magnitude_limit}"}, catalog=catalog_name)
            catalog.rename({'kepmag':'mag', 'KIC':'ID', 'pmDE':'pmDEC'})
            # apply_propermotion
            skycatalog =  apply_propermotion(catalog, equinox=equinox, epoch=epoch)
        elif catalog_name.lower() in ['k2', 'ktwo']:
            catalog_name = _catalog_dict['kepler']
            catalog = Vizier(columns=['ID','RAJ2000', 'DEJ2000','pmRA','pmDEC', "Kpmag"],
              column_filters={"Kpmag":"f"<{magnitude_limit}""}, catalog=catalog_name)
            catalog.rename({'Kpmag':'mag'})
            # apply_propermotion
            skycatalog =  apply_propermotion(catalog, equinox=equinox, epoch=epoch)
        elif catalog_name.lower() in ['tess']:
            catalog = 'IV/39/tic82'
            vizeR_cols = Vizier(columns=['TIC', 'RAJ2000', 'DEJ2000','pmRA','pmDE', 'Tmag'],
           column_filters={"Tmag":"<18"}, 
           keywords=["TESS"])
            skycatalog =  get_skycatalog(SkyCoord(self.ra, self.dec,  unit=(u.deg, u.deg),frame='icrs'), radius=arcsec_radius, catalog=catalog, equinox=equinox, epoch=epoch)
        elif catalog_name.lower() in ['gaia', 'dr3']:
            #raise ValueError("Cannot parse `mission` attribute")
            #Changed this to look at the Gaia DR3 catalog
            catalog = 'I/355'
            vizeR_cols = Vizier(columns=['DR3Name','RAJ2000','DEJ2000','pmRA','pmDE','Gmag'],
                column_filters={"Gmag":"<18"}, 
                keywords=["Gaia"])
            skycatalog =  get_skycatalog(SkyCoord(self.ra, self.dec,  unit=(u.deg, u.deg),frame='icrs'), radius=arcsec_radius, catalog=catalog, equinox=equinox, epoch=epoch)
        else:
             try:
                # Assuming they've put in a real catalog on vizier, you won't be able to rename anything, you won't be able to assume columns, so give them everything
            except:
                raise ValueError("") # tell them you don't understand the catalog
        return skycatalog  
