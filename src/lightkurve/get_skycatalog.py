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
from astropy.table import Table


# This creates a dictionary of the relevant catalogs in VizieR
_catalog_dict = {
    "kepler": "V/133",
    "k2": "IV/34",
    "tess": "IV/39/tic82",
    "gaia": "I/355",
}


# This function should take the information from the skycatalog search and apply the proper motion
def apply_propermotion(catalog: Table, epoch: Time, equinox: Time = Time('2000')) -> Table:
    """
    Function that returns an astropy table of sources with the proper motion applied

    Parameters:
    -----------
    catalog :
        astropy.table.Table which contains the coordinates of targets and proper motion values
    epoch : astropy.time.Time
        Time of the observation - this needs to be an astropy.time.Time object
    equinox : astropy.time.Time
        This is the date of the catalog, assumed to be J2000

    Output:
    ------
    catalog : astropy.table.Table
        Returns an astropy table with ID, corrected RA, corrected Dec
    """

    # Get the input data from the catalog
    c = SkyCoord(
        catalog["RAJ2000"],
        catalog["DEJ2000"],
        pm_ra_cosdec=catalog["pmRA"],
        pm_dec=catalog["pmDEC"],
        frame="icrs",
        obstime=equinox,
    )

    c1 = c.apply_space_motion(new_obstime=epoch)


    catalog["RA"] = c1.ra.to(u.deg).value
    catalog["Dec"] = c1.dec.to(u.deg).value
#    catalog.drop("RAJ2000")
#    catalog.drop("DEJ2000")
    return catalog

def query_skycatalog(
    coord: SkyCoord,
    catalog_name: str,
    radius: float = 20.0,
    magnitude_limit: float = 18.0,
    epoch: Time = Time('2000'), 
    equinox: Time = Time('2000'), 
) -> Table:
    """Function that returns an astropy table of sources in the region of interest

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    radius : float
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    epoch: astropy.time.Time
    equinox: astropy.time.Time
    
    Output:
    -------
    Returns an astropy.table of the sources within radius query
    """

    if not isinstance(coord, SkyCoord):
        raise ValueError("Must pass an `astropy.coordinates.SkyCoord` object.")

    if catalog_name.lower() in ["kepler"]:
        catalog_name = _catalog_dict["kepler"]
        filters = Vizier(
            columns=["KIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "kepmag"],
            column_filters={"kepmag": f"<{magnitude_limit}"},
        )
        catalog = filters.query_region(
            coord, catalog=catalog_name, radius=Angle(radius, "arcsec")
        )[catalog_name]
        catalog.rename_columns(("KIC", 'pmDE', "kepmag"), ("ID", "pmDEC", "Mag"))

        # apply_propermotion
        catalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)

    elif catalog_name.lower() in ["k2", "ktwo"]:
        catalog_name = _catalog_dict["k2"]
        filters = Vizier(
            columns=["ID", "RAJ2000", "DEJ2000", "pmRA", "pmDEC", "Kpmag"],
            column_filters={"Kpmag": f"<{magnitude_limit}"},
        )
        catalog = filters.query_object(
            coord, catalog=catalog_name, radius=Angle(radius, "arcsec")
        )[catalog_name]
        catalog.rename_column("Kpmag", "Mag")

        # apply_propermotion
        catalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)

    elif catalog_name.lower() in ["tess"]:
        catalog_name = _catalog_dict["tess"]

        filters = Vizier(
            columns=["TIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Tmag"],
            column_filters={"Tmag": f"<{magnitude_limit}"},
        )

        catalog = filters.query_region(
            coord, catalog=str(catalog_name), radius=Angle(radius, "arcsec")
        )[catalog_name]

        catalog.rename_column("TIC", "ID")
        catalog.rename_column("pmDE", "pmDEC")
        catalog.rename_column("Tmag", "Mag")

        # apply_propermotion
        catalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)

    elif catalog_name.lower() in ["gaia", "dr3"]:
        catalog_name = _catalog_dict["gaia"]
        filters = Vizier(
            columns=["DR3Name", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Gmag"],
            column_filters={"Gmag": f"<{magnitude_limit}"},
        )
        catalog = filters.query_region(
            coord, catalog=catalog_name, radius=Angle(radius, "arcsec")
        )[catalog_name]
        catalog.rename_column("DR3Name", "ID")
        catalog.rename_column("pmDE", "pmDEC")
        catalog.rename_column("Gmag", "Mag")

        # apply_propermotion
        catalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)
    else:
        raise ValueError(
            "This is not a valid catalog name please refer to Lightkurve docs for a list of catalogs that can be used"
        )
    return catalog
