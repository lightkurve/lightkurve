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
def apply_propermotion(catalog: Table, equinox: Time, epoch: Time):
    """
    Function that returns an astropy table of sources with the proper motion applied

    Parameters:
    -----------
    catalog :
        astropy.table.Table which contains the coordinates of targets and proper motion values
    equinox: astropy.time.Time
        The R.A and Dec. vaulues taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely Julian date 2451545.0 TT.
    epoch : astropy.time.Time
        Time of the observation - This is taken from the catalog R.A and Dec. values and re-formatted as an astropy.time.Time object

    Output:
    ------
    catalog : astropy.table.Table
        Returns an astropy table with ID, corrected RA, corrected Dec, and Mag(?Some ppl might find this benifical for contamination reasons?)
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

    catalog["RAJ2000"] = c1.ra.to(u.deg).value
    catalog["DEJ2000"] = c1.dec.to(u.deg).value

    return catalog


def query_skycatalog(
    coord: SkyCoord,
    epoch: Time,
    catalog_name: str,
    radius: float = 20.0,
    magnitude_limit: float = 18.0,
    equinox: Time = Time(2451545.0, format="jd", scale="tt"),
):
    """Function that returns an astropy table of sources in the region of interest

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD and TT. Note that tess data is in btjd & tdb - so a user would have to specify in the Time object
        For example you could put in `Time(np.mean(lc.time.value), scale='tdb', format='btjd')`
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    equinox: astropy.time.Time
        The R.A and Dec. vaulues taken from the catalogs is in J2000.
        The J2000. 0 epoch is precisely Julian date 2451545.0 TT.

    Output:
    -------
    Returns an astropy.table of the sources within radius query
    """

    if not isinstance(coord, SkyCoord):
        raise ValueError("Must pass an `astropy.coordinates.SkyCoord object.")
    if not isinstance(epoch, Time):
        raise ValueError("Must pass an `astropy.time.Time object.")
    if not isinstance(equinox, Time):
        raise ValueError("Must pass an `astropy.time.Time object.")

    if catalog_name.lower() in ["kepler"]:
        catalog_name = _catalog_dict["kepler"]
        filters = Vizier(
            columns=["KIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "kepmag"],
            column_filters={"kepmag": f"<{magnitude_limit}"},
        )
        catalog = filters.query_region(
            coord, catalog=catalog_name, radius=Angle(radius, "arcsec")
        )[catalog_name]
        catalog.rename_columns(("KIC", "pmDE", "kepmag"), ("ID", "pmDEC", "Mag"))

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
        catalog.rename_columns(("TIC", "pmDE", "Tmag"), ("ID", "pmDEC", "Mag"))

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
        catalog.rename_columns(("DR3Name", "pmDE", "Gmag"), ("ID" "pmDEC", "Mag"))

        # apply_propermotion
        catalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)
    else:
        raise ValueError(
            "This is not a valid catalog name please refer to Lightkurve docs for a list of catalogs that can be used"
        )
    return catalog
