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
def apply_propermotion(catalog, equinox: str, epoch: float):
    """
    Function that returns an astropy table of sources with the proper motion applied

    Parameters:
    -----------
    coord :
        astropy.table which contains the coordinates of all targets and proper motion values
    equinox : float
        This is the date of the current ra and dec - J2000
    epoch : float
        Time of the observation - this needs to be a astropy.time format

    Output:
    ------
    Returna and astopy table with ID, corrected RA, corrected Dec, and mag

    """

    # Get the input data from the catalog
    ra_list = catalog["RAJ2000"]
    dec_list = catalog["DEJ2000"]
    pm_ra = catalog["pmRA"]
    pm_dec = catalog["pmDEC"]

    # Below is then code pulled in/adapted from #1332 by Veselin
    # _get_nearby_gaia_objects
    # _get_corrected_coordinates
    if (
        ra_list is None
        or dec_list is None
        or pm_ra is None
        or pm_dec is None
        or (np.all(pm_ra == 0) and np.all(pm_dec == 0))
        or equinox is None
    ):
        return ra_list, dec_list, False

    correction = SkyCoord(
        ra_list,
        dec_list,
        pm_ra_cosdec=pm_ra,
        pm_dec=pm_dec,
        frame="icrs",
        obstime=equinox,
    )

    pm_values = correction.apply_space_motion(new_obstime=epoch)

    corrected_ra = pm_values.ra.to(u.deg).value
    corrected_dec = pm_values.ra.to(u.deg).value

    # Now need to create a new astropy table with these values
    pm_table = Table()
    pm_table["ID"] = catalog["ID"]
    pm_table["RA_corrected"] = corrected_ra
    pm_table["DEC_corrected"] = corrected_dec
    pm_table["Mag"] = catalog["Mag"]

    return pm_table


# This function should just query the vizieR catalogs:
def query_skycatalog(
    coord: None,
    tpf_or_lc: None,
    catalog_name: str,
    arcsec_radius: float = 20.0,
    magnitude_limit: float = 18.0,
):
    """Function that returns an astropy table of sources in the region of interest ---

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
    Output:
    -------
    Returns an astropy.table of the sources in the TPF"""

    if coord == None:
        # Get the co-ordinates from the ra and dec of the tpf_or_lc
        coord = SkyCoord(tpf_or_lc.ra, tpf_or_lc.dec, unit=(u.deg, u.deg), frame="icrs")
    else:
        coord == coord

    # Coordinate frame equinox time - TIC, KIC, EPIC, and Gaia all use J2000
    # Since we are using these co-ordinates i dont think we need to take it from the header
    equinox = "J2000"
    # If we do need to get it from the header then use this
    # equinox = Time(self.EQUINOX, format='jyear', out_subfmt='jyear_str')

    # The epoch should be the time(s) of observation.
    # We can get this information from the header and then convert it into a year.
    epoch = Time(np.mean(tpf_or_lc.time.value), scale="tdb", format="btjd")

    if catalog_name.lower() in ["kepler"]:
        catalog_name = _catalog_dict["kepler"]
        filters = Vizier(
            columns=["KIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "kepmag"],
            column_filters={"kepmag": f"<{magnitude_limit}"},
        )
        catalog = filters.query_region(
            coord, catalog=catalog_name, radius=Angle(arcsec_radius, "arcsec")
        )
        catalog = catalog[catalog_name]
        catalog.rename_column("KIC", "ID")
        catalog.rename_column("pmDE", "pmDEC")
        catalog.rename_column("kepmag", "Mag")

        # apply_propermotion
        skycatalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)

    elif catalog_name.lower() in ["k2", "ktwo"]:
        catalog_name = _catalog_dict["k2"]
        filters = Vizier(
            columns=["ID", "RAJ2000", "DEJ2000", "pmRA", "pmDEC", "Kpmag"],
            column_filters={"Kpmag": f"<{magnitude_limit}"},
        )
        catalog = filters.query_object(
            coord, catalog=catalog_name, radius=Angle(arcsec_radius, "arcsec")
        )
        catalog = catalog[catalog_name]
        catalog.rename_column("Kpmag", "Mag")

        # apply_propermotion
        skycatalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)

    elif catalog_name.lower() in ["tess"]:
        catalog_name = _catalog_dict["tess"]

        filters = Vizier(
            columns=["TIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Tmag"],
            column_filters={"Tmag": f"<{magnitude_limit}"},
        )

        catalog = filters.query_region(
            coord, catalog=str(catalog_name), radius=Angle(arcsec_radius, "arcsec")
        )
        catalog = catalog[catalog_name]
        catalog.rename_column("TIC", "ID")
        catalog.rename_column("pmDE", "pmDEC")
        catalog.rename_column("Tmag", "Mag")

        # apply_propermotion
        skycatalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)

    elif catalog_name.lower() in ["gaia", "dr3"]:
        catalog_name = _catalog_dict["k2"]
        filters = Vizier(
            columns=["DR3Name", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Gmag"],
            column_filters={"Gmag": f"<{magnitude_limit}"},
        )
        catalog = filters.query_region(
            coord, catalog=catalog_name, radius=Angle(arcsec_radius, "arcsec")
        )
        catalog = catalog[catalog_name]
        catalog.rename_column("DR3Name", "ID")
        catalog.rename_column("pmDE", "pmDEC")
        catalog.rename_column("Gmag", "Mag")

        # apply_propermotion
        skycatalog = apply_propermotion(catalog, equinox=equinox, epoch=epoch)
    else:
        try:
            # This assumes  that the name of the catalog they entered is correct,
            # We cant filter this and just have to return everything.
            filters = Vizier()
            catalog = filters.query_object(
                coord, catalog=catalog_name, radius=Angle(arcsec_radius, "arcsec")
            )

            # Here we want to check the catalog for an RAJ2000, a DEJ200 and the proper motions
            # If these do not exist then we will have to stop.

        except:
            raise ValueError(
                "This is not a valid catalog name please refer to https://vizier.cds.unistra.fr/ for a list of catalogs that can be used"
            )
    return skycatalog
