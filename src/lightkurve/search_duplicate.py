"""Defines tools to see if your object of interest is an ARTIFACT, SPLIT, or JOIN"""
from __future__ import division
import os
import glob
import logging
import re
import warnings

import numpy as np
from astropy.table import join, Table, Row
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy import units as u
from astropy.utils import deprecated
from astropy.time import Time
from astroquery.mast import Catalogs

log = logging.getLogger(__name__)

__all__ = ["search_duplicate"]


def search_duplicate(name=None, ra=None, dec=None, tic=None):
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
    >>> lk.SearchDuplicate(tic=1716106614)
    
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
