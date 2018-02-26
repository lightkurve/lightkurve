"""Functions which wrap `astroquery.mast` to obtain Kepler/K2 data from MAST."""

from __future__ import division, print_function

import numpy as np
from astroquery.mast import Observations
from astroquery.exceptions import ResolverError


class ArchiveError(Exception):
    """Raised if there is a problem accessing data."""
    pass


def download_products(products):
    """Download data products from MAST.

    Downloads will be cached using astroquery.mast's caching system.

    Parameters
    ----------
    products : astropy.Table
        A table detailing the products to be downloaded, i.e. the output
        of `search_kepler_products` or `search_kepler_tpf_products`.

    Returns
    -------
    paths : astropy.Table.Column
        Detailing the paths of the downloaded or cached products.
    """
    # Note: by default, MAST will only let us download "Minimum Recommended
    # Products" (MRPs), which do not include e.g. Target Pixel Files.
    # We need to set `mrp=False` to ensure MAST downloads whatever we want.
    dl = Observations.download_products(products, mrp_only=False)
    return dl['Local Path']


def search_kepler_products(target):
    """Returns a table of Kepler/K2 pipeline products for a given target.

    Raises an ArchiveError if no products are found.

    Parameters
    ----------
    target : str or int
        If the value is an integer in a specific range, we'll assume it is
        a KIC or EPIC ID.

    Returns
    -------
    products : astropy.Table
        Table detailing the available data products.
    """
    try:
        # If `target` looks like a KIC or EPIC ID, we will pass the exact
        # `target_name` under which MAST will know the object.
        target = int(target)
        if (target > 0) and (target < 200000000):
            target_name = 'kplr{:09d}'.format(target)
        elif (target > 200000000) and (target < 300000000):
            target_name = 'ktwo{:09d}'.format(target)
        else:
            raise ValueError("Not in the Kepler KIC or EPIC ID range")
        obs = Observations.query_criteria(target_name=target_name,
                                          project=["Kepler", "K2"])
    except ValueError:
        # If `target` did not look like a KIC or EPIC ID, then we let MAST
        # resolve the target name to a sky position.
        try:
            obs = Observations.query_criteria(objectname=target,
                                              radius='1 arcsec',
                                              project=["Kepler", "K2"])
        except ResolverError as exc:
            raise ArchiveError(exc)
    return Observations.get_product_list(obs)


def search_kepler_tpf_products(target, cadence='long', quarter=None,
                               campaign=None):
    """Returns a table of Kepler or K2 Target Pixel Files for a given target.

    Parameters
    ----------
    cadence: 'short' or 'long'
        Specify short (1-min) or long (30-min) cadence data.
    quarter, campaign : int
        Specify the Kepler Quarter or K2 Campaign Number.
        If None, then return the products for all Quarters/Campaigns.

    Returns
    -------
    products : astropy.Table
        Table detailing the available Target Pixel File products.
    """
    products = search_kepler_products(target)
    # Because MAST doesn't let us query based on Kepler-specific meta data
    # fields, we need to identify short/long-cadence TPFs by their filename.
    if cadence in ['short', 'sc']:
        suffix = "spd-targ"
    else:
        suffix = "lpd-targ"
    mask = np.array([suffix in fn for fn in products['productFilename']])
    # Identify the campaign or quarter by the description.
    quarter_or_campaign = campaign if campaign is not None else quarter
    if quarter_or_campaign is not None:
        mask &= np.array([desc.endswith('Q{}'.format(quarter_or_campaign)) or
                          desc.endswith('C{:02d}'.format(quarter_or_campaign))
                          for desc in products['description']])
    return products[mask]


def search_kepler_lightcurve_products(target, cadence='long', quarter=None,
                                      campaign=None):
    """Returns a table of Kepler or K2 lightcurve files for a given target.

    This only returns products produced by the official Kepler pipeline,
    which is not necessarily the best choice for every use case.

    Parameters
    ----------
    cadence: 'short' or 'long'
        Specify short (1-min) or long (30-min) cadence data.
    quarter, campaign : int
        Specify the Kepler Quarter or K2 Campaign Number.
        If None, then return the products for all Quarters/Campaigns.

    Returns
    -------
    products : astropy.Table
        Table detailing the available Target Pixel File products.
    """
    products = search_kepler_products(target)
    # Because MAST doesn't let us query based on Kepler-specific meta data
    # fields, we need to identify short/long-cadence TPFs by their filename.
    if cadence in ['short', 'sc']:
        suffix = "_slc.fits"
    else:  # long cadence
        suffix = "_llc.fits"
    mask = np.array([suffix in fn for fn in products['productFilename']])
    # Identify the campaign or quarter by the description.
    quarter_or_campaign = campaign if campaign is not None else quarter
    if quarter_or_campaign is not None:
        mask &= np.array([desc.endswith('Q{}'.format(quarter_or_campaign)) or
                          desc.endswith('C{:02d}'.format(quarter_or_campaign))
                          for desc in products['description']])
    return products[mask]
