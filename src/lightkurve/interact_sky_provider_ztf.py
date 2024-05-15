from typing import Union

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroquery.ipac.irsa import Irsa

import numpy as np


def _to_lc_url(oid, data_release, format):
    # see: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html
    if isinstance(oid, (int, np.int64, str)):
        return f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID={oid}&COLLECTION=ztf_dr{data_release}&FORMAT={format}"
    else:
        return [_to_lc_url(a_oid, data_release, format) for a_oid in oid]


def query_catalog(
    coord: SkyCoord,
    radius: Union[float, u.Quantity],
    magnitude_limit: float = 18.0,
    ngoodobsrel_min: int = 100,
    filtercode: str = None,
    data_release: int = 20,
) -> Table:
    catalog = f"ztf_objects_dr{data_release}"
    # OPEN: consider memoize the result, as astroquery v0.47 does not support caching for Irsa
    rs = Irsa.query_region(
        coordinates=coord,
        catalog=catalog,
        spatial="Cone",
        radius=radius)

    if magnitude_limit is not None:
        # column medianmag better reflects observed mag, but it could be 0
        rs = rs[(rs["medianmag"] < magnitude_limit) & (rs["medianmag"] != 0.0)]
    if ngoodobsrel_min is not None:
        rs = rs[rs["ngoodobsrel"] >= ngoodobsrel_min]
    if filtercode is not None:
        rs = rs[rs["filtercode"] == filtercode]

    # add URL of the lightcurve data (use csv format)
    rs["url"] = _to_lc_url(rs["oid"], data_release, "csv")

    # use standardized names for the required columns (across different catalogs)
    #
    # RA / DEC : coordinate in degree, proper motion corrected if possible
    rs.rename_column("ra", "RA")
    rs.rename_column("dec", "DEC")
    # magForSize: the magnitude used for sizing the circles in the plots
    rs["magForSize"] = rs["medianmag"]

    return rs


def add_to_data_source(result: Table, source: dict):
    more_data = dict()
    for col in [  # the additional columns to be included in the data source
        "oid",
        "filtercode",
        "ngoodobsrel",   # num. of good observations in the release
        "refmag", "refmagerr", "medianmag", "medmagerr", "maxmag", "minmag",
        "astrometricrms",  # Root Mean Squared deviation in epochal positions relative to object RA,Dec
        "url",
    ]:
        more_data[col] = result[col]
    source.update(more_data)


def get_tooltips() -> list:
    return [
        ("ZTF oid", "@oid"),
        ("Separation (\")", "@separation{0.00}"),
        ("filter", "@filtercode"),
        ("num. good obs.", "@ngoodobsrel"),
        ("median mag", "@medianmag"),
        ("RA", "@ra{0,0.00000000}"),
        ("DEC", "@dec{0,0.00000000}"),
        ("column", "@x{0.0}"),
        ("row", "@y{0.0}"),
        ]


def get_detail_view(data: dict) -> dict:
    return {
        "ZTF OID": f"""{data['oid']} (<a href="{data['url']}" target="_blank">LC</a>)""",
        'Separation (")': f"{data['separation']:.2f}",
        "filter": data['filtercode'],
        "num. good obs.": data['ngoodobsrel'],
        "Reference Mag": f"{data['refmag']:.3f} <span class='error'>{data['refmagerr']:.3f}<span>",
        "Median Mag": f"{data['medianmag']:.3f} <span class='error'>{data['medmagerr']:.3f}<span>",
        "Max Mag": f"{data['maxmag']:.3f}",
        "Min Mag": f"{data['minmag']:.3f}",
        "RA": f"{data['ra']:.8f}",
        "DEC": f"{data['dec']:.8f}",
        'astrometric RMS (")': f"{data['astrometricrms'] * 3600:.4f}",
        "column": f"{data['x']:.1f}",
        "row": f"{data['y']:.1f}",
    }
