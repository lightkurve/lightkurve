import re
from typing import Tuple, Union


import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table, MaskedColumn

import numpy as np

import requests

from .interact_sky_provider import ProperMotionCorrectionMeta, InteractSkyCatalogProvider


def _parse_limit_mag_uncertainty_band(text):
    if text is np.ma.masked:
        return dict(l="", mag=np.nan, u="", band="")

    # handle text such as 12.9 V
    matches = re.match(r"\s*(?P<l>[><]?)(?P<mag>-?\d+([.]\d+)?)(?P<u>:?)\s*(?P<band>[^\s]*)", text)
    if matches is None:
        # parse unexpectedly failed, put the entire text in band for now
        # (as it's meant to be str)
        return dict(l="", mag=np.nan, u="", band=text)

    # converting to float should always work, given the regex
    res = matches.groupdict()
    res["mag"] = float(res["mag"])
    return res


def _parse_limit_mag_amp_uncertainty_band(text):
    if text is np.ma.masked:
        return dict(l="", mag=np.nan, u="", band="", a="")

    no_amp_res = _parse_limit_mag_uncertainty_band(text)  # limit flag has already been extracted
    if not np.isnan(no_amp_res["mag"]):
        no_amp_res["a"] = ""
        return no_amp_res

    # in addition to mag band, also handle text such as (0.05) V
    matches = re.match(r"\s*(?P<l>[><]?)[(](?P<mag>-?\d+([.]\d+)?)(?P<u>:?)[)]\s*(?P<band>[^\s]*)", text)
    if matches is None:
        # parse unexpectedly failed, put the entire text in band for now
        # (as it's meant to be str)
        return dict(l="", mag=np.nan, u="", band=text, a="")
    with_amp_res = matches.groupdict()
    with_amp_res["a"] = "Y"
    return with_amp_res


def _parse_number_with_uncertainty_flag(text):
    if text is np.ma.masked:
        return np.nan, ""
    matches = re.match(r"\s*(-?\d+([.]\d+)?)(:?)\s*", text)
    if matches is None:
        # parse unexpectedly failed, put the entire text in uncertain flag for now
        # (as it's meant to be str)
        return np.nan, ""
    # in Vizier, the uncertain flag is character `:`
    return float(matches[1]), matches[matches.lastindex]


def _do_remote_query(query_url):
    resp = requests.get(query_url)
    resp.raise_for_status()
    return resp.json()


def _parse_response(result):
    result = result.get("VSXObjects")
    if result is None:
        return None
    result = result.get("VSXObject")
    if result is None:
        return None  # TODO: create an minimal empty table

    tab = Table(rows=result)

    # use the column names in Vizier.
    # Note: pmRA and pmDE are not in Vizier, but the names are common
    # column names in Vizier
    # TODO: handle cases sometimes columns such as SpectralType does not exist
    tab.rename_columns(
        [
            "RA2000",
            "Declination2000",
            "ProperMotionRA",
            "ProperMotionDec",
            "VariabilityType",
            "SpectralType",
        ],
        [
            "RAJ2000",
            "DEJ2000",
            "pmRA",
            "pmDE",
            "Type",
            "Sp",
        ],
    )

    for c in ["RAJ2000", "DEJ2000"]:
        if c in tab.colnames:
            tab[c] = tab[c].astype(float)
            tab[c].unit = u.deg

    for c in ["pmRA", "pmDE"]:
        if c in tab.colnames:
            if isinstance(tab[c], MaskedColumn):
                tab[c].fill_value = np.nan
            tab[c] = tab[c].astype(float)
            tab[c].unit = u.mas / u.year

    for c in ["Period", "Epoch"]:
        if c in tab.colnames:
            val_uncertain_flag = [_parse_number_with_uncertainty_flag(v) for v in tab[c]]
            tab[c] = [t[0] for t in val_uncertain_flag]
            tab[c].unit = u.day
            tab[f"u_{c}"] = [t[1] for t in val_uncertain_flag]

    tab["OID"] = tab["OID"].astype(int)

    # the same at both: OID, Name, Epoch

    # MaxMag: parse to Vizier's l_max, max, n_max
    if "MaxMag" in tab.colnames:
        max_parsed = [_parse_limit_mag_uncertainty_band(v) for v in tab["MaxMag"]]
        tab["l_max"] = [t["l"] for t in max_parsed]
        tab["max"] = [t["mag"] for t in max_parsed]
        tab["max"].unit = u.mag
        tab["u_max"] = [t["u"] for t in max_parsed]
        tab["n_max"] = [t["band"] for t in max_parsed]

    # MinMag: parse to Vizier's min, n_min, and f_min ("Y" for amplitude)
    if "MinMag" in tab.colnames:
        min_parsed = [_parse_limit_mag_amp_uncertainty_band(v) for v in tab["MinMag"]]
        tab["l_min"] = [t["l"] for t in min_parsed]
        tab["min"] = [t["mag"] for t in min_parsed]
        tab["f_min"] = [t["a"] for t in min_parsed]  # amplitude, Y or ""
        tab["u_min"] = [t["u"] for t in min_parsed]
        tab["n_min"] = [t["band"] for t in min_parsed]

    # columns not in Vizier:
    # - AUID, , Constellation, Discoverer
    # -  Category (Variable, Suspected)
    # - RiseDuration, Discoverer
    return tab


# VSX API documentation: https://www.aavso.org/apis-aavso-resources#:~:text=The%20VSX%20API%20is%20not%20%22official%22%20yet
# See also VSX in Vizier : https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=B/vsx/vsx
def _query_cone_region(ra2000, dec2000, radius_deg, magnitude_limit=None):
    """Perform cone search against live VSX catalog.
    The result is formatted to be compatible with Vizier in VSX (`B/vsx/vsx`) to the extent possible.
    """

    query_url = f"https://www.aavso.org/vsx/index.php?view=api.list&ra={ra2000}&dec={dec2000}&radius={radius_deg}&format=json"
    if magnitude_limit is None:
        query_url += f"&tomag={magnitude_limit}"

    result = _do_remote_query(query_url)
    return _parse_response(result)


class VSXInteractSkyCatalogProvider(InteractSkyCatalogProvider):
    pass
