import logging
import re
from typing import Tuple, Union


import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table, MaskedColumn
from astropy.time import Time

import numpy as np

import requests

from .. import LightkurveError
from .core import ProperMotionCorrectionMeta, InteractSkyCatalogProvider

log = logging.getLogger(__name__)


def _parse_limit_mag_uncertainty_band(text):
    # to handle case input is number (i.e., the origin astropy column is of type float)
    text = str(text)
    if text == "--" or text == "nan":  # string  representation of np.ma.masked and nan respectively
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
    # to handle case input is number (i.e., the origin astropy column is of type float)
    text = str(text)
    if text == "--" or text == "nan":  # string  representation of np.ma.masked and nan respectively
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
    # to handle case input is number (i.e., the origin astropy column is of type float)
    text = str(text)
    if text == "--" or text == "nan":  # string  representation of np.ma.masked and nan respectively
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
    if not isinstance(result, dict):
        # handle no result case (which is an empty list)
        return None
    result = result.get("VSXObject")
    if result is None:
        return None

    tab = Table(rows=result)

    # ensure it has the needed columns (the JSON result from VSX API would skip columns with no data)
    for c in ["RA2000", "Declination2000", "ProperMotionRA", "ProperMotionDec", "Epoch", "Period"]:
        if c not in tab.colnames:
            tab[c] = np.nan
    for c in ["Name", "OID", "VariabilityType", "SpectralType", "MaxMag", "MinMag"]:
        if c not in tab.colnames:
            tab[c] = ""

    # use the column names in Vizier.
    # Note: pmRA and pmDE are not in Vizier, but the names are common
    # column names in Vizier
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
        tab[c] = tab[c].astype(float)
        tab[c].unit = u.deg

    for c in ["pmRA", "pmDE"]:
        if isinstance(tab[c], MaskedColumn):
            tab[c].fill_value = np.nan
        tab[c] = tab[c].astype(float)
        tab[c].unit = u.mas / u.year

    for c in ["Period", "Epoch"]:
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
        tab["max"] = tab["max"].astype(float)  # needed in case if the parsed data has nan
        tab["max"].unit = u.mag
        tab["u_max"] = [t["u"] for t in max_parsed]
        tab["n_max"] = [t["band"] for t in max_parsed]

    # MinMag: parse to Vizier's min, n_min, and f_min ("Y" for amplitude)
    if "MinMag" in tab.colnames:
        min_parsed = [_parse_limit_mag_amp_uncertainty_band(v) for v in tab["MinMag"]]
        tab["l_min"] = [t["l"] for t in min_parsed]
        tab["min"] = [t["mag"] for t in min_parsed]
        tab["min"] = tab["min"].astype(float)  # needed in case if the parsed data has nan
        tab["min"].unit = u.mag
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
    if magnitude_limit is not None:
        query_url += f"&tomag={magnitude_limit}"

    try:
        log.debug(f"Query VSX with: {query_url}")
        result = _do_remote_query(query_url)
        return _parse_response(result)
    except Exception as e:
        raise LightkurveError(f"Error in querying VSX with {query_url}", e)


def _to_mag_text(tab):
    def _do_to_text(r):  # for one row
        if np.isnan(r["min"]):
            # no range / amplitude
            return f'{r["l_max"]}{r["max"]}{r["u_max"]} {r["n_max"]}'

        # show max - min or mag (amplitude)
        if r["n_max"] == r["n_min"]:
            # passband the same, abbreviate them to be shown only
            n_max, n_min = "", f' {r["n_min"]}'
        else:
            n_max, n_min = f' {r["n_max"]}', f' {r["n_min"]}'

        if r["f_min"] != "Y":
            # max mag - min mag format
            return f'{r["l_max"]}{r["max"]}{r["u_max"]}{n_max} - {r["l_min"]}{r["min"]}{r["u_min"]}{n_min}'
        else:
            # magnitude - amplitude format
            return f'{r["l_max"]}{r["max"]}{r["u_max"]}{n_max}  ({r["l_min"]}{r["min"]}{r["u_min"]}){n_min}'

    return [_do_to_text(row) for row in tab]


class VSXInteractSkyCatalogProvider(InteractSkyCatalogProvider):
    """
    Provide AAVSO Variable Star Index data to
    `TargetPixelFile.interact_sky() <lightkurve.TargetPixelFile.interact_sky>`.

    More information: http://aavso.org/vsx/index.php

    https://www.aavso.org/apis-aavso-resources#:~:text=The%20VSX%20API%20is%20not%20%22official%22%20yet

    The class is used by ``interact_sky()`` internally. The behavior can
    be customized by supplying a dictionary of keyword parameters to
    `catalogs` parameter of ``interact_sky()``. The keyword parameters are
    then used to customize the parameters passed to the constructor here.

    Parameters
    ----------
    coord: `~astropy.coordinates.SkyCoord`
        the coordinate of the target.

    radius: float or `~astropy.units.Quantity`
        the cone search radius, in arc seconds if the value is float.

    magnitude_limit: float
        A value to limit the results in based on magnitude in VSX (mean or maximum).

    scatter_kwargs: dict
        keyword arguments passed to bokeh's ``figure.scatter()``
        function to plot the stars.

    """

    J2000 = Time(2000.0, format="jyear", scale="tt")

    def __init__(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
    ) -> None:
        if scatter_kwargs is None:
            scatter_kwargs = dict(
                marker="x",
                line_color="firebrick",
                line_width=2,
            )
        super().__init__(coord, radius, magnitude_limit, scatter_kwargs)
        # VSX-specific
        self.cols_for_source = [
            "Name",
            "OID",
            "Type",
            "Period",
            "Epoch",
            "magText",
        ]

    @property
    def label(self) -> str:
        return "VSX"

    def query_catalog(self) -> Table:
        # delay the import to here to avoid circular import
        from ..interact import _correct_with_proper_motion

        # VSX data is in J2000 coordinate, so we transform the given coordinate, usually PM corrected,
        # back to one in J2000

        # Perform PM correction only if the supplied coordinate has PM info
        # otherwise, if a coordinate has no PM
        # - calling apply_space_motion() would raise ValueError: SkyCoord requires velocity data to evolve the position.
        # - one cannot test if the coordinate has PM with `.pm_ra_cosdec` or `.pm_dec` attributes, as accessing them
        #   would lead to TypeError: Frame data has no associated differentials (i.e. the frame has no velocity
        #   data) - represent_as() only accepts a new representation.
        # - so for now we use a rather obscure implementation details to test if the coord as PM.
        # see: https://github.com/astropy/astropy/issues/16541
        if "s" in self.coord.frame.data.differentials:
            ra2000, dec2000, _ = _correct_with_proper_motion(
                self.coord.ra, self.coord.dec,
                self.coord.pm_ra_cosdec, self.coord.pm_dec,
                "icrs", self.coord.obstime,
                self.J2000
            )
        else:
            ra2000, dec2000 = self.coord.ra, self.coord.dec

        rs = _query_cone_region(
            ra2000.to(u.deg).value, dec2000.to(u.deg).value, self.radius.to(u.deg).value, self.magnitude_limit
        )
        if rs is not None:
            rs["magForSize"] = 10  # use constant marker size
            rs["magText"] = _to_mag_text(rs)
        return rs

    def get_proper_motion_correction_meta(self) -> ProperMotionCorrectionMeta:
        return ProperMotionCorrectionMeta("RAJ2000", "DEJ2000", "pmRA", "pmDE", "icrs", self.J2000)

    def get_tooltips(self) -> list:
        return [
            ("Name", "@Name"),
            ('Separation (")', "@separation{0.00}"),
            ("Type", "@Type"),
            ("Magnitude", "@magText"),
            ("RA", "@ra{0,0.00000000}"),
            ("DEC", "@dec{0,0.00000000}"),
            ("column", "@x{0.0}"),
            ("row", "@y{0.0}"),
        ]

    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        vsx_url = f"https://www.aavso.org/vsx/index.php?view=detail.top&oid={data['OID']}"
        if np.isnan(data["Epoch"]):
            epoch_text = ""
        else:
            # iso date, e.g. 2019-07-31, followed by HJD
            epoch_text = Time(data["Epoch"], format="jd", scale="utc").to_value("iso", subfmt="date")
            epoch_text += f' (HJD {data["Epoch"]})'
        return {
            "Name": f"""{data['Name']} (<a href="{vsx_url}" target="_blank">VSX</a>)""",
            'Separation (")': f"{data['separation']:.2f}",
            "Variability type": data["Type"],
            "Magnitude": data["magText"],
            "Period (d)": data["Period"],
            "Epoch": epoch_text,
            "RA": f"{data['ra']:.8f}",
            "DEC": f"{data['dec']:.8f}",
            "column": f"{data['x']:.1f}",
            "row": f"{data['y']:.1f}",
        }, None
