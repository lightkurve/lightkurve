import logging
from typing import Tuple, Union

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

import numpy as np

import lightkurve as lk

from .core import ProperMotionCorrectionMeta, InteractSkyCatalogProvider

log = logging.getLogger(__name__)

# import skyPatrol, or print a friendly error otherwise.
# - requires pip install skypatrol
# - https://github.com/asas-sn/skypatrol
_SKYPATROL_IMPORT_ERROR = None
try:
    from pyasassn.client import SkyPatrolClient
except Exception as e:
    _SKYPATROL_IMPORT_ERROR = e


def _query_cone_region(coord, radius) -> Table:
    client = SkyPatrolClient(verbose=False)

    # URL for lightcurve: no easy way to construct one
    # e.g., the CSV from their UI is via a blob URL
    # need to use their python client
    # OPEN: consider to using AQDL and join with asasn_discoveries table

    df = client.cone_search(
        coord.ra.to(u.deg).value,
        coord.dec.to(u.deg).value,
        radius.to(u.arcsec).value,
        units="arcsec",
        catalog="stellar_main",  # the default master_list has too little data
        cols=["asas_sn_id", "gaia_mag", "ra_deg", "dec_deg", "pm_ra", "pm_dec"],
    )

    tab = Table.from_pandas(df)
    tab["gaia_mag"].unit = u.mag
    tab["ra_deg"].unit = u.deg
    tab["dec_deg"].unit = u.deg
    tab["pm_ra"].unit = u.milliarcsecond / u.year
    tab["pm_dec"].unit = u.milliarcsecond / u.year

    return tab


def get_lightcurve(
    asas_sn_id, use_native=False, shift_mag=True, good_quality_only=True
):
    client = SkyPatrolClient(verbose=False)
    lcc = client.query_list(
        [asas_sn_id], catalog="stellar_main", id_col="asas_sn_id", download=True
    )
    if len(lcc) < 1:
        return None

    lc = lcc[asas_sn_id]

    if use_native:
        return lc

    # convert their lightcurve to lk LightCurve object
    # only extract columns for plotting
    data = lc.data

    # columns skipped
    # 'flux', 'flux_err', 'limit', 'fwhm', 'image_id', 'camera', 'quality',
    lc = lk.LightCurve(
        data=dict(
            time=Time(data.jd, format="jd", scale="utc"),  # assumed to be HJD for now
            flux=data.mag,
            flux_err=data.mag_err,
            quality=data.quality,
            phot_filter=data.phot_filter,
        )
    )
    for c in ["flux", "flux_err"]:
        # specify the units, somehow doing it in `data` above does not work.
        lc[c] = lc[c] * u.mag

    lc.meta.update(
        {"TARGETID": asas_sn_id, "LABEL": f"ASAS-SN Sky Patrol {asas_sn_id}"}
    )

    if good_quality_only:
        lc = lc[lc["quality"] == "G"]

    if shift_mag:
        lc = _shift_mag(lc)

    return lc


def _shift_mag(lc):
    filters = np.unique(lc["phot_filter"])
    if len(filters) < 2:
        return lc

    lc_all = lc

    base_f = filters[0]
    lc = lc_all[lc_all["phot_filter"] == base_f]
    base_median_mag = np.nanmedian(lc.flux)
    lc.meta["BASE_OF_SHIFT"] = base_f
    label_suffix = f", {base_f}"

    for f in filters[1:]:
        f_lc = lc_all[lc_all["phot_filter"] == f]
        f_median_mag = np.nanmedian(f_lc.flux)
        f_shift_mag = base_median_mag - f_median_mag
        f_lc.flux += f_shift_mag
        lc = lc.append(f_lc)

        if f_shift_mag >= 0:
            f_shift_str = f"{f} +{f_shift_mag:.2f} mag"
        else:
            f_shift_str = f"{f} {f_shift_mag:.2f} mag"

        lc.meta[f"SHIFT_{f.upper()}"] = f_shift_mag
        label_suffix += f", {f_shift_str}"

    # update label at the end once, to avoid
    # the ambiguity in label to use during lc.append()
    lc.meta["LABEL"] += label_suffix

    return lc


class SkyPatrol2InteractSkyCatalogProvider(InteractSkyCatalogProvider):
    """
    Provide ASAS-SN SkyPatrol V2.0 Archive data to
    `TargetPixelFile.interact_sky() <lightkurve.TargetPixelFile.interact_sky>`.

    More information: http://asas-sn.ifa.hawaii.edu/skypatrol/


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
        A value to limit the results in based on Gaia Gmag.

    scatter_kwargs: dict
        keyword arguments passed to bokeh's ``figure.scatter()``
        function to plot the stars.
    """

    J2015_5 = Time(2015.5, format="jyear", scale="tt")

    def __init__(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
    ) -> None:
        if _SKYPATROL_IMPORT_ERROR is not None:
            log.error(
                "Using SkyPatrol v2 with `interact_sky()` requires the `skypatrol` Python package; "
                "you can install bokeh using e.g. `pip install skypatrol`."
            )
            raise _SKYPATROL_IMPORT_ERROR

        if scatter_kwargs is None:
            scatter_kwargs = dict(
                marker="diamond",
                fill_alpha=0.2,
                line_color=None,
                selection_color="red",
                nonselection_fill_alpha=0.2,
                nonselection_line_color=None,
                nonselection_line_alpha=1.0,
                fill_color="red",
                hover_fill_color="red",
                hover_alpha=0.9,
                hover_line_color="white",
            )
        super().__init__(coord, radius, magnitude_limit, scatter_kwargs)
        # SkyPatrol-specific
        self.cols_for_source = [  # extra columns to be included in bokeh data source
            "asas_sn_id",
            "gaia_mag",
        ]

    @property
    def label(self) -> str:
        return "SkyPatrol v2"

    def query_catalog(self) -> Table:
        rs = _query_cone_region(self.coord, self.radius)

        # Tweak result to fit interact_sky() needs

        # magForSize: use a constant size, to avoid
        # 1. bright targets distracting (usually there'd be a Gaia DR3 counterpart)
        # 2. dots too small for dim ones
        rs["magForSize"] = 11

        return rs

    def get_proper_motion_correction_meta(self) -> ProperMotionCorrectionMeta:
        # the ra / dec returned is from Gaia DR2, using J2015.5 epoch
        return ProperMotionCorrectionMeta(
            "ra_deg", "dec_deg", "pm_ra", "pm_dec", "icrs", self.J2015_5
        )

    def get_tooltips(self) -> list:
        return [
            ("ASAS-SN ID", "@asas_sn_id"),
            ('Separation (")', "@separation{0.00}"),
            ("Gaia Mag", "@gaia_mag"),
            ("RA", "@ra{0,0.00000000}"),
            ("DEC", "@dec{0,0.00000000}"),
            ("column", "@x{0.0}"),
            ("row", "@y{0.0}"),
        ]

    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        skypatrol2_site_url = (
            f"http://asas-sn.ifa.hawaii.edu/skypatrol/objects/{data['asas_sn_id']}"
        )
        return {
            "ASAS-SN ID": f"""{data['asas_sn_id']} (<a href="{skypatrol2_site_url}" target="_blank">SkyPatrol v2</a>)""",
            'Separation (")': f"{data['separation']:.2f}",
            "Gaia Mag": f"{data['gaia_mag']:.3f}",
            "RA": f"{data['ra']:.8f}",
            "DEC": f"{data['dec']:.8f}",
            "column": f"{data['x']:.1f}",
            "row": f"{data['y']:.1f}",
        }, None
