from typing import Tuple, Union

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

# make SkyPatrol import optional
# - requires pip install skypatrol
# - https://github.com/asas-sn/skypatrol
_SKYPATROL_IMPORT_ERROR = None
try:
    from pyasassn.client import SkyPatrolClient
except Exception as e:
    _SKYPATROL_IMPORT_ERROR = e

import numpy as np

import lightkurve as lk

from .core import ProperMotionCorrectionMeta, InteractSkyCatalogProvider


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

    return Table.from_pandas(df)


def get_lightcurve(asas_sn_id, use_native=False, shift_mag=True, good_quality_only=True):
    client = SkyPatrolClient(verbose=False)
    lcc = client.query_list([asas_sn_id], catalog="stellar_main", id_col="asas_sn_id", download=True)
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
    lc = lk.LightCurve(data=dict(
        time=Time(data.jd, format="jd", scale="utc"),  # assumed to be HJD for now
        flux=data.mag*u.mag,
        flux_err=data.mag_err*u.mag,
        quality=data.quality,
        phot_filter=data.phot_filter,
    ))
    lc.meta.update({
        "TARGETID": asas_sn_id,
        "LABEL": f"ASAS-SN Sky Patrol {asas_sn_id}"
    })

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
