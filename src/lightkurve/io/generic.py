"""Read a generic FITS table containing a light curve."""
import logging
import warnings

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import UnitsWarning
import numpy as np

from ..utils import validate_method
from ..lightcurve import LightCurve
from ..units import ppm


log = logging.getLogger(__name__)


def read_generic_lightcurve(
    filename,
    time_column="time",
    flux_column="flux",
    flux_err_column="flux_err",
    quality_column="quality",
    cadenceno_column="cadenceno",
    centroid_col_column="mom_centr1",
    centroid_row_column="mom_centr2",
    time_format=None,
    ext=1,
):
    """Generic helper function to convert a Kepler ot TESS light curve file
    into a generic `LightCurve` object.
    """
    to_close_hdul = True
    if (isinstance(filename, str) and filename.startswith('s3://')):
        # Filename is an S3 cloud URI
        log.debug('Reading file from AWS S3 cloud.')
        hdulist = fits.open(filename, use_fsspec=True, fsspec_kwargs={"anon": True})
    elif isinstance(filename, fits.HDUList):
        hdulist, to_close_hdul = filename, False # Allow HDUList to be passed
    else:
        hdulist = fits.open(filename)

    try:
        # Raise an exception if the requested extension is invalid
        if isinstance(ext, str):
            validate_method(ext, supported_methods=[hdu.name.lower() for hdu in hdulist])
        with warnings.catch_warnings():
            # By default, AstroPy emits noisy warnings about units commonly used
            # in archived TESS data products (e.g., "e-/s" and "pixels").
            # We ignore them here because they don't affect Lightkurve's features.
            # Inconsistencies between TESS data products and the FITS standard
            # out to be addressed at the archive level. (See issue #1216.)
            warnings.simplefilter("ignore", category=UnitsWarning)
            tab = Table.read(hdulist[ext], format="fits")

        # Make sure the meta data also includes header fields from extension #0
        tab.meta.update(hdulist[0].header)

        tab.meta = {k: v for k, v in tab.meta.items()}

        for colname in tab.colnames:
            # Ensure units have the correct astropy format
            # Speed-up: comparing units by their string representation is 1000x
            # faster than performing full-blown unit comparison
            unitstr = str(tab[colname].unit)
            if unitstr == "e-/s":
                tab[colname].unit = "electron/s"
            elif unitstr == "pixels":
                tab[colname].unit = "pixel"
            elif unitstr == "ppm" and repr(tab[colname].unit).startswith("Unrecognized"):
                # Workaround for issue #956
                tab[colname].unit = ppm
            elif unitstr == "ADU":
                tab[colname].unit = "adu"
            elif unitstr.lower() == "unitless":
                tab[colname].unit = ""
            elif unitstr.lower() == "degcelcius":
                # CDIPS has non-astropy units
                tab[colname].unit = "deg_C"
            # Rename columns to lowercase
            tab.rename_column(colname, colname.lower())

        # Some KEPLER files used to have a T column instead of TIME.
        if time_column == "time" and "time" not in tab.columns and "t" in tab.colnames:
            tab.rename_column("t", "time")
        if time_column != "time":
            tab.rename_column(time_column, "time")

        # We *have* to remove rows with TIME=NaN because the Astropy Time
        # object does not support the presence of NaNs.
        # Fortunately, such rows are always bad data.
        nans = np.isnan(tab["time"].data)
        if np.any(nans):
            log.debug("Ignoring {} rows with NaN times".format(np.sum(nans)))
        tab = tab[~nans]

        # Prepare a special time column
        if not time_format:
            if hdulist[ext].header.get("BJDREFI") == 2454833:
                time_format = "bkjd"
            elif hdulist[ext].header.get("BJDREFI") == 2457000:
                time_format = "btjd"
            else:
                raise ValueError(f"Input file has unclear time format: {filename}")
        time = Time(
            tab["time"].data,
            scale=hdulist[ext].header.get("TIMESYS", "tdb").lower(),
            format=time_format,
        )
        tab.remove_column("time")

        # For backwards compatibility with Lightkurve v1.x,
        # we make sure standard columns and attributes exist.
        if "flux" not in tab.columns:
            tab.add_column(tab[flux_column], name="flux", index=0)
        if "flux_err" not in tab.columns:
            # Try falling back to `{flux_column}_err` if possible
            if flux_err_column not in tab.columns:
                flux_err_column = flux_column + "_err"
            if flux_err_column in tab.columns:
                tab.add_column(tab[flux_err_column], name="flux_err", index=1)
        if "quality" not in tab.columns and quality_column in tab.columns:
            tab.add_column(tab[quality_column], name="quality", index=2)
        if "cadenceno" not in tab.columns and cadenceno_column in tab.columns:
            tab.add_column(tab[cadenceno_column], name="cadenceno", index=3)
        if "centroid_col" not in tab.columns and centroid_col_column in tab.columns:
            tab.add_column(tab[centroid_col_column], name="centroid_col", index=4)
        if "centroid_row" not in tab.columns and centroid_row_column in tab.columns:
            tab.add_column(tab[centroid_row_column], name="centroid_row", index=5)

        tab.meta["LABEL"] = hdulist[0].header.get("OBJECT")
        tab.meta["MISSION"] = hdulist[0].header.get(
            "MISSION", hdulist[0].header.get("TELESCOP")
        )
        tab.meta["RA"] = hdulist[0].header.get("RA_OBJ")
        tab.meta["DEC"] = hdulist[0].header.get("DEC_OBJ")
        tab.meta["FILENAME"] = filename
        tab.meta["FLUX_ORIGIN"] = flux_column

        return LightCurve(time=time, data=tab)
    finally:
        if to_close_hdul:
            # avoid hdulist closing from emitting exceptions
            hdulist.close(output_verify="warn")
