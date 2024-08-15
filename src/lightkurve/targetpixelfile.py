"""Defines TargetPixelFile, KeplerTargetPixelFile, and TessTargetPixelFile."""

from __future__ import division
import datetime
import os
import warnings
import logging

import collections

from astropy.io import fits
from astropy.io.fits import Undefined, BinTableHDU
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from astropy.coordinates import SkyCoord
from astropy.stats.funcs import median_absolute_deviation as MAD
from astropy.utils.decorators import deprecated
from astropy.time import Time
from astropy.units import Quantity
import astropy.units as u

import matplotlib
from matplotlib import animation
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
from copy import deepcopy

from . import PACKAGEDIR, MPLSTYLE
from .lightcurve import LightCurve, KeplerLightCurve, TessLightCurve
from .prf import KeplerPRF
from .utils import (
    KeplerQualityFlags,
    TessQualityFlags,
    plot_image,
    LightkurveWarning,
    LightkurveDeprecationWarning,
    validate_method,
    centroid_quadratic,
    _query_solar_system_objects,
    finalize_notebook_url
)
from .io import detect_filetype


__all__ = ["KeplerTargetPixelFile", "TessTargetPixelFile"]


log = logging.getLogger(__name__)


# OPEN: consider to move to utils and
# consolidate with the helper in lightcurve.py (for time label)
_TIME_LABEL_DICT_BRIEF = {"": "Phase", "bkjd": "[BKJD days]", "btjd": "[BTJD days]"}


def _time_label_brief(time):
    format = getattr(time, "format", "")
    return _TIME_LABEL_DICT_BRIEF.get(format, format.upper())


class HduToMetaMapping(collections.abc.Mapping):
    """Provides a read-only view of HDU header in `astropy.timeseries.TimeSeries.meta` format"""

    def __init__(self, hdu):
        # use OrderedDict rather than simple dict for 2 reasons:
        # 1. more friendly __repr__ and __str__
        # 2. make the behavior between a TPF and a LC is more consistent.
        #    (LightCurve.meta is an OrderedDict)
        self._dict = collections.OrderedDict()
        self._dict.update(hdu.header)

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __repr__(self):
        return self._dict.__repr__()

    def __str__(self):
        return self._dict.__str__()


class TargetPixelFile(object):
    """Abstract class representing FITS files which contain time series imaging data.

    You should probably not be using this abstract class directly;
    see `KeplerTargetPixelFile` and `TessTargetPixelFile` instead.
    """

    def __init__(self, path, quality_bitmask="default", targetid=None, **kwargs):
        self.path = path
        if isinstance(path, fits.HDUList):
            self.hdu = path
        elif (isinstance(path, str) and path.startswith('s3://')):
            # Filename is an S3 cloud URI
            self.hdu = fits.open(path, use_fsspec=True, fsspec_kwargs={"anon": True}, **kwargs)
        else:
            self.hdu = fits.open(self.path, **kwargs)
        try:
            self.quality_bitmask = quality_bitmask
            self.targetid = targetid

            # For consistency with `LightCurve`, provide a `meta` dictionary
            self.meta = HduToMetaMapping(self.hdu[0])
        except Exception as e:
            # Cannot instantiate TargetPixelFile, close the HDU to release the file handle
            self.hdu.close()
            raise e

    def __getitem__(self, key):
        """Implements indexing and slicing.

        Note: the implementation below cannot be be simplified using
            `copy[1].data = copy[1].data[self.quality_mask][key]`
        due to the complicated behavior of AstroPy's `FITS_rec`.
        """
        # Step 1: determine the indexes of the data to return.
        # We start by determining the indexes of the good-quality cadences.
        quality_idx = np.where(self.quality_mask)[0]
        # Then we apply the index or slice to the good-quality indexes.
        if isinstance(key, int):
            # Ensure we always have a range; this is necessary to ensure
            # that we always ge a  `FITS_rec` instead of a `FITS_record` below.
            if key == -1:
                selected_idx = quality_idx[key:]
            else:
                selected_idx = quality_idx[key : key + 1]
        else:
            selected_idx = quality_idx[key]

        # Step 2: use the indexes to create a new copy of the data.
        with warnings.catch_warnings():
            # Ignore warnings about empty fields
            warnings.simplefilter("ignore", UserWarning)
            # AstroPy added `HDUList.copy()` in v3.1, allowing us to avoid manually
            # copying the HDUs, which brought along unexpected memory leaks.
            copy = self.hdu.copy()
            copy[1] = BinTableHDU(
                data=self.hdu[1].data[selected_idx], header=self.hdu[1].header
            )
        return self.__class__(
            copy, quality_bitmask=self.quality_bitmask, targetid=self.targetid
        )

    def __len__(self):
        return len(self.time)

    def __add__(self, other):
        if isinstance(other, Quantity):
            other = other.value
        hdu = deepcopy(self.hdu)
        hdu[1].data["FLUX"][self.quality_mask] += other
        return type(self)(hdu, quality_bitmask=self.quality_bitmask)

    def __mul__(self, other):
        if isinstance(other, Quantity):
            other = other.value
        hdu = deepcopy(self.hdu)
        hdu[1].data["FLUX"][self.quality_mask] *= other
        hdu[1].data["FLUX_ERR"][self.quality_mask] *= other
        return type(self)(hdu, quality_bitmask=self.quality_bitmask)

    def __rtruediv__(self, other):
        if isinstance(other, Quantity):
            other = other.value
        hdu = deepcopy(self.hdu)
        hdu[1].data["FLUX"][self.quality_mask] /= other
        hdu[1].data["FLUX_ERR"][self.quality_mask] /= other
        return type(self)(hdu, quality_bitmask=self.quality_bitmask)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    @property
    @deprecated("2.0", alternative="time", warning_type=LightkurveDeprecationWarning)
    def astropy_time(self):
        """Returns an AstroPy Time object for all good-quality cadences."""
        return self.time

    @property
    def hdu(self):
        return self._hdu

    @hdu.setter
    def hdu(self, value, keys=("FLUX", "QUALITY")):
        """Verify the file format when setting the value of `self.hdu`.

        Raises a ValueError if `value` does not appear to be a Target Pixel File.
        """
        for key in keys:
            if ~(
                np.any(
                    [
                        value[1].header[ttype] == key
                        for ttype in value[1].header["TTYPE*"]
                    ]
                )
            ):
                raise ValueError(
                    "File {} does not have a {} column, "
                    "is this a target pixel file?".format(self.path, key)
                )
        self._hdu = value

    def get_keyword(self, keyword, hdu=0, default=None):
        """Returns a header keyword value.

        If the keyword is Undefined or does not exist,
        then return ``default`` instead.
        """
        return self.hdu[hdu].header.get(keyword, default)

    @property
    @deprecated(
        "2.0", alternative="get_header()", warning_type=LightkurveDeprecationWarning
    )
    def header(self):
        """DEPRECATED. Please use ``get_header()`` instead."""
        return self.hdu[0].header

    def get_header(self, ext=0):
        """Returns the metadata embedded in the file.

        Target Pixel Files contain embedded metadata headers spread across three
        different FITS extensions:

        1. The "PRIMARY" extension (``ext=0``) provides a metadata header
           providing details on the target and its CCD position.
        2. The "PIXELS" extension (``ext=1``) provides details on the
           data column and their coordinate system (WCS).
        3. The "APERTURE" extension (``ext=2``) provides details on the
           aperture pixel mask and the expected coordinate system (WCS).

        Parameters
        ----------
        ext : int or str
            FITS extension name or number.

        Returns
        -------
        header : `~astropy.io.fits.header.Header`
            Header object containing metadata keywords.
        """
        return self.hdu[ext].header

    @property
    def ra(self):
        """Right Ascension of target ('RA_OBJ' header keyword)."""
        return self.get_keyword("RA_OBJ")

    @property
    def dec(self):
        """Declination of target ('DEC_OBJ' header keyword)."""
        return self.get_keyword("DEC_OBJ")

    @property
    def column(self):
        """CCD pixel column number ('1CRV5P' header keyword)."""
        return self.get_keyword("1CRV5P", hdu=1, default=0)

    @property
    def row(self):
        """CCD pixel row number ('2CRV5P' header keyword)."""
        return self.get_keyword("2CRV5P", hdu=1, default=0)

    @property
    def pos_corr1(self):
        """Returns the column position correction."""
        return self.hdu[1].data["POS_CORR1"][self.quality_mask]

    @property
    def pos_corr2(self):
        """Returns the row position correction."""
        return self.hdu[1].data["POS_CORR2"][self.quality_mask]

    @property
    def pipeline_mask(self):
        """Returns the optimal aperture mask used by the pipeline.

        If the aperture extension is missing from the file, a mask
        composed of all `True` values will be returned.
        """
        # Both Kepler and TESS flag the pixels in the optimal aperture using
        # bit number 2 in the aperture mask extension, e.g. see Section 6 of
        # the TESS Data Products documentation (EXP-TESS-ARC-ICD-TM-0014.pdf).
        try:
            return self.hdu[2].data & 2 > 0
        except (IndexError, TypeError):
            # `IndexError` may be raised if the aperture extension (#2) is missing
            # `TypeError` may be raised because early versions of TESScut returned floats in HDU 2
            return np.ones(self.hdu[1].data["FLUX"][0].shape, dtype=bool)

    @property
    def shape(self):
        """Return the cube dimension shape."""
        return self.flux.shape

    @property
    def time(self) -> Time:
        """Returns the time for all good-quality cadences."""
        time_values = self.hdu[1].data["TIME"][self.quality_mask]
        # Some data products have missing time values;
        # we need to set these to zero or `Time` cannot be instantiated.
        time_values[~np.isfinite(time_values)] = 0

        bjdrefi = self.hdu[1].header.get("BJDREFI")
        if bjdrefi == 2454833:
            time_format = "bkjd"
        elif bjdrefi == 2457000:
            time_format = "btjd"
        else:
            time_format = "jd"

        return Time(
            time_values,
            scale=self.hdu[1].header.get("TIMESYS", "tdb").lower(),
            format=time_format,
        )

    @property
    def cadenceno(self):
        """Return the cadence number for all good-quality cadences."""
        cadenceno = self.hdu[1].data["CADENCENO"][self.quality_mask]
        # The TESScut service returns an array of zeros as CADENCENO.
        # If this is the case, return frame numbers from 0 instead.
        if cadenceno[0] == 0:
            return np.arange(0, len(cadenceno), 1, dtype=int)
        return cadenceno

    @property
    def nan_time_mask(self):
        """Returns a boolean mask flagging cadences whose time is `nan`."""
        return self.time.value == 0

    @property
    def flux(self) -> Quantity:
        """Returns the flux for all good-quality cadences."""
        unit = None
        if self.get_header(1).get("TUNIT5") == "e-/s":
            unit = "electron/s"
        return Quantity(self.hdu[1].data["FLUX"][self.quality_mask], unit=unit)

    @property
    def flux_err(self) -> Quantity:
        """Returns the flux uncertainty for all good-quality cadences."""
        unit = None
        if self.get_header(1).get("TUNIT6") == "e-/s":
            unit = "electron/s"
        return Quantity(self.hdu[1].data["FLUX_ERR"][self.quality_mask], unit=unit)

    @property
    def flux_bkg(self) -> Quantity:
        """Returns the background flux for all good-quality cadences."""
        return Quantity(
            self.hdu[1].data["FLUX_BKG"][self.quality_mask], unit="electron/s"
        )

    @property
    def flux_bkg_err(self) -> Quantity:
        return Quantity(
            self.hdu[1].data["FLUX_BKG_ERR"][self.quality_mask], unit="electron/s"
        )

    @property
    def quality(self):
        """Returns the quality flag integer of every good cadence."""
        return self.hdu[1].data["QUALITY"][self.quality_mask]

    @property
    def wcs(self) -> WCS:
        """Returns an `astropy.wcs.WCS` object with the World Coordinate System
        solution for the target pixel file.

        Returns
        -------
        w : `astropy.wcs.WCS` object
            WCS solution
        """
        if "MAST" in self.hdu[0].header["ORIGIN"]:  # Is it a TessCut TPF?
            # TPF's generated using the TESSCut service in early 2019 only appear
            # to contain a valid WCS in the second extension (the aperture
            # extension), so we treat such files as a special case.
            return WCS(self.hdu[2])
        else:
            # For standard (Ames-pipeline-produced) TPF files, we use the WCS
            # keywords provided in the first extension (the data table extension).
            # Specifically, we use the WCS keywords for the 5th data column (FLUX).
            wcs_keywords = {
                "1CTYP5": "CTYPE1",
                "2CTYP5": "CTYPE2",
                "1CRPX5": "CRPIX1",
                "2CRPX5": "CRPIX2",
                "1CRVL5": "CRVAL1",
                "2CRVL5": "CRVAL2",
                "1CUNI5": "CUNIT1",
                "2CUNI5": "CUNIT2",
                "1CDLT5": "CDELT1",
                "2CDLT5": "CDELT2",
                "11PC5": "PC1_1",
                "12PC5": "PC1_2",
                "21PC5": "PC2_1",
                "22PC5": "PC2_2",
                "NAXIS1": "NAXIS1",
                "NAXIS2": "NAXIS2",
            }
            mywcs = {}
            for oldkey, newkey in wcs_keywords.items():
                if self.hdu[1].header.get(oldkey, None) is not None:
                    mywcs[newkey] = self.hdu[1].header[oldkey]
            return WCS(mywcs)

    def get_coordinates(self, cadence="all"):
        """Returns two 3D arrays of RA and Dec values in decimal degrees.

        If cadence number is given, returns 2D arrays for that cadence. If
        cadence is 'all' returns one RA, Dec value for each pixel in every cadence.
        Uses the WCS solution and the POS_CORR data from TPF header.

        Parameters
        ----------
        cadence : 'all' or int
            Which cadences to return the RA Dec coordinates for.

        Returns
        -------
        ra : numpy array, same shape as tpf.flux[cadence]
            Array containing RA values for every pixel, for every cadence.
        dec : numpy array, same shape as tpf.flux[cadence]
            Array containing Dec values for every pixel, for every cadence.
        """
        w = self.wcs
        X, Y = np.meshgrid(np.arange(self.shape[2]), np.arange(self.shape[1]))
        pos_corr1_pix = np.copy(self.hdu[1].data["POS_CORR1"])
        pos_corr2_pix = np.copy(self.hdu[1].data["POS_CORR2"])

        # We zero POS_CORR* when the values are NaN or make no sense (>50px)
        with warnings.catch_warnings():  # Comparing NaNs to numbers is OK here
            warnings.simplefilter("ignore", RuntimeWarning)
            bad = np.any(
                [
                    ~np.isfinite(pos_corr1_pix),
                    ~np.isfinite(pos_corr2_pix),
                    np.abs(pos_corr1_pix - np.nanmedian(pos_corr1_pix)) > 50,
                    np.abs(pos_corr2_pix - np.nanmedian(pos_corr2_pix)) > 50,
                ],
                axis=0,
            )
        pos_corr1_pix[bad], pos_corr2_pix[bad] = 0, 0

        # Add in POSCORRs
        X = np.atleast_3d(X).transpose([2, 0, 1]) + np.atleast_3d(
            pos_corr1_pix
        ).transpose([1, 2, 0])
        Y = np.atleast_3d(Y).transpose([2, 0, 1]) + np.atleast_3d(
            pos_corr2_pix
        ).transpose([1, 2, 0])

        # Pass through WCS
        ra, dec = w.wcs_pix2world(X.ravel(), Y.ravel(), 1)
        ra = ra.reshape((pos_corr1_pix.shape[0], self.shape[1], self.shape[2]))
        dec = dec.reshape((pos_corr2_pix.shape[0], self.shape[1], self.shape[2]))
        ra, dec = ra[self.quality_mask], dec[self.quality_mask]
        if cadence != "all":
            return ra[cadence], dec[cadence]
        return ra, dec

    def show_properties(self):
        """Prints a description of all non-callable attributes.

        Prints in order of type (ints, strings, lists, arrays, others).
        """
        attrs = {}
        for attr in dir(self):
            if not attr.startswith("_") and attr != "header" and attr != "astropy_time":
                res = getattr(self, attr)
                if callable(res):
                    continue
                if attr == "hdu":
                    attrs[attr] = {"res": res, "type": "list"}
                    for idx, r in enumerate(res):
                        if idx == 0:
                            attrs[attr]["print"] = "{}".format(r.header["EXTNAME"])
                        else:
                            attrs[attr]["print"] = "{}, {}".format(
                                attrs[attr]["print"], "{}".format(r.header["EXTNAME"])
                            )
                    continue
                else:
                    attrs[attr] = {"res": res}
                if isinstance(res, int):
                    attrs[attr]["print"] = "{}".format(res)
                    attrs[attr]["type"] = "int"
                elif isinstance(res, np.ndarray):
                    attrs[attr]["print"] = "array {}".format(res.shape)
                    attrs[attr]["type"] = "array"
                elif isinstance(res, list):
                    attrs[attr]["print"] = "list length {}".format(len(res))
                    attrs[attr]["type"] = "list"
                elif isinstance(res, str):
                    if res == "":
                        attrs[attr]["print"] = "{}".format("None")
                    else:
                        attrs[attr]["print"] = "{}".format(res)
                    attrs[attr]["type"] = "str"
                elif attr == "wcs":
                    attrs[attr]["print"] = "astropy.wcs.wcs.WCS"
                    attrs[attr]["type"] = "other"
                else:
                    attrs[attr]["print"] = "{}".format(type(res))
                    attrs[attr]["type"] = "other"
        output = Table(names=["Attribute", "Description"], dtype=[object, object])
        idx = 0
        types = ["int", "str", "list", "array", "other"]
        for typ in types:
            for attr, dic in attrs.items():
                if dic["type"] == typ:
                    output.add_row([attr, dic["print"]])
                    idx += 1
        output.pprint(max_lines=-1, max_width=-1)

    def to_lightcurve(self, method="sap", corrector=None, **kwargs):
        """Performs photometry on the pixel data and returns a LightCurve object.

        The valid keyword arguments depends on the method chosen:

        - 'sap' or 'aperture': see the docstring of `extract_aperture_photometry()`
        - 'prf': see the docstring of `extract_prf_photometry()`
        - 'pld': see the docstring of `to_corrector()`

        For methods 'sff' and 'cbv', they are syntactic shortcuts of:

        - creating a lightcurve using 'sap' method,
        - corrects the created lightcurve using `LightCurve.to_corrector()`
          of the respective method.

        Parameters
        ----------
        method : 'aperture', 'prf', 'sap', 'sff', 'cbv', 'pld'.
            Photometry method to use. 'aperture' is an alias of 'sap'.
        **kwargs : dict
            Extra arguments to be passed to the `extract_aperture_photometry()`, the
            `extract_prf_photometry()`, or the `to_corrector()` method of this class.

        Returns
        -------
        lc : LightCurve object
            Object containing the resulting lightcurve.
        """
        method = validate_method(method, supported_methods=["aperture", "prf", "sap", "sff", "cbv", "pld"])
        if method in ["aperture", "sap"]:
            return self.extract_aperture_photometry(**kwargs)
        elif method == "prf":
            return self.prf_lightcurve(**kwargs)
        elif method in ["sff", "cbv"]:
            lc = self.extract_aperture_photometry(**kwargs)
            return lc.to_corrector(method).correct()
        elif method == "pld":
            return self.to_corrector("pld", **kwargs).correct()

    def _resolve_default_aperture_mask(self, aperture_mask):
        if isinstance(aperture_mask, str):
            if (aperture_mask == "default"):
                # returns 'pipeline', unless it is missing. Falls back to 'threshold'
                return "pipeline" if np.any(self.pipeline_mask) else "threshold"
            else:
                return aperture_mask
        else:
            return aperture_mask

    def _parse_aperture_mask(self, aperture_mask):
        """Parse the `aperture_mask` parameter as given by a user.

        The `aperture_mask` parameter is accepted by a number of methods.
        This method ensures that the parameter is always parsed in the same way.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'all', 'threshold', 'default',
        'background', or None
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If None or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
            If 'default' is passed, 'pipeline' mask will be used when available,
            with 'threshold' as the fallback.
            If 'background' is passed, all pixels fainter than the median flux
            will be used.
            If 'empty' is passed, no pixels will be used.

        Returns
        -------
        aperture_mask : ndarray
            2D boolean numpy array containing `True` for selected pixels.
        """
        aperture_mask = self._resolve_default_aperture_mask(aperture_mask)

        # If 'pipeline' mask is requested but missing, fall back to 'threshold'
        # To Do: Should pipeline mask always be True?
        if isinstance(aperture_mask, str):
            if (aperture_mask == "pipeline") and ~np.any(self.pipeline_mask):
                raise ValueError(
                    "_parse_aperture_mask: 'pipeline' is requested, but it is missing or empty."
                )

        # Input validation
        if hasattr(aperture_mask, "shape"):
            if (aperture_mask.shape != self.shape[1:]):
                raise ValueError(
                    "`aperture_mask` has shape {}, "
                    "but the flux data has shape {}"
                    "".format(aperture_mask.shape, self.shape[1:])
                )

        if aperture_mask is None:
            aperture_mask = np.ones((self.shape[1], self.shape[2]), dtype=bool)
        elif isinstance(aperture_mask, str):
            if aperture_mask.lower() == "all":
                aperture_mask = np.ones((self.shape[1], self.shape[2]), dtype=bool)
            elif aperture_mask.lower() == "pipeline":
                aperture_mask = self.pipeline_mask
            elif aperture_mask.lower() == "threshold":
                aperture_mask = self.create_threshold_mask()
            elif aperture_mask.lower() == "background":
                aperture_mask = ~self.create_threshold_mask(
                    threshold=0, reference_pixel=None
                )
            elif aperture_mask.lower() == "empty":
                aperture_mask = np.zeros((self.shape[1], self.shape[2]), dtype=bool)
        elif isinstance(aperture_mask, np.ndarray):
            # Kepler and TESS pipeline style integer flags
            if np.issubdtype(aperture_mask.dtype, np.dtype('>i4')):
                aperture_mask = (aperture_mask & 2) == 2
            elif np.issubdtype(aperture_mask.dtype, int):
                if ((aperture_mask & 2) == 2).any():
                    # Kepler and TESS pipeline style integer flags
                    aperture_mask = (aperture_mask & 2) == 2
                else:
                    aperture_mask = aperture_mask.astype(bool)
            elif np.issubdtype(aperture_mask.dtype, float):
                aperture_mask = aperture_mask.astype(bool)
        self._last_aperture_mask = aperture_mask
        return aperture_mask

    def create_threshold_mask(self, threshold=3, reference_pixel="center"):
        """Returns an aperture mask creating using the thresholding method.

        This method will identify the pixels in the TargetPixelFile which show
        a median flux that is brighter than `threshold` times the standard
        deviation above the overall median. The standard deviation is estimated
        in a robust way by multiplying the Median Absolute Deviation (MAD)
        with 1.4826.

        If the thresholding method yields multiple contiguous regions, then
        only the region closest to the (col, row) coordinate specified by
        `reference_pixel` is returned.  For exmaple, `reference_pixel=(0, 0)`
        will pick the region closest to the bottom left corner.
        By default, the region closest to the center of the mask will be
        returned. If `reference_pixel=None` then all regions will be returned.

        Parameters
        ----------
        threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.
        reference_pixel: (int, int) tuple, 'center', or None
            (col, row) pixel coordinate closest to the desired region.
            For example, use `reference_pixel=(0,0)` to select the region
            closest to the bottom left corner of the target pixel file.
            If 'center' (default) then the region closest to the center pixel
            will be selected. If `None` then all regions will be selected.

        Returns
        -------
        aperture_mask : ndarray
            2D boolean numpy array containing `True` for pixels above the
            threshold.
        """
        if reference_pixel == "center":
            reference_pixel = (self.shape[2] / 2, self.shape[1] / 2)
        # Calculate the median image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_image = np.nanmedian(self.flux, axis=0)
        vals = median_image[np.isfinite(median_image)].flatten()
        # Calculate the theshold value in flux units
        mad_cut = (1.4826 * MAD(vals) * threshold) + np.nanmedian(median_image)
        # Create a mask containing the pixels above the threshold flux
        threshold_mask = np.nan_to_num(median_image) >= mad_cut
        if (reference_pixel is None) or (not threshold_mask.any()):
            # return all regions above threshold
            return threshold_mask
        else:
            # Return only the contiguous region closest to `region`.
            # First, label all the regions:
            labels = label(threshold_mask)[0]
            # For all pixels above threshold, compute distance to reference pixel:
            label_args = np.argwhere(labels > 0)
            distances = [
                np.hypot(crd[0], crd[1])
                for crd in label_args
                - np.array([reference_pixel[1], reference_pixel[0]])
            ]
            # Which label corresponds to the closest pixel?
            closest_arg = label_args[np.argmin(distances)]
            closest_label = labels[closest_arg[0], closest_arg[1]]
            return labels == closest_label

    def estimate_background(self, aperture_mask="background"):
        """Returns an estimate of the median background level in the FLUX column.

        In the case of official Kepler and TESS Target Pixel Files, the
        background estimates should be close to zero because these products
        have already been background-subtracted by the pipeline (i.e. the values
        in the `FLUX_BKG` column have been subtracted from the values in `FLUX`).
        Background subtraction is often imperfect however, and this method aims
        to allow users to estimate residual background signals using different
        methods.

        Target Pixel Files created by the MAST TESSCut service have
        not been background-subtracted. For such products, or other community-
        generated pixel files, this method provides a first-order estimate of
        the background levels.

        This method estimates the per-pixel background flux over time by
        computing the median pixel value across the `aperture mask`.

        Parameters
        ----------
        aperture_mask : 'background', 'all', or array-like
            Which pixels should be used to estimate the background?
            If None or 'all' are passed, all pixels in the pixel file will be
            used.  If 'background' is passed, all pixels fainter than the
            median flux will be used. Alternatively, users can pass a boolean
            array describing the aperture mask such that `True` means that the
            pixel will be used.

        Returns
        -------
        lc : `LightCurve` object
            Median background flux in units electron/second/pixel.
        """
        mask = self._parse_aperture_mask(aperture_mask)
        # For each cadence, compute the median pixel flux across the background
        simple_bkg = np.nanmedian(self.flux[:, mask], axis=1) / u.pixel
        return LightCurve(time=self.time, flux=simple_bkg)

    def estimate_centroids(self, aperture_mask="default", method="moments"):
        """Returns the flux center of an object inside ``aperture_mask``.

        Telescopes tend to smear out the light from a point-like star over
        multiple pixels.  For this reason, it is common to estimate the position
        of a star by computing the *geometric center* of its image.
        Astronomers refer to this position as the *centroid* of the object,
        i.e. the term *centroid* is often used as a generic synonym to refer
        to the measured position of an object in a telescope exposure.

        This function provides two methods to estimate the position of a star:

        * `method='moments'` will compute the "center of mass" of the light
          based on the 2D image moments of the pixels inside ``aperture_mask``.
        * `method='quadratic'` will fit a two-dimensional, second-order
          polynomial to the 3x3 patch of pixels centered on the brightest pixel
          inside the ``aperture_mask``, and return the peak of that polynomial.
          Following Vakili & Hogg 2016 (ArXiv:1610.05873, Section 3.2).

        Parameters
        ----------
        aperture_mask : 'pipeline', 'threshold', 'all', 'default', or array-like
            Which pixels contain the object to be measured, i.e. which pixels
            should be used in the estimation?  If None or 'all' are passed,
            all pixels in the pixel file will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
            If 'default' is passed, 'pipeline' mask will be used when available,
            with 'threshold' as the fallback.
            Alternatively, users can pass a boolean array describing the
            aperture mask such that `True` means that the pixel will be used.
        method : 'moments' or 'quadratic'
            Defines which method to use to estimate the centroids. 'moments'
            computes the centroid based on the sample moments of the data.
            'quadratic' fits a 2D polynomial to the data and returns the
            coordinate of the peak of that polynomial.

        Returns
        -------
        columns, rows : `~astropy.units.Quantity`, `~astropy.units.Quantity`
            Arrays containing the column and row positions for the centroid
            for each cadence, or NaN for cadences where the estimation failed.
        """
        method = validate_method(method, ["moments", "quadratic"])
        if method == "moments":
            return self._estimate_centroids_via_moments(aperture_mask=aperture_mask)
        elif method == "quadratic":
            return self._estimate_centroids_via_quadratic(aperture_mask=aperture_mask)

    def _estimate_centroids_via_moments(self, aperture_mask):
        """Compute the "center of mass" of the light based on the 2D moments;
        this is a helper method for `estimate_centroids()`."""
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        yy, xx = np.indices(self.shape[1:])
        yy = self.row + yy
        xx = self.column + xx
        total_flux = np.nansum(self.flux[:, aperture_mask], axis=1)
        with warnings.catch_warnings():
            # RuntimeWarnings may occur below if total_flux contains zeros
            warnings.simplefilter("ignore", RuntimeWarning)
            col_centr = (
                np.nansum(xx * aperture_mask * self.flux, axis=(1, 2)) / total_flux
            )
            row_centr = (
                np.nansum(yy * aperture_mask * self.flux, axis=(1, 2)) / total_flux
            )
        return col_centr * u.pixel, row_centr * u.pixel

    def _estimate_centroids_via_quadratic(self, aperture_mask):
        """Estimate centroids by fitting a 2D quadratic to the brightest pixels;
        this is a helper method for `estimate_centroids()`."""
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        col_centr, row_centr = [], []
        for idx in range(len(self.time)):
            col, row = centroid_quadratic(self.flux[idx], mask=aperture_mask)
            col_centr.append(col)
            row_centr.append(row)
        col_centr = np.asfarray(col_centr) + self.column
        row_centr = np.asfarray(row_centr) + self.row
        col_centr = Quantity(col_centr, unit="pixel")
        row_centr = Quantity(row_centr, unit="pixel")
        return col_centr, row_centr

    def _aperture_photometry(
        self, aperture_mask, flux_method="sum", centroid_method="moments"
    ):
        """Helper method for ``extract_aperture photometry``.

        Returns
        -------
        flux, flux_err, centroid_col, centroid_row
        """
        # Validate the aperture mask
        apmask = self._parse_aperture_mask(aperture_mask)
        if apmask.sum() == 0:
            log.warning("Warning: aperture mask contains zero pixels.")

        # Estimate centroids
        centroid_col, centroid_row = self.estimate_centroids(
            apmask, method=centroid_method
        )

        # Estimate flux
        if flux_method == "sum":
            flux = np.nansum(self.flux[:, apmask], axis=1)

        elif flux_method == "median":
            flux = np.nanmedian(self.flux[:, apmask], axis=1)

        elif flux_method == "mean":
            flux = np.nanmean(self.flux[:, apmask], axis=1)
        else:
            raise ValueError("`flux_method` must be one of 'sum', 'median', or 'mean'.")

        # In the future we may wish to add a user specified function

        # We use ``np.nansum`` above to be robust against a subset of pixels
        # being NaN, however if *all* pixels are NaN, we propagate a NaN.
        is_allnan = ~np.any(np.isfinite(self.flux[:, apmask]), axis=1)
        flux[is_allnan] = np.nan

        # Similarly, if *all* pixel values across the TPF are exactly zero,
        # we propagate NaN (cf. #873 for an example of this happening)
        is_allzero = np.all(self.flux == 0, axis=(1, 2))
        flux[is_allzero] = np.nan

        # Estimate flux_err
        with warnings.catch_warnings():
            # Ignore warnings due to negative errors
            warnings.simplefilter("ignore", RuntimeWarning)
            if flux_method == "sum":
                flux_err = np.nansum(self.flux_err[:, apmask] ** 2, axis=1) ** 0.5

            elif flux_method == "median":
                flux_err = np.nanmedian(self.flux_err[:, apmask] ** 2, axis=1) ** 0.5

            elif flux_method == "mean":
                flux_err = np.nanmean(self.flux_err[:, apmask] ** 2, axis=1) ** 0.5

            is_allnan = ~np.any(np.isfinite(self.flux_err[:, apmask]), axis=1)
            flux_err[is_allnan] = np.nan

        if self.get_header(1).get("TUNIT5") == "e-/s":
            flux = Quantity(flux, unit="electron/s")
        if self.get_header(1).get("TUNIT6") == "e-/s":
            flux_err = Quantity(flux_err, unit="electron/s")

        return flux, flux_err, centroid_col, centroid_row

    def query_solar_system_objects(
        self,
        cadence_mask="outliers",
        radius=None,
        sigma=3,
        cache=True,
        return_mask=False,
        show_progress=True
    ):
        """Returns a list of asteroids or comets which affected the target pixel files.

        Light curves of stars or galaxies are frequently affected by solar
        system bodies (e.g. asteroids, comets, planets).  These objects can move
        across a target's photometric aperture mask on time scales of hours to
        days.  When they pass through a mask, they tend to cause a brief spike
        in the brightness of the target.  They can also cause dips by moving
        through a local background aperture mask (if any is used).

        The artifical spikes and dips introduced by asteroids are frequently
        confused with stellar flares, planet transits, etc.  This method helps
        to identify false signals injects by asteroids by providing a list of
        the solar system objects (name, brightness, time) that passed in the
        vicinity of the target during the span of the light curve.

        This method queries the `SkyBot API <http://vo.imcce.fr/webservices/skybot/>`_,
        which returns a list of asteroids/comets/planets given a location, time,
        and search cone.

        Notes
        -----
        * This method will use the `ra` and `dec` properties of the `LightCurve`
          object to determine the position of the search cone.
        * The size of the search cone is 5 spacecraft pixels + TPF dimension by default. You
          can change this by passing the `radius` parameter (unit: degrees).
        * By default, this method will only search points in time during which the light
          curve showed 3-sigma outliers in flux. You can override this behavior
          and search for specific times by passing `cadence_mask`. See examples for details.

        Parameters
        ----------
        cadence_mask : str, or boolean array with length of self.time
            mask in time to select which frames or points should be searched for SSOs.
            Default "outliers" will search for SSOs at points that are `sigma` from the mean.
            "all" will search all cadences. Alternatively, pass a boolean array with values of "True"
            for times to search for SSOs.
        radius : optional, float
            Radius to search for bodies. If None, will search for SSOs within 5 pixels of
            all pixels in the TPF.
        sigma : optional, float
            If `cadence_mask` is set to `"outlier"`, `sigma` will be used to identify
            outliers.
        cache : optional, bool
            If True will cache the search result in the astropy cache. Set to False
            to request the search again.
        return_mask: optional, bool
            If True will return a boolean mask in time alongside the result
        show_progress: optional, bool
            If True will display a progress bar during the download

        Returns
        -------
        result : pandas.DataFrame
            DataFrame containing the list objects in frames that were identified to contain
            SSOs.

        Examples
        --------
        Find if there are SSOs affecting the target pixel file for the given time frame:

            >>> df_sso = tpf.query_solar_system_objects(cadence_mask=(tpf.time.value >= 2014.1) & (tpf.time.value <= 2014.9))  # doctest: +SKIP

        Find if there are SSOs affecting the target pixel file for all times, but it will be much slower:

            >>> df_sso = tpf.query_solar_system_objects(cadence_mask='all')  # doctest: +SKIP

        """

        for attr in ["mission", "ra", "dec"]:
            if not hasattr(self, "{}".format(attr)):
                raise ValueError("Input does not have a `{}` attribute.".format(attr))

        location = self.mission.lower()

        if isinstance(cadence_mask, str):
            if cadence_mask == "outliers":
                aper = self.pipeline_mask
                if aper.sum() == 0:
                    aper = "all"
                lc = self.to_lightcurve(aperture_mask=aper)
                cadence_mask = lc.remove_outliers(sigma=sigma, return_mask=True)[1]
                # Avoid searching times with NaN flux; this is necessary because e.g.
                # `remove_outliers` includes NaNs in its mask.
                cadence_mask &= ~np.isnan(lc.flux)
            elif cadence_mask == "all":
                cadence_mask = np.ones(len(self.time)).astype(bool)
            else:
                raise ValueError("invalid `cadence_mask` string argument")
        elif isinstance(cadence_mask, collections.abc.Sequence):
            cadence_mask = np.array(cadence_mask)
        elif isinstance(cadence_mask, (bool)):
            # for boundary case of a single element tuple, e.g., (True)
            cadence_mask = np.array([cadence_mask])
        elif not isinstance(cadence_mask, np.ndarray):
            raise ValueError("Pass a cadence_mask method or a cadence_mask")

        if (location == "kepler") | (location == "k2"):
            pixel_scale = 4
        if location == "tess":
            pixel_scale = 27

        if radius == None:
            radius = (
                2 ** 0.5 * (pixel_scale * (np.max(self.shape[1:]) + 5))
            ) * u.arcsecond.to(u.deg)

        res = _query_solar_system_objects(
            ra=self.ra,
            dec=self.dec,
            times=self.time.jd[cadence_mask],
            location=location,
            radius=radius,
            cache=cache,
            show_progress=show_progress,
        )
        if return_mask:
            return res, np.in1d(self.time.jd, res.epoch)
        return res

    def plot(
        self,
        ax=None,
        frame=0,
        cadenceno=None,
        bkg=False,
        column="FLUX",
        aperture_mask=None,
        show_colorbar=True,
        mask_color="red",
        title=None,
        style="lightkurve",
        **kwargs,
    ):
        """Plot the pixel data for a single frame (i.e. at a single time).

        The time can be specified by frame index number (`frame=0` will show the
        first frame) or absolute cadence number (`cadenceno`).

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        frame : int
            Frame number. The default is 0, i.e. the first frame.
        cadenceno : int, optional
            Alternatively, a cadence number can be provided.
            This argument has priority over frame number.
        bkg : bool
            If True and `column="FLUX"`, background will be added to the pixel values.
        column : str
            Choose the FITS data column to be plotted. May be one of ('FLUX',
            'FLUX_ERR','FLUX_BKG','FLUX_BKG_ERR','COSMIC_RAYS','RAW_CNTS').
        aperture_mask : ndarray or str
            Highlight pixels selected by aperture_mask.
        show_colorbar : bool
            Whether or not to show the colorbar
        mask_color : str
            Color to show the aperture mask
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Keywords arguments passed to `~lightkurve.utils.plot_image`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if style == "lightkurve" or style is None:
            style = MPLSTYLE
        if cadenceno is not None:
            try:
                frame = np.argwhere(cadenceno == self.cadenceno)[0][0]
            except IndexError:
                raise ValueError(
                    "cadenceno {} is out of bounds, "
                    "must be in the range {}-{}.".format(
                        cadenceno, self.cadenceno[0], self.cadenceno[-1]
                    )
                )
        try:
            if column == "FLUX":
                if bkg and np.any(np.isfinite(self.flux_bkg[frame])):
                    data_to_plot = self.flux[frame] + self.flux_bkg[frame]
                else:
                    data_to_plot = self.flux[frame]
            else:
                data_to_plot = self.hdu[1].data[column][self.quality_mask][frame]
        except KeyError:
            raise ValueError(
                "column must be one of the following: ('FLUX','FLUX_ERR',"
                "'FLUX_BKG','FLUX_BKG_ERR','COSMIC_RAYS','RAW_CNTS')"
            )
        except IndexError:
            raise ValueError(
                "frame {} is out of bounds, must be in the range "
                "0-{}.".format(frame, self.shape[0])
            )

        # Make list of preset colour labels
        clabels = {
            "FLUX": "Flux ($e^{-}s^{-1}$)",
            "FLUX_ERR": "Flux Err. ($e^{-}s^{-1}$)",
            "FLUX_BKG": "Background Flux ($e^{-}s^{-1}$)",
            "FLUX_BKG_ERR": "Background Flux Err. ($e^{-}s^{-1}$)",
            "COSMIC_RAYS": "Cosmic Ray Flux ($e^{-}s^{-1}$)",
            "RAW_CNTS": "Raw Counts",
        }

        with plt.style.context(style):
            if title is None:
                title = "Target ID: {}, Cadence: {}".format(
                    self.targetid, self.cadenceno[frame]
                )

            # We subtract -0.5 because pixel coordinates refer to the middle of
            # a pixel, e.g. (col, row) = (10.0, 20.0) is a pixel center.
            img_extent = (
                self.column - 0.5,
                self.column + self.shape[2] - 0.5,
                self.row - 0.5,
                self.row + self.shape[1] - 0.5,
            )

            # If an axes is passed that used WCS projection, don't use img_extent
            # This addresses lk issue #1095, where the tpf coordinates were incorrectly plotted

            # By default ax=None
            if ax != None:
                if hasattr(ax, "wcs"):
                    img_extent = None



            ax = plot_image(
                data_to_plot,
                ax=ax,
                title=title,
                extent=img_extent,
                show_colorbar=show_colorbar,
                clabel=clabels.get(column, column),
                **kwargs,
            )
            ax.grid(False)


        # Overlay the aperture mask if given
        if aperture_mask is not None:
            aperture_mask = self._parse_aperture_mask(aperture_mask)
            in_aperture = np.where(aperture_mask)
            if hasattr(ax, "wcs"):
                ap_row = in_aperture[0] - 0.5
                ap_col = in_aperture[1] - 0.5
            else:
                ap_row = in_aperture[0] + self.row - 0.5
                ap_col = in_aperture[1] + self.column - 0.5    
            for ii in range(len(ap_row)):
                
                rect=patches.Rectangle((ap_col[ii],ap_row[ii]),1,1, fill=False, hatch="//", color=mask_color)
                ax.add_patch(rect)
                
        return ax

    def _to_matplotlib_animation(
        self, step: int = None, interval: int = 200, **plot_args
    ) -> "matplotlib.animation.FuncAnimation":
        """Returns a `matplotlib.animation.FuncAnimation` object.

        The animation shows the flux values over time by calling `tpf.plot()` for multiple frames.

        Parameters
        ----------
        step : int
            Spacing between frames.  By default, the spacing will be determined such that
            50 frames are shown, i.e. `step = len(tpf) // 50`.  Showing more than 50 frames
            will be slow on many systems.
        interval : int
            Delay between frames in milliseconds.
        **plot_args : dict
            Optional parameters passed to tpf.plot().
        """
        if step is None:
            step = len(self) // 50
            if step < 1:
                step = 1

        column = plot_args.get("column", "FLUX")
        ax = self.plot(**plot_args)

        def init():
            return ax.images

        def animate(i):
            frame = i * step
            ax.images[0].set_data(self.hdu[1].data[column][self.quality_mask][frame])
            ax.set_title(f"Frame {frame}")
            return ax.images

        plt.close(ax.figure)  # prevent figure from showing up in interactive mode

        # `blit=True` means only re-draw the parts that have changed.
        frames = len(self) // step
        anim = matplotlib.animation.FuncAnimation(
            ax.figure,
            animate,
            init_func=init,
            frames=frames,
            interval=interval,
            blit=True,
        )
        return anim

    def animate(self, step: int = None, interval: int = 200, **plot_args):
        """Displays an interactive HTML matplotlib animation.

        This feature requires a Jupyter notebook environment to display correctly.

        Parameters
        ----------
        step : int
            Spacing between frames.  By default, the spacing will be determined such that
            50 frames are shown, i.e. `step = len(tpf) // 50`.  Showing more than 50 frames
            will be slow on many systems.
        interval : int
            Delay between frames in milliseconds.
        **plot_args : dict
            Optional parameters passed to tpf.plot().
        """
        try:
            # To make installing Lightkurve easier, ipython is an optional dependency,
            # because we can assume it is installed when notebook-specific features are called
            from IPython.display import HTML
            return HTML(self._to_matplotlib_animation(step=step, interval=interval, **plot_args).to_jshtml())
        except ModuleNotFoundError:
            log.error("ipython needs to be installed for animate() to work (e.g., `pip install ipython`)")

    def to_fits(self, output_fn=None, overwrite=False):
        """Writes the TPF to a FITS file on disk."""
        if output_fn is None:
            output_fn = "{}-targ.fits".format(self.targetid)
        self.hdu.writeto(output_fn, overwrite=overwrite, checksum=True)

    def interact(
        self,
        notebook_url=None,
        max_cadences=200000,
        aperture_mask="default",
        exported_filename=None,
        transform_func=None,
        ylim_func=None,
        **kwargs,
    ):
        """Display an interactive Jupyter Notebook widget to inspect the pixel data.

        The widget will show both the lightcurve and pixel data.  By default,
        the lightcurve shown is obtained by calling the `to_lightcurve()` method,
        unless the user supplies a custom `LightCurve` object.
        This feature requires an optional dependency, bokeh (v0.12.15 or later).
        This dependency can be installed using e.g. `conda install bokeh`.

        At this time, this feature only works inside an active Jupyter
        Notebook, and tends to be too slow when more than ~30,000 cadences
        are contained in the TPF (e.g. short cadence data).

        Parameters
        ----------
        notebook_url : str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
            For use with JupyterHub, set the environment variable LK_JUPYTERHUB_EXTERNAL_URL
            to the public hostname of your JupyterHub and notebook_url will
            be defined appropriately automatically.
        max_cadences : int
            Print an error message if the number of cadences shown is larger than
            this value. This limit helps keep browsers from becoming unresponsive.
        aperture_mask : array-like, 'pipeline', 'threshold', 'default', or 'all'
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If None or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
            If 'default' is passed, 'pipeline' mask will be used when available,
            with 'threshold' as the fallback.
        exported_filename: str
            An optional filename to assign to exported fits files containing
            the custom aperture mask generated by clicking on pixels in interact.
            The default adds a suffix '-custom-aperture-mask.fits' to the
            TargetPixelFile basename.
        transform_func: function
            A function that transforms the lightcurve.  The function takes in a
            LightCurve object as input and returns a LightCurve object as output.
            The function can be complex, such as detrending the lightcurve.  In this
            way, the interactive selection of aperture mask can be evaluated after
            inspection of the transformed lightcurve.  The transform_func is applied
            before saving a fits file.  Default: None (no transform is applied).
        ylim_func: function
            A function that returns ylimits (low, high) given a LightCurve object.
            The default is to return a window approximately around 5 sigma-clipped
            lightcurve flux values.

        Examples
        --------
        To select an aperture mask for V827 Tau::

            >>> import lightkurve as lk
            >>> tpf = lk.search_targetpixelfile("V827 Tau", mission="K2").download()  # doctest: +SKIP
            >>> tpf.interact()  # doctest: +SKIP


        To see the full y-axis dynamic range of your lightcurve and normalize
        the lightcurve after each pixel selection::

            >>> ylim_func = lambda lc: (0.0, lc.flux.max())  # doctest: +SKIP
            >>> transform_func = lambda lc: lc.normalize()  # doctest: +SKIP
            >>> tpf.interact(ylim_func=ylim_func, transform_func=transform_func)  # doctest: +SKIP

        """
        from .interact import show_interact_widget

        notebook_url = finalize_notebook_url(notebook_url)

        return show_interact_widget(
            self,
            notebook_url=notebook_url,
            max_cadences=max_cadences,
            aperture_mask=aperture_mask,
            exported_filename=exported_filename,
            transform_func=transform_func,
            ylim_func=ylim_func,
            **kwargs,
        )


    def interact_sky(self, notebook_url=None, aperture_mask="empty", magnitude_limit=18):
        """Display a Jupyter Notebook widget showing Gaia DR2 positions on top of the pixels.

        Parameters
        ----------
        notebook_url : str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
            For use with JupyterHub, set the environment variable LK_JUPYTERHUB_EXTERNAL_URL
            to the public hostname of your JupyterHub and notebook_url will
            be defined appropriately automatically.
        aperture_mask : array-like, 'pipeline', 'threshold', 'default', 'background', or 'empty'
            Highlight pixels selected by aperture_mask.
            Default is 'empty': no pixel is highlighted.
        magnitude_limit : float
            A value to limit the results in based on Gaia Gmag. Default, 18.
        """
        from .interact import show_skyview_widget

        notebook_url = finalize_notebook_url(notebook_url)

        return show_skyview_widget(
            self, notebook_url=notebook_url, aperture_mask=aperture_mask, magnitude_limit=magnitude_limit
        )

    def to_corrector(self, method="pld", **kwargs):
        """Returns a `~correctors.corrector.Corrector` instance to remove systematics.

        Parameters
        ----------
        methods : string
            Currently, only "pld" is supported.  This will return a
            `~correctors.PLDCorrector` class instance.
        **kwargs : dict
            Extra keyword arguments to be passed on to the corrector class.

        Returns
        -------
        correcter : `~correctors.corrector.Corrector`
            Instance of a Corrector class, which typically provides `~correctors.PLDCorrector.correct()`
            and `~correctors.PLDCorrector.diagnose()` methods.
        """
        allowed_methods = ["pld"]
        if method == "sff":
            raise ValueError(
                "The 'sff' method requires a `LightCurve` instead "
                "of a `TargetPixelFile` object.  Use `to_lightcurve()` "
                "to obtain a `LightCurve` first."
            )
        if method not in allowed_methods:
            raise ValueError(
                ("Unrecognized method '{0}'\n" "allowed methods are: {1}").format(
                    method, allowed_methods
                )
            )
        if method == "pld":
            from .correctors import PLDCorrector

            return PLDCorrector(self, **kwargs)

    def cutout(self, center=None, size=5):
        """Cut a rectangle out of the Target Pixel File.

        This methods returns a new `TargetPixelFile` object containing a
        rectangle of a given ``size`` cut out around a given ``center``.

        Parameters
        ----------
        center : (int, int) tuple or `astropy.SkyCoord`
            Center of the cutout.  If an (int, int) tuple is passed, it will be
            interpreted as the (column, row) coordinates relative to
            the bottom-left corner of the TPF.  If an `astropy.SkyCoord` is
            passed then the sky coordinate will be used instead.
            If `None` (default) then the center of the TPF will be used.
        size : int or (int, int) tuple
            Number of pixels to cut out. If a single integer is passed then
            a square of that size will be cut. If a tuple is passed then a
            rectangle with dimensions (column_size, row_size) will be cut.

        Returns
        -------
        tpf : `lightkurve.TargetPixelFile` object
            New and smaller Target Pixel File object containing only the data
            cut out.
        """
        imshape = self.flux.shape[1:]

        # Parse the user input (``center``) into an (x, y) coordinate
        if center is None:
            x, y = imshape[0] // 2, imshape[1] // 2
        elif isinstance(center, SkyCoord):
            x, y = self.wcs.world_to_pixel(center)
        elif isinstance(center, (tuple, list, np.ndarray)):
            x, y = center
        col = int(x)
        row = int(y)

        # Parse the user input (``size``)
        if isinstance(size, int):
            s = (size / 2, size / 2)
        elif isinstance(size, (tuple, list, np.ndarray)):
            s = (size[0] / 2, size[1] / 2)

        # Find the TPF edges
        col_edges = np.asarray(
            [np.max([0, col - s[0]]), np.min([col + s[0], imshape[1]])], dtype=int
        )
        row_edges = np.asarray(
            [np.max([0, row - s[1]]), np.min([row + s[1], imshape[0]])], dtype=int
        )

        # Make a copy of the data extension
        hdu = self.hdu[0].copy()

        # Find the new object coordinates
        r, d = self.get_coordinates(cadence=len(self.flux) // 2)
        hdu.header["RA_OBJ"] = np.nanmean(
            r[row_edges[0] : row_edges[1], col_edges[0] : col_edges[1]]
        )
        hdu.header["DEC_OBJ"] = np.nanmean(
            d[row_edges[0] : row_edges[1], col_edges[0] : col_edges[1]]
        )

        # Remove any KIC labels
        labels = [
            "*MAG",
            "PM*",
            "GL*",
            "PARALLAX",
            "*COLOR",
            "TEFF",
            "LOGG",
            "FEH",
            "EBMINUSV",
            "AV",
            "RADIUS",
            "TMINDEX",
        ]
        for label in labels:
            if label in hdu.header:
                hdu.header[label] = fits.card.Undefined()

        # HDUList
        hdus = [hdu]

        # Copy the header
        hdr = deepcopy(self.hdu[1].header)

        # Trim any columns that have the shape of the image, to be the new shape
        data_columns = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for idx, datacol in enumerate(self.hdu[1].columns):
                # We exclude Kepler's obscure "RB_LEVEL" column from cutouts
                # for now because it has an awkward shape
                if datacol.name == "RB_LEVEL":
                    continue
                # If the column is 3D
                if len(self.hdu[1].data[datacol.name].shape) == 3:
                    # Make a copy, trim it and change the format
                    datacol = deepcopy(datacol)
                    datacol.array = datacol.array[
                        :, row_edges[0] : row_edges[1], col_edges[0] : col_edges[1]
                    ]
                    datacol._dim = "{}".format(datacol.array.shape[1:]).replace(" ", "")
                    datacol._dims = datacol.array.shape[1:]
                    datacol._format = fits.column._ColumnFormat(
                        "{}{}".format(
                            np.product(datacol.array.shape[1:]), datacol._format[-1]
                        )
                    )
                    data_columns.append(datacol)
                    hdr["TDIM{}".format(idx)] = "{}".format(
                        datacol.array.shape[1:]
                    ).replace(" ", "")
                    hdr["TDIM9"] = "{}".format(datacol.array.shape[1:]).replace(" ", "")
                else:
                    data_columns.append(datacol)

        # Get those coordinates sorted for the corner of the TPF and the WCS
        hdr["1CRV*P"] = hdr["1CRV4P"] + col_edges[0]
        hdr["2CRV*P"] = hdr["2CRV4P"] + row_edges[0]
        hdr["1CRPX*"] = hdr["1CRPX4"] - col_edges[0]
        hdr["2CRPX*"] = hdr["2CRPX4"] - row_edges[0]

        # Make a table for the data
        data_columns[-1]._dim = "{}".format(
            (0, int(data_columns[5]._dim.split(",")[1][:-1]))
        ).replace(" ", "")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            btbl = fits.BinTableHDU.from_columns(data_columns, header=hdr)

        # Append it to the hdulist
        hdus.append(btbl)

        # Correct the aperture mask
        hdu = self.hdu[2].copy()
        ar = hdu.data
        ar = ar[row_edges[0] : row_edges[1], col_edges[0] : col_edges[1]]
        hdu.header["NAXIS1"] = ar.shape[0]
        hdu.header["NAXIS2"] = ar.shape[1]
        hdu.data = ar
        hdus.append(hdu)

        # Make a new tpf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            newfits = fits.HDUList(hdus)
        return self.__class__(newfits, quality_bitmask=self.quality_bitmask)

    @staticmethod
    def from_fits_images(
        images_flux,
        position,
        images_raw_cnts=None,
        images_flux_err=None,
        images_flux_bkg=None,
        images_flux_bkg_err=None,
        images_cosmic_rays=None,
        size=(11, 11),
        extension=1,
        target_id="unnamed-target",
        hdu0_keywords=None,
        **kwargs,
    ):
        """Creates a new Target Pixel File from a set of images.

        This method is intended to make it easy to cut out targets from
        Kepler/K2 "superstamp" regions or TESS FFI images.

        Parameters
        ----------
        images_flux : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the flux data from.
        position : astropy.SkyCoord
            Position around which to cut out pixels.
        images_raw_cnts : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the raw counts data from.
        images_flux_err : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the flux error data from.
        images_flux_bkg : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the background data from.
        images_flux_bkg_err : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the background error data from.
        images_cosmic_rays : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the cosmic rays data from.
        size : (int, int)
            Dimensions (cols, rows) to cut out around `position`.
        extension : int or str
            If `images` is a list of filenames, provide the extension number
            or name to use. This should be the same for all flux inputs
            provided. Default: 1.
        target_id : int or str
            Unique identifier of the target to be recorded in the TPF.
        hdu0_keywords : dict
            Additional keywords to add to the first header file.
        **kwargs : dict
            Extra arguments to be passed to the `TargetPixelFile` constructor.

        Returns
        -------
        tpf : TargetPixelFile
            A new Target Pixel File assembled from the images.
        """
        len_images = len(images_flux)

        if len_images == 0:
            raise ValueError("One or more images must be passed.")
        if not isinstance(position, SkyCoord):
            raise ValueError("Position must be an astropy.coordinates.SkyCoord.")
        if hdu0_keywords is None:
            hdu0_keywords = {}

        basic_keywords = [
            "MISSION",
            "TELESCOP",
            "INSTRUME",
            "QUARTER",
            "CAMPAIGN",
            "CHANNEL",
            "MODULE",
            "OUTPUT",
            "CAMERA",
            "CCD",
            "SECTOR",
        ]
        carry_keywords = {}

        # Define a helper function to accept images in a flexible way
        def _open_image(img, extension):
            if isinstance(img, fits.ImageHDU):
                hdu = img
            elif isinstance(img, fits.HDUList):
                hdu = img[extension]
            else:
                with fits.open(img) as hdulist:
                    hdu = hdulist[extension].copy()
            return hdu

        # Define a helper function to cutout images if not None
        def _cutout_image(hdu, position, wcs_ref, size):
            if hdu is None:
                cutout_data = None
                cutout_wcs = None
            elif position is None:
                cutout_data = hdu.data
                cutout_wcs = hdu.wcs
            else:
                cutout = Cutout2D(
                    hdu.data, position, wcs=wcs_ref, size=size, mode="partial"
                )
                cutout_data = cutout.data
                cutout_wcs = cutout.wcs
            return cutout_data, cutout_wcs

        # Set the default extension if unspecified
        if extension is None:
            extension = 0
            if isinstance(images_flux[0], str) and images_flux[0].endswith("ffic.fits"):
                extension = 1  # TESS FFIs have the image data in extension #1

        # If no position is given, ensure the cut-out size matches the image size
        if position is None:
            size = _open_image(images_flux[0], extension).data.shape

        # Find middle image to use as a WCS reference
        try:
            mid_hdu = _open_image(images_flux[int(len_images / 2) - 1], extension)

            wcs_ref = WCS(mid_hdu)
            column, row = wcs_ref.all_world2pix(
                np.asarray([[position.ra.deg], [position.dec.deg]]).T, 0
            )[0]
        except Exception as e:
            raise e

        # Create a factory and set default keyword values based on the middle image
        factory = TargetPixelFileFactory(
            n_cadences=len_images, n_rows=size[0], n_cols=size[1], target_id=target_id
        )

        # Get some basic keywords
        for kw in basic_keywords:
            if kw in mid_hdu.header:
                if not isinstance(mid_hdu.header[kw], Undefined):
                    carry_keywords[kw] = mid_hdu.header[kw]
        if ("MISSION" not in carry_keywords) and ("TELESCOP" in carry_keywords):
            carry_keywords["MISSION"] = carry_keywords["TELESCOP"]

        allkeys = hdu0_keywords.copy()
        allkeys.update(carry_keywords)

        img_list = [
            images_raw_cnts,
            images_flux,
            images_flux_err,
            images_flux_bkg,
            images_flux_bkg_err,
            images_cosmic_rays,
        ]

        for idx, img in tqdm(enumerate(images_flux), total=len_images):
            # Open images if provided and get HDUs
            hdu_list = [
                _open_image(i[idx], extension) if i is not None else None
                for i in img_list
            ]

            # Use the header in the flux image for each frame
            hdu_idx = hdu_list[1].header

            if idx == 0:  # Get default keyword values from the first flux image
                factory.keywords = hdu_idx

            # Get positional shift of the image compared to the reference WCS
            wcs_current = WCS(hdu_idx)

            column_current, row_current = wcs_current.all_world2pix(
                np.asarray([[position.ra.deg], [position.dec.deg]]).T, 0
            )[0]
            column_ref, row_ref = wcs_ref.all_world2pix(
                np.asarray([[position.ra.deg], [position.dec.deg]]).T, 0
            )[0]

            with warnings.catch_warnings():
                # Using `POS_CORR1` as a header keyword violates the FITS
                # standard for being too long, but we use it for consistency
                # with the TPF column name.  Hence we ignore the warning.
                warnings.simplefilter("ignore", AstropyWarning)
                hdu_idx["POS_CORR1"] = column_current - column_ref
                hdu_idx["POS_CORR2"] = row_current - row_ref

            # Cutout (if neccessary) and get data
            cutout_list = [
                _cutout_image(hdu, position, wcs_ref, size) for hdu in hdu_list
            ]
            # Flatten output list
            cutout_list = [item for sublist in cutout_list for item in sublist]
            (
                raw_cnts,
                _,
                flux,
                wcs,
                flux_err,
                _,
                flux_bkg,
                _,
                flux_bkg_err,
                _,
                cosmic_rays,
                _,
            ) = cutout_list

            factory.add_cadence(
                frameno=idx,
                raw_cnts=raw_cnts,
                flux=flux,
                flux_err=flux_err,
                flux_bkg=flux_bkg,
                flux_bkg_err=flux_bkg_err,
                cosmic_rays=cosmic_rays,
                header=hdu_idx,
            )

        ext_info = {}
        ext_info["TFORM4"] = "{}J".format(size[0] * size[1])
        ext_info["TDIM4"] = "({},{})".format(size[0], size[1])
        ext_info.update(wcs.to_header(relax=True))

        # TPF contains multiple data columns that require WCS
        for m in [4, 5, 6, 7, 8, 9]:
            if m > 4:
                ext_info["TFORM{}".format(m)] = "{}E".format(size[0] * size[1])
                ext_info["TDIM{}".format(m)] = "({},{})".format(size[0], size[1])
            # Compute the distance from the star to the TPF lower left corner
            # That is approximately half the TPF size, with an adjustment factor if the star's pixel
            #    position gets rounded up or not.
            # The first int is there so that even sizes always round to one less than half of their value

            half_tpfsize_col = int((size[0] - 1) / 2.0) + (
                int(round(column)) - int(column)
            ) * ((size[0] + 1) % 2)
            half_tpfsize_row = int((size[1] - 1) / 2.0) + (
                int(round(row)) - int(row)
            ) * ((size[1] + 1) % 2)

            ext_info["1CRV{}P".format(m)] = (
                int(round(column)) - half_tpfsize_col + factory.keywords["CRVAL1P"] - 1
            )
            ext_info["2CRV{}P".format(m)] = (
                int(round(row)) - half_tpfsize_row + factory.keywords["CRVAL2P"] - 1
            )

        return factory.get_tpf(hdu0_keywords=allkeys, ext_info=ext_info, **kwargs)

    def plot_pixels(
        self,
        ax=None,
        periodogram=False,
        aperture_mask=None,
        show_flux=False,
        corrector_func=None,
        style="lightkurve",
        title=None,
        markersize=0.5,
        **kwargs,
    ):
        """Show the light curve of each pixel in a single plot.

        Note that all values are autoscaled and axis labels are not provided.
        This utility is designed for by-eye inspection of signal morphology.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        periodogram : bool
            Default: False; if True, periodograms will be plotted, using normalized light curves.
            Note that this keyword overrides normalized.
        aperture_mask : ndarray or str
            Highlight pixels selected by aperture_mask.
            Only `pipeline`, `threshold`, or custom masks will be plotted.
            `all` and None masks will be ignored.
        show_flux : bool
            Default: False; if True, shade pixels with frame 0 flux colour
            Inspired by https://github.com/noraeisner/LATTE
        corrector_func : function
            Function that accepts and returns a `~lightkurve.lightcurve.LightCurve`.
            This function is applied to each pixel's light curve.
            The default is to return a 5 sigma-clipped light curve.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        markersize : float
            Size of the markers in the lightcurve plot. For periodogram plot, it is used as the line width.
            Default: 0.5
        kwargs : dict
            e.g. extra parameters to be passed to `~lightkurve.LightCurve.to_periodogram`.

        Examples
        --------
        Inspect the lightcurve around a possible transit at per-pixel level::

            >>> import lightkurve as lk
            >>> # A possible transit around time BTJD 2277.0. Inspect the lightcurve around that time
            >>> tpf = tpf[(tpf.time.value >= 2276.5) & (tpf.time.value <= 2277.5)]    # doctest: +SKIP
            >>> tpf.plot_pixels(aperture_mask='pipeline')  # doctest: +SKIP
            >>>
            >>> # Variation: shade the pixel based on the flux at frame 0
            >>> # increase markersize so that it is more legible for pixels with yellow background (the brightest pixels)
            >>> tpf.plot_pixels(aperture_mask='pipeline', show_flux=True, markersize=1.5)  # doctest: +SKIP
            >>>
            >>> # Variation: Customize the plot's size so that each pixel is about 1 inch by 1 inch
            >>> import matplotlib.pyplot as plt
            >>> fig = plt.figure(figsize=(tpf.flux[0].shape[1] * 1.0, tpf.flux[0].shape[0] * 1.0))    # doctest: +SKIP
            >>> tpf.plot_pixels(ax=fig.gca(), aperture_mask='pipeline')    # doctest: +SKIP


        """
        if style == "lightkurve" or style is None:
            style = MPLSTYLE
        if title is None:
            title = "Target ID: {0}, {1:.2f} - {2:.2f} {3}".format(
                self.targetid,
                self.time[0].value,
                self.time[-1].value,
                _time_label_brief(self.time),
            )
        if corrector_func is None:
            corrector_func = lambda x: x.remove_outliers()
        if show_flux:
            cmap = plt.get_cmap()
            norm = plt.Normalize(
                vmin=np.nanmin(self.flux[0].value), vmax=np.nanmax(self.flux[0].value)
            )
        mask = self._parse_aperture_mask(aperture_mask)

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=(RuntimeWarning, LightkurveWarning)
            )

            # get an aperture mask for each pixel
            masks = np.zeros(
                (self.shape[1] * self.shape[2], self.shape[1], self.shape[2]),
                dtype="bool",
            )
            for i in range(self.shape[1] * self.shape[2]):
                masks[i][np.unravel_index(i, (self.shape[1], self.shape[2]))] = True

            pixel_list = []
            for j in range(self.shape[1] * self.shape[2]):
                lc = self.to_lightcurve(aperture_mask=masks[j])
                lc = corrector_func(lc)

                if periodogram:
                    try:
                        pixel_list.append(lc.to_periodogram(**kwargs))
                    except IndexError:
                        pixel_list.append(None)
                else:
                    if len(lc.remove_nans().flux) == 0:
                        pixel_list.append(None)
                    else:
                        pixel_list.append(lc)

        with plt.style.context(style):
            if ax is None:
                fig = plt.figure()
                ax = plt.gca()
                set_size = True
            else:
                fig = ax.get_figure()
                set_size = False

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if periodogram:
                ax.set(
                    title=title,
                    xlabel="Frequency / Column (pixel)",
                    ylabel="Power / Row (pixel)",
                )
            else:
                ax.set(
                    title=title,
                    xlabel="Time / Column (pixel)",
                    ylabel="Flux / Row (pixel)",
                )

            gs = gridspec.GridSpec(
                self.shape[1], self.shape[2], wspace=0.01, hspace=0.01
            )

            for k in range(self.shape[1] * self.shape[2]):
                if pixel_list[k]:
                    x, y = np.unravel_index(k, (self.shape[1], self.shape[2]))

                    # Highlight aperture mask in red
                    if aperture_mask is not None and mask[x, y]:
                        rc = {"axes.linewidth": 2, "axes.edgecolor": "red"}
                    else:
                        rc = {"axes.linewidth": 1}
                    with plt.rc_context(rc=rc):
                        gax = fig.add_subplot(gs[self.shape[1] - x - 1, y])

                    # Determine background and foreground color
                    if show_flux:
                        gax.set_facecolor(cmap(norm(self.flux.value[0, x, y])))
                        markercolor = "white"
                    else:
                        markercolor = "black"

                    # Plot flux or periodogram
                    if periodogram:
                        gax.plot(
                            pixel_list[k].frequency.value,
                            pixel_list[k].power.value,
                            marker="None",
                            color=markercolor,
                            lw=markersize,
                        )
                    else:
                        gax.plot(
                            pixel_list[k].time.value,
                            pixel_list[k].flux.value,
                            marker=".",
                            color=markercolor,
                            ms=markersize,
                            lw=0,
                        )

                    gax.margins(y=0.1, x=0)
                    gax.set_xticklabels("")
                    gax.set_yticklabels("")
                    gax.set_xticks([])
                    gax.set_yticks([])

                    # add row/column numbers to start / end
                    if x == 0 and y == 0:
                        gax.set_xlabel(f"{self.column}")
                        gax.set_ylabel(f"{self.row}")
                    if x == 0 and y == self.shape[2] - 1:  # lower right
                        gax.set_xlabel(f"{self.column + self.shape[2] - 1}")
                    if x == self.shape[1] - 1 and y == 0:  # upper left
                        gax.set_ylabel(f"{self.row + self.shape[1] - 1}")

            if set_size:  # use default size when caller does not supply ax
                fig.set_size_inches((y * 1.5, x * 1.5))

        return ax


class KeplerTargetPixelFile(TargetPixelFile):
    """Class to read and interact with the pixel data products
    ("Target Pixel Files") created by NASA's Kepler pipeline.

    This class offers a user-friendly way to open a Kepler Target Pixel File
    (TPF), access its meta data, visualize its contents, extract light curves
    with custom aperture masks, estimate centroid positions, and more.

    Please consult the `TargetPixelFile tutorial
    <https://docs.lightkurve.org/tutorials/01-target-pixel-files.html>`_
    in the online documentation for examples on using this class.

    Parameters
    ----------
    path : str or `~astropy.io.fits.HDUList`
        Path to a Kepler Target Pixel file. Alternatively, you can pass a
        `.HDUList` object, which is the AstroPy object returned by
        the `astropy.io.fits.open` function.
    quality_bitmask : "none", "default", "hard", "hardest", or int
        Bitmask that should be used to ignore bad-quality cadences.
        If a string is passed, it has the following meaning:

            * "none": no cadences will be ignored (equivalent to
              ``quality_bitmask=0``).
            * "default": cadences with severe quality issues will be ignored
              (equivalent to ``quality_bitmask=1130799``).
            * "hard": more conservative choice of flags to ignore
              (equivalent to ``quality_bitmask=1664431``).
              This is known to remove good data.
            * "hardest": remove all cadences that have one or more flags raised
              (equivalent to ``quality_bitmask=2096639``). This mask is not
              recommended because some quality flags can safely be ignored.

        If an integer is passed, it will be used as a bitmask, i.e. it will
        have the effect of removing cadences where
        ``(tpf.hdu[1].data['QUALITY'] & quality_bitmask) > 0``.
        See the :class:`KeplerQualityFlags` class for details on the bitmasks.
    **kwargs : dict
        Optional keyword arguments passed on to `astropy.io.fits.open`.

    References
    ----------
    .. [1] Kepler: A Search for Terrestrial Planets. Kepler Archive Manual.
        http://archive.stsci.edu/kepler/manuals/archive_manual.pdf
    """

    def __init__(self, path, quality_bitmask="default", **kwargs):
        super(KeplerTargetPixelFile, self).__init__(
            path, quality_bitmask=quality_bitmask, **kwargs
        )
        try:
            self.quality_mask = KeplerQualityFlags.create_quality_mask(
                quality_array=self.hdu[1].data["QUALITY"], bitmask=quality_bitmask
            )

            # check to make sure the correct filetype has been provided
            filetype = detect_filetype(self.hdu)
            if filetype == "TessTargetPixelFile":
                warnings.warn(
                    "A TESS data product is being opened using the "
                    "`KeplerTargetPixelFile` class. "
                    "Please use `TessTargetPixelFile` instead.",
                    LightkurveWarning,
                )
            elif filetype is None:
                warnings.warn(
                    "File header not recognized as Kepler or TESS " "observation.",
                    LightkurveWarning,
                )

            # Use the KEPLERID keyword as the default targetid
            if self.targetid is None:
                self.targetid = self.get_header().get("KEPLERID")
        except Exception as e:
            # Cannot instantiate TargetPixelFile, close the HDU to release the file handle
            self.hdu.close()
            raise e

    def __repr__(self):
        return "KeplerTargetPixelFile Object (ID: {})".format(self.targetid)

    def get_prf_model(self):
        """Returns an object of KeplerPRF initialized using the
        necessary metadata in the tpf object.

        Returns
        -------
        prf : instance of SimpleKeplerPRF
        """

        return KeplerPRF(
            channel=self.channel, shape=self.shape[1:], column=self.column, row=self.row
        )

    @property
    def obsmode(self):
        """'short cadence' or 'long cadence'. ('OBSMODE' header keyword)"""
        return self.get_keyword("OBSMODE")

    @property
    def module(self):
        """Kepler CCD module number. ('MODULE' header keyword)"""
        return self.get_keyword("MODULE")

    @property
    def output(self):
        """Kepler CCD module output number. ('OUTPUT' header keyword)"""
        return self.get_keyword("OUTPUT")

    @property
    def channel(self):
        """Kepler CCD channel number. ('CHANNEL' header keyword)"""
        return self.get_keyword("CHANNEL")

    @property
    def quarter(self):
        """Kepler quarter number. ('QUARTER' header keyword)"""
        return self.get_keyword("QUARTER")

    @property
    def campaign(self):
        """K2 Campaign number. ('CAMPAIGN' header keyword)"""
        return self.get_keyword("CAMPAIGN")

    @property
    def mission(self):
        """'Kepler' or 'K2'. ('MISSION' header keyword)"""
        return self.get_keyword("MISSION")

    def extract_aperture_photometry(
        self, aperture_mask="default", flux_method="sum", centroid_method="moments"
    ):
        """Returns a LightCurve obtained using aperture photometry.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'threshold', 'default', or 'all'
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If None or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
            If 'default' is passed, 'pipeline' mask will be used when available,
            with 'threshold' as the fallback.
        flux_method: 'sum', 'median', or 'mean'
            Determines how the pixel values within the aperture mask are combined
            at each cadence. Defaults to 'sum'.
        centroid_method : str, 'moments' or 'quadratic'
            For the details on this arguments, please refer to the documentation
            for `estimate_centroids()`.

        Returns
        -------
        lc : KeplerLightCurve object
            Array containing the summed flux within the aperture for each
            cadence.
        """
        # explicitly resolve default, so that the aperture_mask set in meta
        # later will be the resolved one
        aperture_mask = self._resolve_default_aperture_mask(aperture_mask)

        flux, flux_err, centroid_col, centroid_row = self._aperture_photometry(
            aperture_mask=aperture_mask,
            flux_method=flux_method,
            centroid_method=centroid_method,
        )
        keys = {
            "centroid_col": centroid_col,
            "centroid_row": centroid_row,
            "quality": self.quality,
            "channel": self.channel,
            "campaign": self.campaign,
            "quarter": self.quarter,
            "mission": self.mission,
            "cadenceno": self.cadenceno,
            "ra": self.ra,
            "dec": self.dec,
            "label": self.get_keyword("OBJECT", default=self.targetid),
            "targetid": self.targetid,
        }
        meta = {"APERTURE_MASK": aperture_mask}
        return KeplerLightCurve(
            time=self.time, flux=flux, flux_err=flux_err, **keys, meta=meta
        )

    def get_bkg_lightcurve(self, aperture_mask=None):
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        # Ignore warnings related to zero or negative errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_bkg_err = (
                np.nansum(self.flux_bkg_err[:, aperture_mask] ** 2, axis=1) ** 0.5
            )
        keys = {
            "quality": self.quality,
            "channel": self.channel,
            "campaign": self.campaign,
            "quarter": self.quarter,
            "mission": self.mission,
            "cadenceno": self.cadenceno,
            "ra": self.ra,
            "dec": self.dec,
            "label": self.get_keyword("OBJECT", default=self.targetid),
            "targetid": self.targetid,
        }
        return KeplerLightCurve(
            time=self.time,
            flux=np.nansum(self.flux_bkg[:, aperture_mask], axis=1),
            flux_err=flux_bkg_err,
            **keys,
        )

    def get_model(self, star_priors=None, **kwargs):
        """Returns a default `TPFModel` object for PRF fitting.

        The default model only includes one star and only allows its flux
        and position to change.  A different set of stars can be added using
        the `star_priors` parameter.

        Parameters
        ----------
        **kwargs : dict
            Arguments to be passed to the `TPFModel` constructor, e.g.
            `star_priors`.

        Returns
        -------
        model : TPFModel object
            Model with appropriate defaults for this Target Pixel File.
        """
        from .prf import TPFModel, StarPrior, BackgroundPrior
        from .prf import UniformPrior, GaussianPrior

        # Set up the model
        if "star_priors" not in kwargs:
            centr_col, centr_row = self.estimate_centroids()
            star_priors = [
                StarPrior(
                    col=GaussianPrior(
                        mean=np.nanmedian(centr_col.value),
                        var=np.nanstd(centr_col.value) ** 2,
                    ),
                    row=GaussianPrior(
                        mean=np.nanmedian(centr_row.value),
                        var=np.nanstd(centr_row.value) ** 2,
                    ),
                    flux=UniformPrior(
                        lb=0.5 * np.nanmax(self.flux[0].value),
                        ub=2 * np.nansum(self.flux[0].value) + 1e-10,
                    ),
                    targetid=self.targetid,
                )
            ]
            kwargs["star_priors"] = star_priors
        if "prfmodel" not in kwargs:
            kwargs["prfmodel"] = self.get_prf_model()
        if "background_prior" not in kwargs:
            if np.all(
                np.isnan(self.flux_bkg)
            ):  # If TargetPixelFile has no background flux data
                # Use the median of the lower half of flux as an estimate for flux_bkg
                clipped_flux = np.ma.masked_where(
                    self.flux.value > np.percentile(self.flux.value, 50),
                    self.flux.value,
                )
                flux_prior = GaussianPrior(
                    mean=np.ma.median(clipped_flux), var=np.ma.std(clipped_flux) ** 2
                )
            else:
                flux_prior = GaussianPrior(
                    mean=np.nanmedian(self.flux_bkg.value),
                    var=np.nanstd(self.flux_bkg.value) ** 2,
                )
            kwargs["background_prior"] = BackgroundPrior(flux=flux_prior)
        return TPFModel(**kwargs)

    def extract_prf_photometry(self, cadences=None, parallel=True, **kwargs):
        """Returns the results of PRF photometry applied to the pixel file.

        Parameters
        ----------
        cadences : list of int
            Cadences to fit.  If `None` (default) then all cadences will be fit.
        parallel : bool
            If `True`, fitting cadences will be distributed across multiple
            cores using Python's `multiprocessing` module.
        **kwargs : dict
            Keywords to be passed to `get_model()` to create the
            `~prf.TPFModel` object that will be fit.

        Returns
        -------
        results : PRFPhotometry object
            Object that provides access to PRF-fitting photometry results and
            various diagnostics.
        """
        from .prf import PRFPhotometry

        log.warning(
            "Warning: PRF-fitting photometry is experimental "
            "in this version of lightkurve."
        )
        prfphot = PRFPhotometry(model=self.get_model(**kwargs))
        prfphot.run(
            self.flux + self.flux_bkg,
            cadences=cadences,
            parallel=parallel,
            pos_corr1=self.pos_corr1,
            pos_corr2=self.pos_corr2,
        )
        return prfphot

    def prf_lightcurve(self, **kwargs):
        lc = self.extract_prf_photometry(**kwargs).lightcurves[0]
        keys = {
            "quality": self.quality,
            "channel": self.channel,
            "campaign": self.campaign,
            "quarter": self.quarter,
            "mission": self.mission,
            "cadenceno": self.cadenceno,
            "ra": self.ra,
            "dec": self.dec,
            "targetid": self.targetid,
        }
        return KeplerLightCurve(time=self.time, flux=lc.flux, **keys)


class FactoryError(Exception):
    """Raised if there is a problem creating a TPF."""

    pass


class TargetPixelFileFactory(object):
    """Class to create a TargetPixelFile."""

    def __init__(
        self, n_cadences, n_rows, n_cols, target_id="unnamed-target", keywords=None
    ):
        self.n_cadences = n_cadences
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.target_id = target_id
        if keywords is None:
            self.keywords = {}
        else:
            self.keywords = keywords

        # Initialize the 3D data structures
        self.raw_cnts = np.empty((n_cadences, n_rows, n_cols), dtype="int")
        self.flux = np.empty((n_cadences, n_rows, n_cols), dtype="float32")
        self.flux_err = np.empty((n_cadences, n_rows, n_cols), dtype="float32")
        self.flux_bkg = np.empty((n_cadences, n_rows, n_cols), dtype="float32")
        self.flux_bkg_err = np.empty((n_cadences, n_rows, n_cols), dtype="float32")
        self.cosmic_rays = np.empty((n_cadences, n_rows, n_cols), dtype="float32")

        # Set 3D data defaults
        self.raw_cnts[:, :, :] = -1
        self.flux[:, :, :] = np.nan
        self.flux_err[:, :, :] = np.nan
        self.flux_bkg[:, :, :] = np.nan
        self.flux_bkg_err[:, :, :] = np.nan
        self.cosmic_rays[:, :, :] = np.nan

        # Initialize the 1D data structures
        self.mjd = np.zeros(n_cadences, dtype="float64")
        self.time = np.zeros(n_cadences, dtype="float64")
        self.timecorr = np.zeros(n_cadences, dtype="float32")
        self.cadenceno = np.zeros(n_cadences, dtype="int")
        self.quality = np.zeros(n_cadences, dtype="int")
        self.pos_corr1 = np.zeros(n_cadences, dtype="float32")
        self.pos_corr2 = np.zeros(n_cadences, dtype="float32")

    def add_cadence(
        self,
        frameno,
        raw_cnts=None,
        flux=None,
        flux_err=None,
        flux_bkg=None,
        flux_bkg_err=None,
        cosmic_rays=None,
        header=None,
    ):
        """Populate the data for a single cadence."""
        if frameno >= self.n_cadences:
            raise FactoryError(
                "Can not add cadence {}, n_cadences set to {}".format(
                    frameno, self.n_cadences
                )
            )
        if header is None:
            header = {}

        # 2D-data
        for col in [
            "raw_cnts",
            "flux",
            "flux_err",
            "flux_bkg",
            "flux_bkg_err",
            "cosmic_rays",
        ]:
            if locals()[col] is not None:
                if locals()[col].shape != (self.n_rows, self.n_cols):
                    raise FactoryError(
                        "Can not add cadence with a different shape ({} x {})".format(
                            self.n_rows, self.n_cols
                        )
                    )

                vars(self)[col][frameno] = locals()[col]

        # 1D-data
        if "TSTART" in header and "TSTOP" in header:
            self.time[frameno] = (header["TSTART"] + header["TSTOP"]) / 2.0
        if "TIMECORR" in header:
            self.timecorr[frameno] = header["TIMECORR"]
        if "CADENCEN" in header:
            self.cadenceno[frameno] = header["CADENCEN"]
        if "QUALITY" in header:
            self.quality[frameno] = header["QUALITY"]
        if "POS_CORR1" in header:
            self.pos_corr1[frameno] = header["POS_CORR1"]
        if "POS_CORR2" in header:
            self.pos_corr2[frameno] = header["POS_CORR2"]

    def _check_data(self):
        """Check the data before writing to a TPF for any obvious errors."""
        if len(self.time) != len(np.unique(self.time)):
            warnings.warn(
                "The factory-created TPF contains cadences with "
                "identical TIME values.",
                LightkurveWarning,
            )
        if ~np.all(self.time == np.sort(self.time)):
            warnings.warn(
                "Cadences in the factory-created TPF do not appear "
                "to be sorted in chronological order.",
                LightkurveWarning,
            )
        if np.nansum(self.flux) == 0:
            warnings.warn(
                "The factory-created TPF does not appear to contain "
                "non-zero flux values.",
                LightkurveWarning,
            )

    def get_tpf(self, hdu0_keywords=None, ext_info=None, **kwargs):
        """Returns a TargetPixelFile object."""
        if hdu0_keywords is None:
            hdu0_keywords = {}
        if ext_info is None:
            ext_info = {}
        self._check_data()
        # Detect filetype
        hdulist = self._hdulist(hdu0_keywords=hdu0_keywords, ext_info=ext_info)
        filetype = detect_filetype(hdulist)
        if filetype == "TessTargetPixelFile":
            tpf = TessTargetPixelFile(hdulist, **kwargs)
        elif filetype == "KeplerTargetPixelFile":
            tpf = KeplerTargetPixelFile(hdulist, **kwargs)
        else:
            warnings.warn(
                "Could not detect filetype as TESSTargetPixelFile or KeplerTargetPixelFile, "
                "returning generic TargetPixelFile instead.",
                LightkurveWarning,
            )
            tpf = TargetPixelFile(hdulist, **kwargs)
        return tpf

    def _hdulist(self, hdu0_keywords, ext_info):
        """Returns an astropy.io.fits.HDUList object."""
        return fits.HDUList(
            [
                self._make_primary_hdu(hdu0_keywords=hdu0_keywords),
                self._make_target_extension(ext_info=ext_info),
                self._make_aperture_extension(),
            ]
        )

    def _header_template(self, extension):
        """Returns a template `fits.Header` object for a given extension."""
        template_fn = os.path.join(
            PACKAGEDIR, "data", "tpf-ext{}-header.txt".format(extension)
        )
        return fits.Header.fromtextfile(template_fn)

    def _make_primary_hdu(self, hdu0_keywords):
        """Returns the primary extension (#0)."""
        hdu = fits.PrimaryHDU()
        # Copy the default keywords from a template file from the MAST archive
        tmpl = self._header_template(0)
        for kw in tmpl:
            hdu.header[kw] = (tmpl[kw], tmpl.comments[kw])
        # Override the defaults where necessary
        hdu.header["ORIGIN"] = "Unofficial data product"
        hdu.header["DATE"] = datetime.datetime.now().strftime("%Y-%m-%d")
        hdu.header["CREATOR"] = "lightkurve.TargetPixelFileFactory"
        hdu.header["OBJECT"] = self.target_id
        if hdu.header["TELESCOP"] is not None and hdu.header["TELESCOP"] == "Kepler":
            hdu.header["KEPLERID"] = self.target_id
        # Empty a bunch of keywords rather than having incorrect info
        for kw in [
            "PROCVER",
            "FILEVER",
            "CHANNEL",
            "MODULE",
            "OUTPUT",
            "TIMVERSN",
            "CAMPAIGN",
            "DATA_REL",
            "TTABLEID",
            "RA_OBJ",
            "DEC_OBJ",
        ]:
            hdu.header[kw] = ""

        # Some keywords just shouldn't be passed to the new header.
        bad_keys = [
            "ORIGIN",
            "DATE",
            "OBJECT",
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "EXTEND",
            "NEXTEND",
            "EXTNAME",
            "NAXIS1",
            "NAXIS2",
            "QUALITY",
        ]
        for kw, val in hdu0_keywords.items():
            if kw in bad_keys:
                continue
            if kw in hdu.header:
                hdu.header[kw] = val
            else:
                hdu.header.append((kw, val))
        return hdu

    def _make_target_extension(self, ext_info):
        """Create the 'TARGETTABLES' extension (i.e. extension #1)."""
        # Turn the data arrays into fits columns and initialize the HDU
        coldim = "({},{})".format(self.n_cols, self.n_rows)
        eformat = "{}E".format(self.n_rows * self.n_cols)
        jformat = "{}J".format(self.n_rows * self.n_cols)
        cols = []
        cols.append(
            fits.Column(name="TIME", format="D", unit="BJD - 2454833", array=self.time)
        )
        cols.append(
            fits.Column(name="TIMECORR", format="E", unit="D", array=self.timecorr)
        )
        cols.append(fits.Column(name="CADENCENO", format="J", array=self.cadenceno))
        cols.append(
            fits.Column(
                name="RAW_CNTS",
                format=jformat,
                unit="count",
                dim=coldim,
                array=self.raw_cnts,
            )
        )
        cols.append(
            fits.Column(
                name="FLUX", format=eformat, unit="e-/s", dim=coldim, array=self.flux
            )
        )
        cols.append(
            fits.Column(
                name="FLUX_ERR",
                format=eformat,
                unit="e-/s",
                dim=coldim,
                array=self.flux_err,
            )
        )
        cols.append(
            fits.Column(
                name="FLUX_BKG",
                format=eformat,
                unit="e-/s",
                dim=coldim,
                array=self.flux_bkg,
            )
        )
        cols.append(
            fits.Column(
                name="FLUX_BKG_ERR",
                format=eformat,
                unit="e-/s",
                dim=coldim,
                array=self.flux_bkg_err,
            )
        )
        cols.append(
            fits.Column(
                name="COSMIC_RAYS",
                format=eformat,
                unit="e-/s",
                dim=coldim,
                array=self.cosmic_rays,
            )
        )
        cols.append(fits.Column(name="QUALITY", format="J", array=self.quality))
        cols.append(
            fits.Column(
                name="POS_CORR1", format="E", unit="pixels", array=self.pos_corr1
            )
        )
        cols.append(
            fits.Column(
                name="POS_CORR2", format="E", unit="pixels", array=self.pos_corr2
            )
        )
        coldefs = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(coldefs)

        # Set the header with defaults
        template = self._header_template(1)
        for kw in template:
            if kw not in ["XTENSION", "NAXIS1", "NAXIS2", "CHECKSUM", "BITPIX"]:
                try:
                    hdu.header[kw] = (self.keywords[kw], self.keywords.comments[kw])
                except KeyError:
                    hdu.header[kw] = (template[kw], template.comments[kw])
        wcs_keywords = {
            "CTYPE1": "1CTYP{}",
            "CTYPE2": "2CTYP{}",
            "CRPIX1": "1CRPX{}",
            "CRPIX2": "2CRPX{}",
            "CRVAL1": "1CRVL{}",
            "CRVAL2": "2CRVL{}",
            "CUNIT1": "1CUNI{}",
            "CUNIT2": "2CUNI{}",
            "CDELT1": "1CDLT{}",
            "CDELT2": "2CDLT{}",
            "PC1_1": "11PC{}",
            "PC1_2": "12PC{}",
            "PC2_1": "21PC{}",
            "PC2_2": "22PC{}",
        }
        # Override defaults using data calculated in from_fits_images
        for kw in ext_info.keys():
            if kw in wcs_keywords.keys():
                for x in [4, 5, 6, 7, 8, 9]:
                    hdu.header[wcs_keywords[kw].format(x)] = ext_info[kw]
            else:
                hdu.header[kw] = ext_info[kw]
        return hdu

    def _make_aperture_extension(self):
        """Create the aperture mask extension (i.e. extension #2)."""
        mask = 3 * np.ones((self.n_rows, self.n_cols), dtype="int32")
        hdu = fits.ImageHDU(mask)

        # Set the header from the template TPF again
        template = self._header_template(2)
        for kw in template:
            if kw not in ["XTENSION", "NAXIS1", "NAXIS2", "CHECKSUM", "BITPIX"]:
                try:
                    hdu.header[kw] = (self.keywords[kw], self.keywords.comments[kw])
                except KeyError:
                    hdu.header[kw] = (template[kw], template.comments[kw])

        # Override the defaults where necessary
        for keyword in [
            "CTYPE1",
            "CTYPE2",
            "CRPIX1",
            "CRPIX2",
            "CRVAL1",
            "CRVAL2",
            "CUNIT1",
            "CUNIT2",
            "CDELT1",
            "CDELT2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
        ]:
            hdu.header[keyword] = ""  # override wcs keywords
        hdu.header["EXTNAME"] = "APERTURE"
        return hdu


class TessTargetPixelFile(TargetPixelFile):
    """Represents pixel data products created by NASA's TESS pipeline.

    This class enables extraction of custom light curves and centroid positions.

    Parameters
    ----------
    path : str
        Path to a TESS Target Pixel (FITS) File.
    quality_bitmask : "none", "default", "hard", "hardest", or int
        Bitmask that should be used to ignore bad-quality cadences.
        If a string is passed, it has the following meaning:

            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with severe quality issues will be ignored
              (`quality_bitmask=175`).
            * "hard": more conservative choice of flags to ignore
              (`quality_bitmask=7407`). This is known to remove good data.
            * "hardest": removes all data that has been flagged
              (`quality_bitmask=8191`). This mask is not recommended.

        If an integer is passed, it will be used as a bitmask, i.e. it will
        have the effect of removing cadences where
        ``(tpf.hdu[1].data['QUALITY'] & quality_bitmask) > 0``.
        See the :class:`KeplerQualityFlags` class for details on the bitmasks.
    kwargs : dict
        Keyword arguments passed to `astropy.io.fits.open()`.
    """

    def __init__(self, path, quality_bitmask="default", **kwargs):
        super(TessTargetPixelFile, self).__init__(
            path, quality_bitmask=quality_bitmask, **kwargs
        )
        try:
            self.quality_mask = TessQualityFlags.create_quality_mask(
                quality_array=self.hdu[1].data["QUALITY"], bitmask=quality_bitmask
            )
            # Early TESS releases had cadences with time=NaN (i.e. missing data)
            # which were not flagged by a QUALITY flag yet; the line below prevents
            # these cadences from being used. They would break most methods!
            if (quality_bitmask != 0) and (quality_bitmask != "none"):
                self.quality_mask &= np.isfinite(self.hdu[1].data["TIME"])

            # check to make sure the correct filetype has been provided
            filetype = detect_filetype(self.hdu)
            if filetype == "KeplerTargetPixelFile":
                warnings.warn(
                    "A Kepler data product is being opened using the "
                    "`TessTargetPixelFile` class. "
                    "Please use `KeplerTargetPixelFile` instead.",
                    LightkurveWarning,
                )
            elif filetype is None:
                warnings.warn(
                    "File header not recognized as Kepler or TESS " "observation.",
                    LightkurveWarning,
                )

            # Use the TICID keyword as the default targetid
            if self.targetid is None:
                self.targetid = self.get_header().get("TICID")
        except Exception as e:
            # Cannot instantiate TargetPixelFile, close the HDU to release the file handle
            self.hdu.close()
            raise e

    def __repr__(self):
        return "TessTargetPixelFile(TICID: {})".format(self.targetid)

    @property
    def background_mask(self):
        """Returns the background mask used by the TESS pipeline."""
        # The TESS pipeline flags the pixels in the background aperture using
        # bit number 4, cf. Section 6 of the TESS Data Products documentation
        # (EXP-TESS-ARC-ICD-TM-0014.pdf).
        try:
            return self.hdu[2].data & 4 > 0
        except (IndexError, TypeError):
            # `IndexError` may be raised if the aperture extension (#2) is missing
            # `TypeError` may be raised because early versions of TESScut returned floats in HDU 2
            return np.zeros(self.hdu[1].data["FLUX"][0].shape, dtype=bool)

    @property
    def sector(self):
        """TESS Sector number ('SECTOR' header keyword)."""
        return self.get_keyword("SECTOR")

    @property
    def camera(self):
        """TESS Camera number ('CAMERA' header keyword)."""
        return self.get_keyword("CAMERA")

    @property
    def ccd(self):
        """TESS CCD number ('CCD' header keyword)."""
        return self.get_keyword("CCD")

    @property
    def mission(self):
        return "TESS"

    def extract_aperture_photometry(
        self, aperture_mask="default", flux_method="sum", centroid_method="moments"
    ):
        """Returns a LightCurve obtained using aperture photometry.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'threshold', 'default', or 'all'
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If None or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
            If 'default' is passed, 'pipeline' mask will be used when available,
            with 'threshold' as the fallback.
        flux_method: 'sum', 'median', or 'mean'
            Determines how the pixel values within the aperture mask are combined
            at each cadence. Defaults to 'sum'.
        centroid_method : str, 'moments' or 'quadratic'
            For the details on this arguments, please refer to the documentation
            for `estimate_centroids()`.

        Returns
        -------
        lc : TessLightCurve object
            Contains the summed flux within the aperture for each cadence.
        """
        # explicitly resolve default, so that the aperture_mask set in meta
        # later will be the resolved one
        aperture_mask = self._resolve_default_aperture_mask(aperture_mask)

        flux, flux_err, centroid_col, centroid_row = self._aperture_photometry(
            aperture_mask=aperture_mask,
            flux_method=flux_method,
            centroid_method=centroid_method,
        )
        keys = {
            "centroid_col": centroid_col,
            "centroid_row": centroid_row,
            "quality": self.quality,
            "sector": self.sector,
            "camera": self.camera,
            "ccd": self.ccd,
            "mission": self.mission,
            "cadenceno": self.cadenceno,
            "ra": self.ra,
            "dec": self.dec,
            "label": self.get_keyword("OBJECT", default=self.targetid),
            "targetid": self.targetid,
        }
        meta = {"APERTURE_MASK": aperture_mask}
        return TessLightCurve(
            time=self.time, flux=flux, flux_err=flux_err, **keys, meta=meta
        )

    def get_bkg_lightcurve(self, aperture_mask=None):
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        # Ignore warnings related to zero or negative errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_bkg_err = (
                np.nansum(self.flux_bkg_err[:, aperture_mask] ** 2, axis=1) ** 0.5
            )
        keys = {
            "quality": self.quality,
            "sector": self.sector,
            "camera": self.camera,
            "ccd": self.ccd,
            "cadenceno": self.cadenceno,
            "ra": self.ra,
            "dec": self.dec,
            "label": self.get_keyword("OBJECT", default=self.targetid),
            "targetid": self.targetid,
        }
        return TessLightCurve(
            time=self.time,
            flux=np.nansum(self.flux_bkg[:, aperture_mask], axis=1),
            flux_err=flux_bkg_err,
            **keys,
        )
