"""Defines LightCurve, KeplerLightCurve, and TessLightCurve."""
import os
import datetime
import logging
import warnings
import collections
from collections.abc import Sequence

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib
from matplotlib import pyplot as plt
from copy import deepcopy

from astropy.table import Table, Column, MaskedColumn
from astropy.io import fits
from astropy.time import TimeBase, Time, TimeDelta
from astropy import units as u
from astropy.units import Quantity
from astropy.timeseries import TimeSeries, aggregate_downsample
from astropy.table import vstack
from astropy.stats import calculate_bin_edges
from astropy.utils.decorators import deprecated, deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked

from . import PACKAGEDIR, MPLSTYLE
from .utils import (
    running_mean,
    bkjd_to_astropy_time,
    btjd_to_astropy_time,
    validate_method,
    _query_solar_system_objects,
    finalize_notebook_url
)
from .utils import LightkurveWarning, LightkurveDeprecationWarning


__all__ = ["LightCurve", "KeplerLightCurve", "TessLightCurve", "FoldedLightCurve"]

log = logging.getLogger(__name__)

_HAS_VAR_BINS = 'time_bin_end' in aggregate_downsample.__kwdefaults__

def _to_unitless_day(data):
    if isinstance(data, Quantity):
        return data.to(u.day).value
    elif not np.isscalar(data):
        return np.asarray([_to_unitless_day(item) for item in data]).flatten()
    else:
        return data


def _is_dict_like(data1):
    return hasattr(data1, "keys") and callable(getattr(data1, "keys"))


def _is_list_like(data1):
    # https://stackoverflow.com/a/37842328
    return isinstance(data1, Sequence) and not isinstance(data1, str)


def _is_np_structured_array(data1):
    return isinstance(data1, np.ndarray) and data1.dtype.names is not None


class LightCurve(TimeSeries):
    """
    Subclass of AstroPy `~astropy.table.Table` guaranteed to have *time*, *flux*, and *flux_err* columns.

    Compared to the generic `~astropy.timeseries.TimeSeries` class, `LightCurve`
    ensures that each object has `time`, `flux`, and `flux_err` columns.
    These three columns are special for two reasons:
    1. they are the key columns upon which all light curve operations operate;
    2. they are always present (though they may be populated with ``NaN`` values).

    `LightCurve` objects also provide user-friendly attribute access to
    columns and meta data.

    Parameters
    ----------
    data : numpy ndarray, dict, list, `~astropy.table.Table`, or table-like object, optional
        Data to initialize time series. This does not need to contain the times
        or fluxes, which can be provided separately, but if it does contain the
        times and fluxes they should be in columns called ``'time'``,
        ``'flux'``, and ``'flux_err'`` to be automatically recognized.
    time : `~astropy.time.Time` or iterable
        Time values.  They can either be given directly as a
        `~astropy.time.Time` array or as any iterable that initializes the
        `~astropy.time.Time` class.
    flux : `~astropy.units.Quantity` or iterable
        Flux values for every time point.
    flux_err : `~astropy.units.Quantity` or iterable
        Uncertainty on each flux data point.
    **kwargs : dict
        Additional keyword arguments are passed to `~astropy.table.QTable`.

    Attributes
    ----------
    meta : `dict`
        meta data associated with the lightcurve. The header of the underlying FITS file (if applicable)
        is store in this dictionary. By convention, keys in this dictionary are usually in uppercase.

    Notes
    -----
    *Attribute access*: You can access a column or a ``meta`` value directly as an attribute.

    >>> lc.flux    # shortcut for lc['flux']   # doctest: +SKIP
    >>> lc.sector  # shortcut for lc.meta['SECTOR']   # doctest: +SKIP
    >>> lc.flux = lc.flux * 1.05  # update the values of a column.   # doctest: +SKIP

    In case the given name is both a column name and a key in ``meta``, the column will be returned.

    Note that you *cannot* create a new column using the attribute interface. If you do so,
    a new attribute is created instead, and a warning is raised.

    If you do create such attributes on purpose, please note that the attributes are not carried
    over when the lightcurve object is copied, or a new lightcurve object is derived
    based on a copy, e.g., ``normalize()``.


    Examples
    --------
    >>> import lightkurve as lk
    >>> lc = lk.LightCurve(time=[1, 2, 3, 4], flux=[0.98, 1.02, 1.03, 0.97])
    >>> lc.time
    <Time object: scale='tdb' format='jd' value=[1. 2. 3. 4.]>
    >>> lc.flux
    <Quantity [0.98, 1.02, 1.03, 0.97]>
    >>> lc.bin(time_bin_size=2, time_bin_start=0.5).flux
    <Quantity [1., 1.]>
    """

    # The constructor of the `TimeSeries` base class will enforce the presence
    # of these columns:
    _required_columns = ["time", "flux", "flux_err"]

    # The following keywords were removed in Lightkurve v2.0.
    # Their use will trigger a warning.
    _deprecated_keywords = (
        "targetid",
        "label",
        "time_format",
        "time_scale",
        "flux_unit",
    )
    _deprecated_column_keywords = [
        "centroid_col",
        "centroid_row",
        "cadenceno",
        "quality",
    ]

    # If an iterable is passed for ``time``, we will initialize an AstroPy
    # ``Time`` object using the following format and scale:
    _default_time_format = "jd"
    _default_time_scale = "tdb"

    # To emulate pandas, we do not support creating new columns or meta data
    # fields via attribute assignment, and raise a warning in __setattr__ when
    # a new attribute is created.  We need to relax this warning during the
    # initial construction of the object using `_new_attributes_relax`.
    _new_attributes_relax = True

    # cf. issue #925
    __array_priority__ = 100_000

    def __init__(self, data=None, *args, time=None, flux=None, flux_err=None, **kwargs):

        # the ` {has,get,set}_time_in_data()`: helpers to handle `data` of different types
        # in some cases, they also need to access kwargs["names"] as well

        def get_time_idx_in(names):
            time_indices = np.argwhere(np.asarray(names) == "time")
            if len(time_indices) > 0:
                return time_indices[0][0]
            else:
                return None

        def get_time_in_data_list():
            if len(data) < 1:
                return None
            names = kwargs.get("names")
            if names is None:
                # the first item MUST be time if no names specified
                if isinstance(data[0], TimeBase):  # Time or TimeDelta
                    return data[0]
                else:
                    return None
            else:
                time_idx = get_time_idx_in(names)
                if time_idx is not None:
                    return data[time_idx]
                else:
                    return None

        def set_time_in_data_list(value):
            if len(data) < 1:
                raise AssertionError("data should be non-empty")
            names = kwargs.get("names")
            if names is None:
                # the first item MUST be time if no names specified
                # this is to support base Table's select columns
                # in __getitem__()
                # https://github.com/astropy/astropy/blob/326435449ad8d859f1abf36800c3fb88d49c27ea/astropy/table/table.py#L1888
                data[0] = value
            else:
                time_idx = get_time_idx_in(names)
                if time_idx is not None:
                    data[time_idx] = value
                else:
                    raise AssertionError("data should have time column")

        def get_time_in_data_np_structured_array():
            if data.dtype.names is None:  # no labeled filed, not a structured array
                return None
            if "time" not in data.dtype.names:
                return None
            return data["time"]

        def remove_time_from_data_np_structured_array():
            if data.dtype.names is None:
                raise AssertionError("data should be a numpy structured array")
            if "time" not in data.dtype.names:
                raise AssertionError("data should have a time field")
            filtered_names = [n for n in data.dtype.names if n != "time"]
            return data[filtered_names]

        def has_time_in_data():
            """Check if the data has a column with the name"""
            if data is None:
                return False
            elif _is_dict_like(data):
                # data is a dict-like object with keys
                return "time" in data.keys()
            elif _is_list_like(data):
                # case data is a list-like object (a list of columns, etc.)
                return get_time_in_data_list() is not None
            elif _is_np_structured_array(data):
                # case numpy structured array (supported by base TimeSeries)
                # https://numpy.org/doc/stable/user/basics.rec.html
                return get_time_in_data_np_structured_array() is not None
            else:
                raise ValueError(f"Unsupported type for time in data: {type(data)}")

        def get_time_in_data():
            if _is_dict_like(data):
                # data is a dict-like object with keys
                return data["time"]
            elif _is_list_like(data):
                return get_time_in_data_list()
            elif _is_np_structured_array(data):
                return get_time_in_data_np_structured_array()
            else:
                # should never reach here. It'd have been caught by `has_time_in()``
                raise AssertionError("Unsupported type for time in data")

        def set_time_in_data(value):
            if _is_dict_like(data):
                # data is a dict-like object with keys
                data["time"] = value
            elif _is_list_like(data):
                set_time_in_data_list(value)
            elif _is_np_structured_array(data):
                # astropy Time cannot be assigned to a column in np structured array
                # we have special codepath handling it outside this function
                raise AssertionError("Setting Time instances to np structured array is not supported")
            else:
                # should never reach here. It'd have been caught by `has_time_in()``
                raise AssertionError("Unsupported type for time in data")

        # Delay checking for required columns until the end
        self._required_columns_relax = True

        # Lightkurve v1.x supported passing time, flux, and flux_err as
        # positional arguments. We support it here for backwards compatibility.
        if len(args) in [1, 2]:
            warnings.warn(
                "passing flux as a positional argument is deprecated"
                ", please use ``flux=...`` instead.",
                LightkurveDeprecationWarning,
            )
            time = data
            flux = args[0]
            data = None
        if len(args) == 2:
            flux_err = args[1]

        # For backwards compatibility with Lightkurve v1.x,
        # we support passing deprecated keywords via **kwargs.
        deprecated_kws = {}
        for kw in self._deprecated_keywords:
            if kw in kwargs:
                deprecated_kws[kw] = kwargs.pop(kw)

        deprecated_column_kws = {}
        for kw in self._deprecated_column_keywords:
            if kw in kwargs:
                deprecated_column_kws[kw] = kwargs.pop(kw)

        # If `time` is passed as keyword argument, we populate it with integer numbers
        if data is None or not has_time_in_data():
            if time is None and flux is not None:
                time = np.arange(len(flux))
            # We are tolerant of missing time format
            if time is not None and not isinstance(time, (Time, TimeDelta)):
                # Lightkurve v1.x supported specifying the time_format
                # as a constructor kwarg
                time = Time(
                    time,
                    format=deprecated_kws.get("time_format", self._default_time_format),
                    scale=deprecated_kws.get("time_scale", self._default_time_scale),
                )

        # Also be tolerant of missing time format if time is passed via `data`
        if data is not None and has_time_in_data():
            if not isinstance(get_time_in_data(), (Time, TimeDelta)):
                tmp_time = Time(
                    get_time_in_data(),
                    format=deprecated_kws.get("time_format", self._default_time_format),
                    scale=deprecated_kws.get("time_scale", self._default_time_scale),
                )
                if _is_np_structured_array(data):
                    # special case for np structured array
                    # one cannot set a `Time` instance to it
                    # so we set the time to the `time` param, and take it out of data
                    time = tmp_time
                    data = remove_time_from_data_np_structured_array()
                else:
                    set_time_in_data(tmp_time)

        # Allow overriding the required columns
        self._required_columns = kwargs.pop("_required_columns", self._required_columns)

        # Call the SampledTimeSeries constructor.
        # Disable required columns for now; we'll check those later.
        tmp = self._required_columns
        self._required_columns = []
        super().__init__(data=data, time=time, **kwargs)
        self._required_columns = tmp

        # For some operations, an empty time series needs to be created, then
        # columns added one by one. We should check that when columns are added
        # manually, time is added first and is of the right type.
        if data is None and time is None and flux is None and flux_err is None:
            self._required_columns_relax = True
            return

        # Load `time`, `flux`, and `flux_err` from the table as local variable names
        time = self.columns["time"]  # super().__init__() guarantees this is a column
        if "flux" in self.colnames:
            if flux is None:
                flux = self.columns["flux"]
            else:
                raise TypeError(
                    f"'flux' has been given both in the `data` table and as a keyword argument"
                )
        if "flux_err" in self.colnames:
            if flux_err is None:
                flux_err = self.columns["flux_err"]
            else:
                raise TypeError(
                    f"'flux_err' has been given both in the `data` table and as a keyword argument"
                )

        # Ensure `flux` and `flux_err` are populated with NaNs if missing
        if flux is None and time is not None:
            flux = np.empty(len(time))
            flux[:] = np.nan
        if not isinstance(flux, Quantity):
            flux = Quantity(flux, deprecated_kws.get("flux_unit"))

        if flux_err is None:
            flux_err = np.empty(len(flux))
            flux_err[:] = np.nan
        if not isinstance(flux_err, Quantity):
            flux_err = Quantity(flux_err, flux.unit)

        # Backwards compatibility with Lightkurve v1.x
        # Ensure attributes are set if passed via deprecated kwargs
        for kw in deprecated_kws:
            if kw not in self.meta:
                self.meta[kw.upper()] = deprecated_kws[kw]

        # Ensure all required columns are in the right order
        with self._delay_required_column_checks():
            for idx, col in enumerate(self._required_columns):
                if col in self.colnames:
                    self.remove_column(col)
                self.add_column(locals()[col], index=idx, name=col)

        # Ensure columns are set if passed via deprecated kwargs
        for kw in deprecated_column_kws:
            if kw not in self.meta and kw not in self.columns:
                self.add_column(deprecated_column_kws[kw], name=kw)

        # Ensure flux and flux_err have the same units
        if self["flux"].unit != self["flux_err"].unit:
            raise ValueError("flux and flux_err must have the same units")

        self._new_attributes_relax = False
        self._required_columns_relax = False
        self._check_required_columns()

    def __getattr__(self, name, **kwargs):
        """Expose all columns and meta keywords as attributes."""
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.__class__.__dict__:
            return self.__class__.__dict__[name].__get__(self)
        elif name in self.columns:
            return self[name]
        elif "_meta" in self.__dict__:
            if name in self.__dict__["_meta"]:
                return self.__dict__["_meta"][name]
            elif name.upper() in self.__dict__["_meta"]:
                return self.__dict__["_meta"][name.upper()]
        raise AttributeError(f"object has no attribute {name}")

    def __setattr__(self, name, value, **kwargs):
        """To get copied, attributes have to be stored in the meta dictionary!"""
        to_set_as_attr = False
        if name in self.__dict__:
            to_set_as_attr = True
        elif name == "time":
            self["time"] = value  # astropy will convert value to Time if needed
        elif ("columns" in self.__dict__) and (name in self.__dict__["columns"]):
            self.replace_column(name, value)
        elif "_meta" in self.__dict__:
            if name in self.__dict__["_meta"]:
                self.__dict__["_meta"][name] = value
            elif name.upper() in self.__dict__["_meta"]:
                self.__dict__["_meta"][name.upper()] = value
            else:
                to_set_as_attr = True
        else:
            to_set_as_attr = True
        if to_set_as_attr:
            if (
                name not in self.__dict__
                and not name.startswith("_")
                and not self._new_attributes_relax
                and name != 'meta'
            ):
                warnings.warn(
                    (
                        "Lightkurve doesn't allow columns or meta values to be created via a new attribute name."
                        "A new attribute is created. It will not be carried over when the object is copied."
                        " - see https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.html"
                    ),
                    UserWarning,
                    stacklevel=2,
                )
            super().__setattr__(name, value, **kwargs)

    def _repr_simple_(self) -> str:
        """Returns a simple __repr__.

        Used by `LightCurveCollection`.
        """
        result = f"<{self.__class__.__name__}"
        if "LABEL" in self.meta:
            result += f" LABEL=\"{self.meta.get('LABEL')}\""
        for kw in ["QUARTER", "CAMPAIGN", "SECTOR", "AUTHOR", "FLUX_ORIGIN"]:
            if kw in self.meta:
                result += f" {kw}={self.meta.get(kw)}"
        result += ">"
        return result

    def _base_repr_(self, html=False, descr_vals=None, **kwargs):
        """Defines the description shown by `__repr__` and `_html_repr_`."""
        if descr_vals is None:
            descr_vals = [self.__class__.__name__]
            if self.masked:
                descr_vals.append("masked=True")
            descr_vals.append("length={}".format(len(self)))
            if "LABEL" in self.meta:
                descr_vals.append(f"LABEL=\"{self.meta.get('LABEL')}\"")
            for kw in ["QUARTER", "CAMPAIGN", "SECTOR", "AUTHOR", "FLUX_ORIGIN"]:
                if kw in self.meta:
                    descr_vals.append(f"{kw}={self.meta.get(kw)}")
        return super()._base_repr_(html=html, descr_vals=descr_vals, **kwargs)

    # Define `time`, `flux`, `flux_err` as class attributes to enable IDE
    # of these required columns auto-completion.

    @property
    def time(self) -> Time:
        """Time values stored as an AstroPy `~astropy.time.Time` object."""
        return self["time"]

    @time.setter
    def time(self, time):
        self["time"] = time

    @property
    def flux(self) -> Quantity:
        """Brightness values stored as an AstroPy `~astropy.units.Quantity` object."""
        return self["flux"]

    @flux.setter
    def flux(self, flux):
        self["flux"] = flux

    @property
    def flux_err(self) -> Quantity:
        """Brightness uncertainties stored as an AstroPy `~astropy.units.Quantity` object."""
        return self["flux_err"]

    @flux_err.setter
    def flux_err(self, flux_err):
        self["flux_err"] = flux_err

    def select_flux(self, flux_column, flux_err_column=None):
        """Assign a different column to be the flux column.

        This method returns a copy of the LightCurve in which the ``flux``
        and ``flux_err`` columns have been replaced by the values contained
        in a different column.

        Parameters
        ----------
        flux_column : str
            Name of the column that should become the 'flux' column.
        flux_err_column : str or `None`
            Name of the column that should become the 'flux_err' column.
            By default, the column will be used that is obtained by adding the
            suffix "_err" to the value of ``flux_column``.  If such a
            column does not exist, ``flux_err`` will be populated with NaN values.

        Returns
        -------
        lc : LightCurve
            Copy of the ``LightCurve`` object with the new flux values assigned.

        Examples
        --------
        You can use this function to change the flux data on which most Lightkurve
        features operate.  For example, to view a periodogram based on the "sap_flux"
        column in a TESS light curve, use::

            >>> lc.select_flux("sap_flux").to_periodogram("lombscargle").plot()  # doctest: +SKIP
        """
        # Input validation
        if flux_column not in self.columns:
            raise ValueError(f"'{flux_column}' is not a column")
        if flux_err_column and flux_err_column not in self.columns:
            raise ValueError(f"'{flux_err_column}' is not a column")

        lc = self.copy()
        lc["flux"] = lc[flux_column]
        if flux_err_column:  # not None
            lc["flux_err"] = lc[flux_err_column]
        else:
            # if `flux_err_column` is unspecified, we attempt to use
            # f"{flux_column}_err" if it exists
            flux_err_column = f"{flux_column}_err"
            if flux_err_column in lc.columns:
                lc["flux_err"] = lc[flux_err_column]
            else:
                lc["flux_err"][:] = np.nan

        lc.meta['FLUX_ORIGIN'] = flux_column
        normalized_new_flux = lc["flux"].unit is None or lc["flux"].unit is u.dimensionless_unscaled
        # Note: here we assume unitless flux means it's normalized
        # it's not exactly true in many constructed lightcurves in unit test
        # but the assumption should hold for any real world use cases, e.g. TESS QLP
        if normalized_new_flux:
            lc.meta["NORMALIZED"] = normalized_new_flux
        else:
            # remove it altogether.
            # Setting to False would suffice;
            # but in typical non-normalized LC, the header will not be there at all.
            lc.meta.pop("NORMALIZED", None)
        return lc

    # Define deprecated attributes for compatibility with Lightkurve v1.x:

    @property
    @deprecated(
        "2.0", alternative="time.format", warning_type=LightkurveDeprecationWarning
    )
    def time_format(self):
        return self.time.format

    @property
    @deprecated(
        "2.0", alternative="time.scale", warning_type=LightkurveDeprecationWarning
    )
    def time_scale(self):
        return self.time.scale

    @property
    @deprecated("2.0", alternative="time", warning_type=LightkurveDeprecationWarning)
    def astropy_time(self):
        return self.time

    @property
    @deprecated(
        "2.0", alternative="flux.unit", warning_type=LightkurveDeprecationWarning
    )
    def flux_unit(self):
        return self.flux.unit

    @property
    @deprecated("2.0", alternative="flux", warning_type=LightkurveDeprecationWarning)
    def flux_quantity(self):
        return self.flux

    @property
    @deprecated(
        "2.0",
        alternative="fits.open(lc.filename)",
        warning_type=LightkurveDeprecationWarning,
    )
    def hdu(self):
        return fits.open(self.filename)

    @property
    @deprecated("2.0", warning_type=LightkurveDeprecationWarning)
    def SAP_FLUX(self):
        """A copy of the light curve in which `lc.flux = lc.sap_flux`
        and `lc.flux_err = lc.sap_flux_err`.  It is provided for backwards-
        compatibility with Lightkurve v1.x and will be removed soon."""
        lc = self.copy()
        lc["flux"] = lc["sap_flux"]
        lc["flux_err"] = lc["sap_flux_err"]
        return lc

    @property
    @deprecated("2.0", warning_type=LightkurveDeprecationWarning)
    def PDCSAP_FLUX(self):
        """A copy of the light curve in which `lc.flux = lc.pdcsap_flux`
        and `lc.flux_err = lc.pdcsap_flux_err`.  It is provided for backwards-
        compatibility with Lightkurve v1.x and will be removed soon."""
        lc = self.copy()
        lc["flux"] = lc["pdcsap_flux"]
        lc["flux_err"] = lc["pdcsap_flux_err"]
        return lc

    def __add__(self, other):
        newlc = self.copy()
        if isinstance(other, LightCurve):
            if len(self) != len(other):
                raise ValueError(
                    "Cannot add LightCurve objects because "
                    "they do not have equal length ({} vs {})."
                    "".format(len(self), len(other))
                )
            if np.any(self.time != other.time):
                warnings.warn(
                    "Two LightCurve objects with inconsistent time "
                    "values are being added.",
                    LightkurveWarning,
                )
            newlc.flux = self.flux + other.flux
            newlc.flux_err = np.hypot(self.flux_err, other.flux_err)
        else:
            newlc.flux = self.flux + other
        return newlc

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        newlc = self.copy()
        if isinstance(other, LightCurve):
            if len(self) != len(other):
                raise ValueError(
                    "Cannot multiply LightCurve objects because "
                    "they do not have equal length ({} vs {})."
                    "".format(len(self), len(other))
                )
            if np.any(self.time != other.time):
                warnings.warn(
                    "Two LightCurve objects with inconsistent time "
                    "values are being multiplied.",
                    LightkurveWarning,
                )
            newlc.flux = self.flux * other.flux
            # Applying standard uncertainty propagation, cf.
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
            newlc.flux_err = abs(newlc.flux) * np.hypot(
                self.flux_err / self.flux, other.flux_err / other.flux
            )
        elif isinstance(
            other, (u.UnitBase, u.FunctionUnitBase)
        ):  # cf. astropy/issues/6517
            newlc.flux = other * self.flux
            newlc.flux_err = other * self.flux_err
        else:
            newlc.flux = other * self.flux
            newlc.flux_err = abs(other) * self.flux_err
        return newlc

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __rtruediv__(self, other):
        newlc = self.copy()
        if isinstance(other, LightCurve):
            if len(self) != len(other):
                raise ValueError(
                    "Cannot divide LightCurve objects because "
                    "they do not have equal length ({} vs {})."
                    "".format(len(self), len(other))
                )
            if np.any(self.time != other.time):
                warnings.warn(
                    "Two LightCurve objects with inconsistent time "
                    "values are being divided.",
                    LightkurveWarning,
                )
            newlc.flux = other.flux / self.flux
            newlc.flux_err = abs(newlc.flux) * np.hypot(
                self.flux_err / self.flux, other.flux_err / other.flux
            )
        else:
            newlc.flux = other / self.flux
            newlc.flux_err = abs((other * self.flux_err) / (self.flux ** 2))
        return newlc

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def show_properties(self):
        """Prints a description of all non-callable attributes.

        Prints in order of type (ints, strings, lists, arrays, others).
        """
        attrs = {}
        deprecated_properties = list(self._deprecated_keywords)
        deprecated_properties += [
            "flux_quantity",
            "SAP_FLUX",
            "PDCSAP_FLUX",
            "astropy_time",
            "hdu",
        ]
        for attr in dir(self):
            if not attr.startswith("_") and attr not in deprecated_properties:
                try:
                    res = getattr(self, attr)
                except Exception:
                    continue
                if callable(res):
                    continue
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

    def append(self, others, inplace=False):
        """Append one or more other `LightCurve` object(s) to this one.

        Parameters
        ----------
        others : `LightCurve`, or list of `LightCurve`
            Light curve(s) to be appended to the current one.
        inplace : bool
            If True, change the current `LightCurve` instance in place instead
            of creating and returning a new one. Defaults to False.

        Returns
        -------
        new_lc : `LightCurve`
            Light curve which has the other light curves appened to it.
        """
        if inplace:
            raise ValueError(
                "the `inplace` parameter is no longer supported "
                "as of Lightkurve v2.0"
            )
        if not hasattr(others, "__iter__"):
            others = (others,)

        # Re-use LightCurveCollection.stitch() to avoid code duplication
        from .collections import LightCurveCollection  # avoid circular import

        return LightCurveCollection((self, *others)).stitch(corrector_func=None)

    def flatten(
        self,
        window_length=101,
        polyorder=2,
        return_trend=False,
        break_tolerance=5,
        niters=3,
        sigma=3,
        mask=None,
        **kwargs,
    ):
        """Removes the low frequency trend using scipy's Savitzky-Golay filter.

        This method wraps `scipy.signal.savgol_filter`.

        Parameters
        ----------
        window_length : int
            The length of the filter window (i.e. the number of coefficients).
            ``window_length`` must be a positive odd integer.
        polyorder : int
            The order of the polynomial used to fit the samples. ``polyorder``
            must be less than window_length.
        return_trend : bool
            If `True`, the method will return a tuple of two elements
            (flattened_lc, trend_lc) where trend_lc is the removed trend.
        break_tolerance : int
            If there are large gaps in time, flatten will split the flux into
            several sub-lightcurves and apply `savgol_filter` to each
            individually. A gap is defined as a period in time larger than
            `break_tolerance` times the median gap.  To disable this feature,
            set `break_tolerance` to None.
        niters : int
            Number of iterations to iteratively sigma clip and flatten. If more than one, will
            perform the flatten several times, removing outliers each time.
        sigma : int
            Number of sigma above which to remove outliers from the flatten
        mask : boolean array with length of self.time
            Boolean array to mask data with before flattening. Flux values where
            mask is True will not be used to flatten the data. An interpolated
            result will be provided for these points. Use this mask to remove
            data you want to preserve, e.g. transits.
        **kwargs : dict
            Dictionary of arguments to be passed to `scipy.signal.savgol_filter`.

        Returns
        -------
        flatten_lc : `LightCurve`
            New light curve object with long-term trends removed.
        If ``return_trend`` is set to ``True``, this method will also return:
        trend_lc : `LightCurve`
            New light curve object containing the trend that was removed.
        """
        if mask is None:
            mask = np.ones(len(self.time), dtype=bool)
        else:
            # Deep copy ensures we don't change the original.
            mask = deepcopy(~mask)
        # Add NaNs & outliers to the mask
        extra_mask = np.isfinite(self.flux)
        extra_mask &= np.nan_to_num(np.abs(self.flux - np.nanmedian(self.flux))) <= (
            np.nanstd(self.flux) * sigma
        )
        # In astropy>=5.0, extra_mask is a masked array
        if hasattr(extra_mask, 'mask'):
            mask &= extra_mask.filled(False)
        else:  # support astropy<5.0
            mask &= extra_mask

        for iter in np.arange(0, niters):
            if break_tolerance is None:
                break_tolerance = np.nan
            if polyorder >= window_length:
                polyorder = window_length - 1
                log.warning(
                    "polyorder must be smaller than window_length, "
                    "using polyorder={}.".format(polyorder)
                )
            # Split the lightcurve into segments by finding large gaps in time
            dt = self.time.value[mask][1:] - self.time.value[mask][0:-1]
            with warnings.catch_warnings():  # Ignore warnings due to NaNs
                warnings.simplefilter("ignore", RuntimeWarning)
                cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
            low = np.append([0], cut)
            high = np.append(cut, len(self.time[mask]))
            # Then, apply the savgol_filter to each segment separately
            trend_signal = Quantity(np.zeros(len(self.time[mask])), unit=self.flux.unit)
            for l, h in zip(low, high):
                # Reduce `window_length` and `polyorder` for short segments;
                # this prevents `savgol_filter` from raising an exception
                # If the segment is too short, just take the median
                if np.any([window_length > (h - l), (h - l) < break_tolerance]):
                    trend_signal[l:h] = np.nanmedian(self.flux[mask][l:h])
                else:
                    # Scipy outputs a warning here that is not useful, will be fixed in version 1.2
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        trsig = savgol_filter(
                            x=self.flux.value[mask][l:h],
                            window_length=window_length,
                            polyorder=polyorder,
                            **kwargs,
                        )
                        trend_signal[l:h] = Quantity(trsig, trend_signal.unit)
            # Ignore outliers; note we add `1e-14` below to avoid detecting
            # outliers which are merely caused by numerical noise.
            mask1 = np.nan_to_num(np.abs(self.flux[mask] - trend_signal)) < (
                np.nanstd(self.flux[mask] - trend_signal) * sigma
                + Quantity(1e-14, self.flux.unit)
            )
            f = interp1d(
                self.time.value[mask][mask1],
                trend_signal[mask1],
                fill_value="extrapolate",
            )
            trend_signal = Quantity(f(self.time.value), self.flux.unit)
            # In astropy>=5.0, mask1 is a masked array
            if hasattr(mask1, 'mask'):
                mask[mask] &= mask1.filled(False)
            else:  # support astropy<5.0
                mask[mask] &= mask1

        flatten_lc = self.copy()
        with warnings.catch_warnings():
            # ignore invalid division warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            flatten_lc.flux = flatten_lc.flux / trend_signal
            flatten_lc.flux_err = flatten_lc.flux_err / trend_signal

        flatten_lc.meta["NORMALIZED"] = True

        if return_trend:
            trend_lc = self.copy()
            trend_lc.flux = trend_signal
            return flatten_lc, trend_lc
        return flatten_lc

    @deprecated_renamed_argument(
        "transit_midpoint",
        "epoch_time",
        "2.0",
        warning_type=LightkurveDeprecationWarning,
    )
    @deprecated_renamed_argument(
        "t0", "epoch_time", "2.0", warning_type=LightkurveDeprecationWarning
    )
    def fold(
        self,
        period=None,
        epoch_time=None,
        epoch_phase=0,
        wrap_phase=None,
        normalize_phase=False,
    ):
        """Returns a `FoldedLightCurve` object folded on a period and epoch.

        This method is identical to AstroPy's `~astropy.timeseries.TimeSeries.fold()`
        method, except it returns a `FoldedLightCurve` object which offers
        convenient plotting methods.

        Parameters
        ----------
        period : float `~astropy.units.Quantity`
            The period to use for folding.  If a ``float`` is passed we'll
            assume it is in units of days.
        epoch_time : `~astropy.time.Time`
            The time to use as the reference epoch, at which the relative time
            offset / phase will be ``epoch_phase``. Defaults to the first time
            in the time series.
        epoch_phase : float or `~astropy.units.Quantity`
            Phase of ``epoch_time``. If ``normalize_phase`` is `True`, this
            should be a dimensionless value, while if ``normalize_phase`` is
            ``False``, this should be a `~astropy.units.Quantity` with time
            units. Defaults to 0.
        wrap_phase : float or `~astropy.units.Quantity`
            The value of the phase above which values are wrapped back by one
            period. If ``normalize_phase`` is `True`, this should be a
            dimensionless value, while if ``normalize_phase`` is ``False``,
            this should be a `~astropy.units.Quantity` with time units.
            Defaults to half the period, so that the resulting time series goes
            from ``-period / 2`` to ``period / 2`` (if ``normalize_phase`` is
            `False`) or -0.5 to 0.5 (if ``normalize_phase`` is `True`).
        normalize_phase : bool
            If `False` phase is returned as `~astropy.time.TimeDelta`,
            otherwise as a dimensionless `~astropy.units.Quantity`.

        Returns
        -------
        folded_lightcurve : `FoldedLightCurve`
            The folded light curve object in which the ``time`` column
            holds the phase values.
        """
        # Lightkurve v1.x assumed that `period` was given in days if no unit
        # was specified. We maintain this behavior for backwards-compatibility.
        if period is not None and not isinstance(period, Quantity):
            period *= u.day
        if epoch_time is not None and not isinstance(epoch_time, Time):
            epoch_time = Time(
                epoch_time, format=self.time.format, scale=self.time.scale
            )
        if (
            epoch_phase is not None
            and not isinstance(epoch_phase, Quantity)
            and not normalize_phase
        ):
            epoch_phase *= u.day
        if wrap_phase is not None and not isinstance(wrap_phase, Quantity):
            wrap_phase *= u.day

        # Warn if `epoch_time` appears to use the wrong format
        if epoch_time is not None and epoch_time.value > 2450000:
            if self.time.format == "bkjd":
                warnings.warn(
                    "`epoch_time` appears to be given in JD, "
                    "however the light curve time uses BKJD "
                    "(i.e. JD - 2454833).",
                    LightkurveWarning,
                )
            elif self.time.format == "btjd":
                warnings.warn(
                    "`epoch_time` appears to be given in JD, "
                    "however the light curve time uses BTJD "
                    "(i.e. JD - 2457000).",
                    LightkurveWarning,
                )

        ts = super().fold(
            period=period,
            epoch_time=epoch_time,
            epoch_phase=epoch_phase,
            wrap_phase=wrap_phase,
            normalize_phase=normalize_phase,
        )

        # The folded time would pass the `TimeSeries` validation check if
        # `normalize_phase=True`, so creating a `FoldedLightCurve` object
        # requires the following three-step workaround:
        # 1. Give the folded light curve a valid time column again
        with ts._delay_required_column_checks():
            folded_time = ts.time.copy()
            ts.remove_column("time")
            ts.add_column(self.time, name="time", index=0)
        # 2. Create the folded object
        lc = FoldedLightCurve(data=ts)
        # 3. Restore the folded time
        with lc._delay_required_column_checks():
            lc.remove_column("time")
            lc.add_column(folded_time, name="time", index=0)

        # Add extra column and meta data specific to FoldedLightCurve
        lc.add_column(
            self.time.copy(), name="time_original", index=len(self._required_columns)
        )
        lc.meta["PERIOD"] = period
        lc.meta["EPOCH_TIME"] = epoch_time
        lc.meta["EPOCH_PHASE"] = epoch_phase
        lc.meta["WRAP_PHASE"] = wrap_phase
        lc.meta["NORMALIZE_PHASE"] = normalize_phase
        lc.sort("time")

        return lc

    def normalize(self, unit="unscaled"):
        """Returns a normalized version of the light curve.

        The normalized light curve is obtained by dividing the ``flux`` and
        ``flux_err`` object attributes by the median flux.
        Optionally, the result will be multiplied by 1e2 (if `unit='percent'`),
        1e3 (`unit='ppt'`), or 1e6 (`unit='ppm'`).

        Parameters
        ----------
        unit : 'unscaled', 'percent', 'ppt', 'ppm'
            The desired relative units of the normalized light curve;
            'ppt' means 'parts per thousand', 'ppm' means 'parts per million'.

        Examples
        --------
            >>> import lightkurve as lk
            >>> lc = lk.LightCurve(time=[1, 2, 3], flux=[25945.7, 25901.5, 25931.2], flux_err=[6.8, 4.6, 6.2])
            >>> normalized_lc = lc.normalize()
            >>> normalized_lc.flux
            <Quantity [1.00055917, 0.99885466, 1.        ]>
            >>> normalized_lc.flux_err
            <Quantity [0.00026223, 0.00017739, 0.00023909]>

        Returns
        -------
        normalized_lightcurve : `LightCurve`
            A new light curve object in which ``flux`` and ``flux_err`` have
            been divided by the median flux.

        Warns
        -----
        LightkurveWarning
            If the median flux is negative or within half a standard deviation
            from zero.
        """
        validate_method(unit, ["unscaled", "percent", "ppt", "ppm"])
        median_flux = np.nanmedian(self.flux)
        std_flux = np.nanstd(self.flux)

        # If the median flux is within half a standard deviation from zero, the
        # light curve is likely zero-centered and normalization makes no sense.
        if (median_flux == 0) or (
            np.isfinite(std_flux) and (np.abs(median_flux) < 0.5 * std_flux)
        ):
            warnings.warn(
                "The light curve appears to be zero-centered "
                "(median={:.2e} +/- {:.2e}); `normalize()` will divide "
                "the light curve by a value close to zero, which is "
                "probably not what you want."
                "".format(median_flux, std_flux),
                LightkurveWarning,
            )
        # If the median flux is negative, normalization will invert the light
        # curve and makes no sense.
        if median_flux < 0:
            warnings.warn(
                "The light curve has a negative median flux ({:.2e});"
                " `normalize()` will therefore divide by a negative "
                "number and invert the light curve, which is probably"
                "not what you want".format(median_flux),
                LightkurveWarning,
            )

        # Create a new light curve instance and normalize its values
        lc = self.copy()
        lc.flux = lc.flux / median_flux
        lc.flux_err = lc.flux_err / median_flux
        if not lc.flux.unit:
            lc.flux *= u.dimensionless_unscaled
        if not lc.flux_err.unit:
            lc.flux_err *= u.dimensionless_unscaled

        # Set the desired relative (dimensionless) units
        if unit == "percent":
            lc.flux = lc.flux.to(u.percent)
            lc.flux_err = lc.flux_err.to(u.percent)
        elif unit in ("ppt", "ppm"):
            lc.flux = lc.flux.to(unit)
            lc.flux_err = lc.flux_err.to(unit)

        lc.meta["NORMALIZED"] = True
        return lc

    def remove_nans(self, column: str = "flux"):
        """Removes cadences where ``column`` is a NaN.

        Parameters
        ----------
        column : str
            Column to check for NaNs.  Defaults to ``'flux'``.

        Returns
        -------
        clean_lightcurve : `LightCurve`
            A new light curve object from which NaNs fluxes have been removed.

        Examples
        --------
            >>> import lightkurve as lk
            >>> import numpy as np
            >>> lc = lk.LightCurve({'time': [1, 2, 3], 'flux': [1., np.nan, 1.]})
            >>> lc.remove_nans()
            <LightCurve length=2>
            time   flux  flux_err
            <BLANKLINE>
            Time float64 float64
            ---- ------- --------
            1.0     1.0      nan
            3.0     1.0      nan
        """
        return self[~np.isnan(self[column])]  # This will return a sliced copy

    def fill_gaps(self, method: str = "gaussian_noise"):
        r"""Fill in gaps in time.

        By default, the gaps will be filled with random white Gaussian noise
        distributed according to
        :math:`\mathcal{N} (\mu=\overline{\mathrm{flux}}, \sigma=\mathrm{CDPP})`.
        No other methods are supported at this time.

        Parameters
        ----------
        method : string {'gaussian_noise'}
            Method to use for gap filling. Fills with Gaussian noise by default.

        Returns
        -------
        filled_lightcurve : `LightCurve`
            A new light curve object in which all NaN values and gaps in time
            have been filled.
        """
        lc = self.copy().remove_nans()
        # nlc = lc.copy()
        newdata = {}

        # Find missing time points
        # Most precise method, taking into account time variation due to orbit
        if hasattr(lc, "cadenceno"):
            dt = lc.time.value - np.median(np.diff(lc.time.value)) * lc.cadenceno.value
            ncad = np.arange(lc.cadenceno.value[0], lc.cadenceno.value[-1] + 1, 1)
            in_original = np.in1d(ncad, lc.cadenceno.value)
            ncad = ncad[~in_original]
            ndt = np.interp(ncad, lc.cadenceno.value, dt)

            ncad = np.append(ncad, lc.cadenceno.value)
            ndt = np.append(ndt, dt)
            ncad, ndt = ncad[np.argsort(ncad)], ndt[np.argsort(ncad)]
            ntime = ndt + np.median(np.diff(lc.time.value)) * ncad
            newdata["cadenceno"] = ncad
        else:
            # Less precise method
            dt = np.nanmedian(lc.time.value[1::] - lc.time.value[:-1:])
            ntime = [lc.time.value[0]]
            for t in lc.time.value[1::]:
                prevtime = ntime[-1]
                while (t - prevtime) > 1.2 * dt:
                    ntime.append(prevtime + dt)
                    prevtime = ntime[-1]
                ntime.append(t)
            ntime = np.asarray(ntime, float)
            in_original = np.in1d(ntime, lc.time.value)

        # Fill in time points
        newdata["time"] = Time(ntime, format=lc.time.format, scale=lc.time.scale)
        f = np.zeros(len(ntime))
        f[in_original] = np.copy(lc.flux)
        fe = np.zeros(len(ntime))
        fe[in_original] = np.copy(lc.flux_err)

        # Temporary workaround for issue #1172.  TODO: remove the `if`` statement
        # below once we adopt AstroPy >=5.0.3 as a minimum dependency.
        if hasattr(lc.flux_err, 'mask'):
            fe[~in_original] = np.interp(ntime[~in_original], lc.time.value, lc.flux_err.unmasked)
        else:
            fe[~in_original] = np.interp(ntime[~in_original], lc.time.value, lc.flux_err)

        if method == "gaussian_noise":
            try:
                std = lc.estimate_cdpp().to(lc.flux.unit).value
            except:
                std = np.nanstd(lc.flux.value)
            f[~in_original] = np.random.normal(
                np.nanmean(lc.flux.value), std, (~in_original).sum()
            )
        else:
            raise NotImplementedError("No such method as {}".format(method))

        newdata["flux"] = Quantity(f, lc.flux.unit)
        newdata["flux_err"] = Quantity(fe, lc.flux_err.unit)

        if hasattr(lc, "quality"):
            quality = np.zeros(len(ntime), dtype=lc.quality.dtype)
            quality[in_original] = np.copy(lc.quality)
            quality[~in_original] += 65536
            newdata["quality"] = quality
        """
        # TODO: add support for other columns
        for column in lc.columns:
            if column in ("time", "flux", "flux_err", "quality"):
                continue
            old_values = lc[column]
            new_values = np.empty(len(ntime), dtype=old_values.dtype)
            new_values[~in_original] = np.nan
            new_values[in_original] = np.copy(old_values)
            newdata[column] = new_values
        """
        return LightCurve(data=newdata, meta=self.meta)

    def remove_outliers(
        self, sigma=5.0, sigma_lower=None, sigma_upper=None, return_mask=False, **kwargs
    ):
        """Removes outlier data points using sigma-clipping.

        This method returns a new `LightCurve` object from which data points
        are removed if their flux values are greater or smaller than the median
        flux by at least ``sigma`` times the standard deviation.

        Sigma-clipping works by iterating over data points, each time rejecting
        values that are discrepant by more than a specified number of standard
        deviations from a center value. If the data contains invalid values
        (NaNs or infs), they are automatically masked before performing the
        sigma clipping.

        .. note::
            This function is a convenience wrapper around
            `astropy.stats.sigma_clip()` and provides the same functionality.
            Any extra arguments passed to this method will be passed on to
            ``sigma_clip``.

        Parameters
        ----------
        sigma : float
            The number of standard deviations to use for both the lower and
            upper clipping limit. These limits are overridden by
            ``sigma_lower`` and ``sigma_upper``, if input. Defaults to 5.
        sigma_lower : float or None
            The number of standard deviations to use as the lower bound for
            the clipping limit. Can be set to float('inf') in order to avoid
            clipping outliers below the median at all. If `None` then the
            value of ``sigma`` is used. Defaults to `None`.
        sigma_upper : float or None
            The number of standard deviations to use as the upper bound for
            the clipping limit. Can be set to float('inf') in order to avoid
            clipping outliers above the median at all. If `None` then the
            value of ``sigma`` is used. Defaults to `None`.
        return_mask : bool
            Whether or not to return a mask (i.e. a boolean array) indicating
            which data points were removed. Entries marked as `True` in the
            mask are considered outliers.  This mask is not returned by default.
        **kwargs : dict
            Dictionary of arguments to be passed to `astropy.stats.sigma_clip`.

        Returns
        -------
        clean_lc : `LightCurve`
            A new light curve object from which outlier data points have been
            removed.
        outlier_mask : NumPy array, optional
            Boolean array flagging which cadences were removed.
            Only returned if `return_mask=True`.

        Examples
        --------
        This example generates a new light curve in which all points
        that are more than 1 standard deviation from the median are removed::

            >>> lc = LightCurve(time=[1, 2, 3, 4, 5], flux=[1, 1000, 1, -1000, 1])
            >>> lc_clean = lc.remove_outliers(sigma=1)
            >>> lc_clean.time
            <Time object: scale='tdb' format='jd' value=[1. 3. 5.]>
            >>> lc_clean.flux
            <Quantity [1., 1., 1.]>

        Instead of specifying `sigma`, you may specify separate `sigma_lower`
        and `sigma_upper` parameters to remove only outliers above or below
        the median. For example::

            >>> lc = LightCurve(time=[1, 2, 3, 4, 5], flux=[1, 1000, 1, -1000, 1])
            >>> lc_clean = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper=1)
            >>> lc_clean.time
            <Time object: scale='tdb' format='jd' value=[1. 3. 4. 5.]>
            >>> lc_clean.flux
            <Quantity [    1.,     1., -1000.,     1.]>

        Optionally, you may use the `return_mask` parameter to return a boolean
        array which flags the outliers identified by the method. For example::

            >>> lc_clean, mask = lc.remove_outliers(sigma=1, return_mask=True)
            >>> mask
            array([False,  True, False,  True, False])
        """
        # The import time for `sigma_clip` is somehow very slow, so we use
        # a local import here.
        from astropy.stats.sigma_clipping import sigma_clip

        # astropy.stats.sigma_clip won't work with masked ndarrays so we convert to regular arrays
        flux = self.flux.copy()
        if isinstance(flux, Masked):
            flux = flux.filled(np.nan)

        # First, we create the outlier mask using AstroPy's sigma_clip function
        with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
            warnings.simplefilter("ignore")
            flux = self.flux
            if isinstance(flux, Masked):
                # Workaround for https://github.com/astropy/astropy/issues/14360
                # in passing MaskedQuantity to sigma_clip, by converting it to Quantity.
                # We explicitly fill masked values with `np.nan` here to ensure they are masked during sigma clipping.
                # To handle unlikely edge case, convert int to float to ensure filing `np.nan` work.
                # The conversion is acceptable because only the mask of the sigma_clip() result is used.
                if np.issubdtype(flux.dtype, np.int_):
                    flux = flux.astype(float)
                flux = flux.filled(np.nan)
            outlier_mask = sigma_clip(
                data=flux,
                sigma=sigma,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                **kwargs,
            ).mask
        # Second, we return the masked light curve and optionally the mask itself
        if return_mask:
            return self.copy()[~outlier_mask], outlier_mask
        return self.copy()[~outlier_mask]

    @deprecated_renamed_argument(
        "binsize",
        new_name=None,
        since="2.0",
        warning_type=LightkurveDeprecationWarning,
        alternative="time_bin_size",
    )
    def bin(
        self,
        time_bin_size=None,
        time_bin_start=None,
        time_bin_end=None,
        n_bins=None,
        aggregate_func=None,
        bins=None,
        binsize=None,
    ):
        """Bins a lightcurve in equally-spaced bins in time.

        If the original light curve contains flux uncertainties (``flux_err``),
        the binned lightcurve will report the root-mean-square error.
        If no uncertainties are included, the binned curve will return the
        standard deviation of the data.

        Parameters
        ----------
        time_bin_size : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`, optional
            The time interval for the binned time series - this is either a scalar
            value (in which case all time bins will be assumed to have the same
            duration) or as an array of values (in which case each time bin can
            have a different duration). If this argument is provided,
            ``time_bin_end`` should not be provided.
            (Default: 0.5 days; default unit: days.)
        time_bin_start : `~astropy.time.Time` or iterable, optional
            The start time for the binned time series - this can be either given
            directly as a `~astropy.time.Time` array or as any iterable that
            initializes the `~astropy.time.Time` class. This can also be a scalar
            value if ``time_bin_size`` is provided. Defaults to the first
            time in the sampled time series.
        time_bin_end : `~astropy.time.Time` or iterable, optional
            The times of the end of each bin - this can be either given directly as
            a `~astropy.time.Time` array or as any iterable that initializes the
            `~astropy.time.Time` class. This can only be given if ``time_bin_start``
            is an array of values. If ``time_bin_end`` is a scalar, time bins are
            assumed to be contiguous, such that the end of each bin is the start
            of the next one, and ``time_bin_end`` gives the end time for the last
            bin. If ``time_bin_end`` is an array, the time bins do not need to be
            contiguous. If this argument is provided, ``time_bin_size`` should not
            be provided. This option, like the iterable form of ``time_bin_start``,
            requires Astropy 5.0.
        n_bins : int, optional
            The number of bins to use. Defaults to the number needed to fit all
            the original points. Note that this will create this number of bins
            of length ``time_bin_size`` independent of the lightkurve length.
        aggregate_func : callable, optional
            The function to use for combining points in the same bin. Defaults
            to np.nanmean.
        bins : int, iterable or str, optional
            If an int, this gives the number of bins to divide the lightkurve into.
            In contrast to ``n_bins`` this adjusts the length of ``time_bin_size``
            to accommodate the input time series length.
            If it is an iterable of ints, it specifies the indices of the bin edges.
            If a string, it must be one of  'blocks', 'knuth', 'scott' or 'freedman'
            defining a method of automatically determining an optimal bin size.
            See `~astropy.stats.histogram` for a description of each method.
            Note that 'blocks' is not a useful method for regularly sampled data.
        binsize : int
            In Lightkurve v1.x, the default behavior of `bin()` was to create
            bins which contained an equal number data points in each bin.
            This type of binning is discouraged because it usually makes more sense to
            create equally-sized bins in time duration, which is the new default
            behavior in Lightkurve v2.x.  Nevertheless, this `binsize` parameter
            allows users to simulate the old behavior of Lightkurve v1.x.
            For ease of implementation, setting this parameter is identical to passing
            ``time_bin_size = lc.time[binsize] - time[0]``, which means that
            the bins are not guaranteed to contain an identical number of
            data points.

        Returns
        -------
        binned_lc : `LightCurve`
            A new light curve which has been binned.
        """
        kwargs = dict()
        if binsize is not None and bins is not None:
            raise ValueError("Only one of ``bins`` and ``binsize`` can be specified.")
        elif (binsize is not None or bins is not None) and (
            time_bin_size is not None or n_bins is not None
        ):
            raise ValueError(
                "``bins`` or ``binsize`` conflicts with "
                "``n_bins`` or ``time_bin_size``."
            )
        elif bins is not None:
            if (bins not in ('blocks', 'knuth', 'scott', 'freedman') and
                    np.array(bins).dtype != np.int_):
                raise TypeError("``bins`` must have integer type.")
            elif (isinstance(bins, str) or np.size(bins) != 1) and not _HAS_VAR_BINS:
                raise ValueError("Sequence or method for ``bins`` requires Astropy 5.0.")

        if time_bin_start is None:
            time_bin_start = self.time[0]
        if not isinstance(time_bin_start, (Time, TimeDelta)):
            if isinstance(self.time, TimeDelta):
                time_bin_start = TimeDelta(
                    time_bin_start, format=self.time.format, scale=self.time.scale
                )
            else:
                time_bin_start = Time(
                    time_bin_start, format=self.time.format, scale=self.time.scale
                )

        # Backwards compatibility with Lightkurve v1.x
        if time_bin_size is None:
            if bins is not None:
                if np.size(bins) == 1 and _HAS_VAR_BINS:
                    # This actually calculates equal-length bins just as the method below;
                    # should it instead set equal-number bins with binsize=int(len(self) / bins)?
                    # Get start times in mjd and convert back to original format
                    bin_starts = calculate_bin_edges(self.time.mjd, bins=bins)[:-1]
                    time_bin_start = Time(Time(bin_starts, format='mjd'), format=self.time.format)
                elif np.size(bins) == 1:
                    warnings.warn(
                        '"classic" `bins` require Astropy 5.0; will use constant lengths in time.',
                        LightkurveWarning)
                    # Odd memory error in np.searchsorted with pytest-memtest?
                    if self.time[0] >= time_bin_start:
                        i = len(self.time)
                    else:
                        i = len(self.time) - np.searchsorted(self.time, time_bin_start)
                    time_bin_size = ((self.time[-1] - time_bin_start) * i /
                                     ((i - 1) * bins)).to(u.day)
                else:
                    time_bin_start = self.time[bins[:-1]]
                    kwargs['time_bin_end'] = self.time[bins[1:]]
            elif binsize is not None:
                if _HAS_VAR_BINS:
                    time_bin_start = self.time[::binsize]
                else:
                    warnings.warn(
                        '`binsize` requires Astropy 5.0 to guarantee equal number of points; '
                        'will use estimated time lengths for bins.', LightkurveWarning)
                    if self.time[0] >= time_bin_start:
                        i = 0
                    else:
                        i = np.searchsorted(self.time, time_bin_start)
                    time_bin_size = (self.time[i + binsize] - self.time[i]).to(u.day)
            else:
                time_bin_size = 0.5 * u.day
        elif not isinstance(time_bin_size, Quantity):
            time_bin_size *= u.day

        # Call AstroPy's aggregate_downsample
        with warnings.catch_warnings():
            # ignore uninteresting empty slice warnings
            warnings.simplefilter("ignore", (RuntimeWarning, AstropyUserWarning))
            ts = aggregate_downsample(
                self,
                time_bin_size=time_bin_size,
                n_bins=n_bins,
                time_bin_start=time_bin_start,
                aggregate_func=aggregate_func,
                **kwargs
            )

            # If `flux_err` is populated, assume the errors combine as the root-mean-square
            if np.any(np.isfinite(self.flux_err)):
                rmse_func = (
                    lambda x: np.sqrt(np.nansum(x ** 2)) / len(np.atleast_1d(x))
                    if np.any(np.isfinite(x))
                    else np.nan
                )
                ts_err = aggregate_downsample(
                    self,
                    time_bin_size=time_bin_size,
                    n_bins=n_bins,
                    time_bin_start=time_bin_start,
                    aggregate_func=rmse_func,
                )
                ts["flux_err"] = ts_err["flux_err"]
            # If `flux_err` is unavailable, populate `flux_err` as nanstd(flux)
            else:
                ts_err = aggregate_downsample(
                    self,
                    time_bin_size=time_bin_size,
                    n_bins=n_bins,
                    time_bin_start=time_bin_start,
                    aggregate_func=np.nanstd,
                )
                ts["flux_err"] = ts_err["flux"]

        # Prepare a LightCurve object by ensuring there is a time column
        ts._required_columns = []
        ts.add_column(ts.time_bin_start + ts.time_bin_size / 2.0, name="time")

        # Ensure the required columns appear in the correct order
        for idx, colname in enumerate(self.__class__._required_columns):
            tmpcol = ts[colname]
            ts.remove_column(colname)
            ts.add_column(tmpcol, name=colname, index=idx)

        return self.__class__(ts, meta=self.meta)

    def estimate_cdpp(
        self, transit_duration=13, savgol_window=101, savgol_polyorder=2, sigma=5.0
    ) -> float:
        """Estimate the CDPP noise metric using the Savitzky-Golay (SG) method.

        A common estimate of the noise in a lightcurve is the scatter that
        remains after all long term trends have been removed. This is the idea
        behind the Combined Differential Photometric Precision (CDPP) metric.
        The official Kepler Pipeline computes this metric using a wavelet-based
        algorithm to calculate the signal-to-noise of the specific waveform of
        transits of various durations. In this implementation, we use the
        simpler "sgCDPP proxy algorithm" discussed by Gilliland et al
        (2011ApJS..197....6G) and Van Cleve et al (2016PASP..128g5002V).

        The steps of this algorithm are:
            1. Remove low frequency signals using a Savitzky-Golay filter with
               window length `savgol_window` and polynomial order `savgol_polyorder`.
            2. Remove outliers by rejecting data points which are separated from
               the mean by `sigma` times the standard deviation.
            3. Compute the standard deviation of a running mean with
               a configurable window length equal to `transit_duration`.

        We use a running mean (as opposed to block averaging) to strongly
        attenuate the signal above 1/transit_duration whilst retaining
        the original frequency sampling.  Block averaging would set the Nyquist
        limit to 1/transit_duration.

        Parameters
        ----------
        transit_duration : int, optional
            The transit duration in units of number of cadences. This is the
            length of the window used to compute the running mean. The default
            is 13, which corresponds to a 6.5 hour transit in data sampled at
            30-min cadence.
        savgol_window : int, optional
            Width of Savitsky-Golay filter in cadences (odd number).
            Default value 101 (2.0 days in Kepler Long Cadence mode).
        savgol_polyorder : int, optional
            Polynomial order of the Savitsky-Golay filter.
            The recommended value is 2.
        sigma : float, optional
            The number of standard deviations to use for clipping outliers.
            The default is 5.

        Returns
        -------
        cdpp : float
            Savitzky-Golay CDPP noise metric in units parts-per-million (ppm).

        Notes
        -----
        This implementation is adapted from the Matlab version used by
        Jeff van Cleve but lacks the normalization factor used there:
        svn+ssh://murzim/repo/so/trunk/Develop/jvc/common/compute_SG_noise.m
        """
        if not isinstance(transit_duration, int):
            raise ValueError(
                "transit_duration must be an integer in units "
                "number of cadences, got {}.".format(transit_duration)
            )

        detrended_lc = self.flatten(
            window_length=savgol_window, polyorder=savgol_polyorder
        )
        cleaned_lc = detrended_lc.remove_outliers(sigma=sigma)
        with warnings.catch_warnings():  # ignore "already normalized" message
            warnings.filterwarnings("ignore", message=".*already.*")
            normalized_lc = cleaned_lc.normalize("ppm")
        mean = running_mean(data=normalized_lc.flux, window_size=transit_duration)
        return np.std(mean)

    def query_solar_system_objects(
        self,
        cadence_mask="outliers",
        radius=None,
        sigma=3,
        location=None,
        cache=True,
        return_mask=False,
        show_progress=True,
    ):
        """Returns a list of asteroids or comets which affected the light curve.

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
        * The size of the search cone is 15 spacecraft pixels by default. You
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
            Radius in degrees to search for bodies. If None, will search for
            SSOs within 15 pixels.
        sigma : optional, float
            If `cadence_mask` is set to `"outlier"`, `sigma` will be used to identify
            outliers.
        location : optional, str
            Spacecraft location. Options include `'kepler'` and `'tess'`. Default: `self.mission`
        cache : optional, bool
            If True will cache the search result in the astropy cache. Set to False
            to request the search again.
        return_mask: optional, bool
            If True will return a boolean mask in time alongside the result
        show_progress: optional, bool
            If True will display a progress bar during the download

        Returns
        -------
        result : `pandas.DataFrame`
            DataFrame object which lists the Solar System objects in frames
            that were identified to contain SSOs.  Returns `None` if no objects
            were found.

        Examples
        --------
        Find if there are SSOs affecting the lightcurve for the given time frame:

            >>> df_sso = lc.query_solar_system_objects(cadence_mask=(lc.time.value >= 2014.1) & (lc.time.value <= 2014.9))  # doctest: +SKIP

        Find if there are SSOs affecting the lightcurve for all times, but it will be much slower:

            >>> df_sso = lc.query_solar_system_objects(cadence_mask='all')  # doctest: +SKIP

        """
        for attr in ["ra", "dec"]:
            if not hasattr(self, "{}".format(attr)):
                raise ValueError("Input does not have a `{}` attribute.".format(attr))

        # Validate `cadence_mask`
        if isinstance(cadence_mask, str):
            if cadence_mask == "outliers":
                cadence_mask = self.remove_outliers(sigma=sigma, return_mask=True)[1]
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
            raise ValueError("the `cadence_mask` argument is missing or invalid")
        # Avoid searching times with NaN flux; this is necessary because e.g.
        # `remove_outliers` includes NaNs in its mask.
        if hasattr(self.flux, 'mask'):
            # Temporary workaround for issue #1172. TODO: remove this `if`` statement
            # once we adopt AstroPy >=5.0.3 as a minimum dependency
            cadence_mask &= ~np.isnan(self.flux.unmasked)
        else:
            cadence_mask &= ~np.isnan(self.flux)

        # Validate `location`
        if location is None:
            if hasattr(self, "mission") and self.mission:
                location = self.mission.lower()
            else:
                raise ValueError("you must pass a value for `location`.")

        # Validate `radius`
        if radius is None:
            # 15 pixels has been chosen as a reasonable default.
            # Comets have long tails which have tripped up users.
            if (location == "kepler") | (location == "k2"):
                radius = (4 * 15) * u.arcsecond.to(u.deg)
            elif location == "tess":
                radius = (21 * 15) * u.arcsecond.to(u.deg)
            else:
                radius = 15 * u.arcsecond.to(u.deg)

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

    def _create_plot(
        self,
        method="plot",
        column="flux",
        ax=None,
        normalize=False,
        xlabel=None,
        ylabel=None,
        title="",
        style="lightkurve",
        show_colorbar=True,
        colorbar_label="",
        offset=None,
        clip_outliers=False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Implements `plot()`, `scatter()`, and `errorbar()` to avoid code duplication.

        Parameters
        ----------
        method : str
            One of 'plot', 'scatter', or 'errorbar'.
        column : str
            Name of data column to plot. Default `flux`.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            X axis label.
        ylabel : str
            Y axis label.
        title : str
            Title shown at the top using matplotlib `set_title`.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        show_colorbar : boolean
            Show the colorbar if colors are given using the `c` argument?
        colorbar_label : str
            Label to show next to the colorbar (if `c` is given).
        offset : float
            Offset value to apply to the Y axis values before plotting. Use this
            to avoid light curves from overlapping on the same plot. By default,
            no offset is applied.
        clip_outliers : bool
            If ``True``, clip the y axis limit to the 95%-percentile range.
        kwargs : dict
            Dictionary of arguments to be passed to Matplotlib's `plot`,
            `scatter`, or `errorbar` methods.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        # Configure the default style
        if style is None or style == "lightkurve":
            style = MPLSTYLE
        # Default xlabel
        if xlabel is None:
            if not hasattr(self.time, "format"):
                xlabel = "Phase"
            elif self.time.format == "bkjd":
                xlabel = "Time - 2454833 [BKJD days]"
            elif self.time.format == "btjd":
                xlabel = "Time - 2457000 [BTJD days]"
            elif self.time.format == "jd":
                xlabel = "Time [JD]"
            else:
                xlabel = "Time"

        # Default ylabel
        if ylabel is None:
            if "flux" == column:
                ylabel = "Flux"
            else:
                ylabel = f"{column}"
            if normalize or (column == "flux" and self.meta.get("NORMALIZED")):
                ylabel = "Normalized " + ylabel
            elif (self[column].unit) and (self[column].unit.to_string() != ""):
                ylabel += f" [{self[column].unit.to_string('latex_inline')}]"

        # Default legend label
        if "label" not in kwargs:
            kwargs["label"] = self.meta.get("LABEL")

        # Workaround for AstroPy v5.0.0 issue #12481: the 'c' argument
        # in matplotlib's scatter does not work with masked quantities.
        if "c" in kwargs and hasattr(kwargs["c"], 'mask'):
            kwargs["c"] = kwargs["c"].unmasked

        flux = self[column]
        try:
            flux_err = self[f"{column}_err"]
        except KeyError:
            flux_err = np.full(len(flux), np.nan)

        # Second workaround for AstroPy v5.0.0 issue #12481:
        # matplotlib does not work well with `MaskedNDArray` arrays.
        if hasattr(flux, 'mask'):
            flux = flux.filled(np.nan)
        if hasattr(flux_err, 'mask'):
            flux_err = flux_err.filled(np.nan)

        # Normalize the data if requested
        if normalize:
            # ignore "light curve is already normalized" message because
            # the user explicitely asked for normalization here
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*already.*")
                if column == "flux":
                    lc_normed = self.normalize()
                else:
                    # Code below is a temporary hack because `normalize()`
                    # does not have a `column` argument yet
                    lc_tmp = self.copy()
                    lc_tmp["flux"] = flux
                    lc_tmp["flux_err"] = flux_err
                    lc_normed = lc_tmp.normalize()
                flux, flux_err = lc_normed.flux, lc_normed.flux_err

        # Apply offset if requested
        if offset:
            flux = flux.copy() + offset * flux.unit

        # Make the plot
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(1)
            if method == "scatter":
                sc = ax.scatter(self.time.value, flux, **kwargs)
                # Colorbars should only be plotted if the user specifies, and there is
                # a color specified that is not a string (e.g. 'C1') and is iterable.
                if (
                    show_colorbar
                    and ("c" in kwargs)
                    and (not isinstance(kwargs["c"], str))
                    and hasattr(kwargs["c"], "__iter__")
                ):
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label(colorbar_label)
                    cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
                    cbar.ax.minorticks_off()
            elif method == "errorbar":
                if np.any(~np.isnan(flux_err)):
                    ax.errorbar(
                        x=self.time.value, y=flux.value, yerr=flux_err.value, **kwargs
                    )
                else:
                    log.warning(f"Column `{column}` has no associated errors.")
            else:
                ax.plot(self.time.value, flux.value, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if np.sum([len(a) for a in legend_labels]) != 0:
                ax.legend(loc="best")

            if clip_outliers and len(flux) > 0:
                ymin, ymax = np.percentile(flux.value, [2.5, 97.5])
                margin = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - margin, ymax + margin)

        return ax

    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the light curve using Matplotlib's `~matplotlib.pyplot.plot` method.

        Parameters
        ----------
        column : str
            Name of data column to plot. Default `flux`.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            X axis label.
        ylabel : str
            Y axis label.
        title : str
            Title shown at the top using matplotlib `set_title`.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        show_colorbar : boolean
            Show the colorbar if colors are given using the `c` argument?
        colorbar_label : str
            Label to show next to the colorbar (if `c` is given).
        offset : float
            Offset value to apply to the Y axis values before plotting. Use this
            to avoid light curves from overlapping on the same plot. By default,
            no offset is applied.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._create_plot(method="plot", **kwargs)

    def scatter(
        self, colorbar_label="", show_colorbar=True, **kwargs
    ) -> matplotlib.axes.Axes:
        """Plots the light curve using Matplotlib's `~matplotlib.pyplot.scatter` method.

        Parameters
        ----------
        column : str
            Name of data column to plot. Default `flux`.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            X axis label.
        ylabel : str
            Y axis label.
        title : str
            Title shown at the top using matplotlib `set_title`.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        show_colorbar : boolean
            Show the colorbar if colors are given using the `c` argument?
        colorbar_label : str
            Label to show next to the colorbar (if `c` is given).
        offset : float
            Offset value to apply to the Y axis values before plotting. Use this
            to avoid light curves from overlapping on the same plot. By default,
            no offset is applied.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.scatter`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._create_plot(
            method="scatter",
            colorbar_label=colorbar_label,
            show_colorbar=show_colorbar,
            **kwargs,
        )

    def errorbar(self, linestyle="", **kwargs) -> matplotlib.axes.Axes:
        """Plots the light curve using Matplotlib's `~matplotlib.pyplot.errorbar` method.

        Parameters
        ----------
        linestyle : str
            Connect the error bars using a line?
        column : str
            Name of data column to plot. Default `flux`.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            X axis label.
        ylabel : str
            Y axis label.
        title : str
            Title shown at the top using matplotlib `set_title`.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        show_colorbar : boolean
            Show the colorbar if colors are given using the `c` argument?
        colorbar_label : str
            Label to show next to the colorbar (if `c` is given).
        offset : float
            Offset value to apply to the Y axis values before plotting. Use this
            to avoid light curves from overlapping on the same plot. By default,
            no offset is applied.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.errorbar`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if "ls" not in kwargs:
            kwargs["linestyle"] = linestyle
        return self._create_plot(method="errorbar", **kwargs)

    def interact_bls(
        self,
        notebook_url=None,
        minimum_period=None,
        maximum_period=None,
        resolution=2000,
    ):
        """Display an interactive Jupyter Notebook widget to find planets.

        The Box Least Squares (BLS) periodogram is a statistical tool used
        for detecting transiting exoplanets and eclipsing binaries in
        light curves.  This method will display a Jupyter Notebook Widget
        which enables the BLS algorithm to be used interactively.
        Behind the scenes, the widget uses the AstroPy implementation of BLS [1]_.

        This feature only works inside an active Jupyter Notebook.
        It requires Bokeh v1.0 (or later). An error message will be shown
        if these dependencies are not available.

        Parameters
        ----------
        notebook_url: str
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
        minimum_period : float or None
            Minimum period to assess the BLS to. If None, default value of 0.3 days
            will be used.
        maximum_period : float or None
            Maximum period to evaluate the BLS to. If None, the time coverage of the
            lightcurve / 2 will be used.
        resolution : int
            Number of points to use in the BLS panel. Lower this value for faster
            but less accurate performance. You can also vary this value using the
            widget's Resolution Slider.

        Examples
        --------
        Load the light curve for Kepler-10, remove long-term trends, and
        display the BLS tool as follows:

            >>> import lightkurve as lk
            >>> lc = lk.search_lightcurve('kepler-10', quarter=3).download()  # doctest: +SKIP
            >>> lc = lc.normalize().flatten()  # doctest: +SKIP
            >>> lc.interact_bls()  # doctest: +SKIP

        References
        ----------
        .. [1] https://docs.astropy.org/en/stable/timeseries/bls.html
        """
        from .interact_bls import show_interact_widget

        notebook_url = finalize_notebook_url(notebook_url)

        return show_interact_widget(
            self,
            notebook_url=notebook_url,
            minimum_period=minimum_period,
            maximum_period=maximum_period,
            resolution=resolution,
        )

    def to_table(self) -> Table:
        return Table(self)

    @deprecated(
        "2.0",
        message="`to_timeseries()` has been deprecated. `LightCurve` is a "
        "sub-class of Astropy TimeSeries as of Lightkurve v2.0 "
        "and no longer needs to be converted.",
        warning_type=LightkurveDeprecationWarning,
    )
    def to_timeseries(self):
        return self

    @staticmethod
    def from_timeseries(ts):
        """Creates a new `LightCurve` from an AstroPy
        `~astropy.timeseries.TimeSeries` object.

        Parameters
        ----------
        ts : `~astropy.timeseries.TimeSeries`
            The AstroPy TimeSeries object.  The object must contain columns
            named 'time', 'flux', and 'flux_err'.
        """
        return LightCurve(
            time=ts["time"].value, flux=ts["flux"], flux_err=ts["flux_err"]
        )

    def to_stingray(self):
        """Returns a `stingray.Lightcurve` object.

        This feature requires `Stingray <https://stingraysoftware.github.io/>`_
        to be installed (e.g. ``pip install stingray``).  An `ImportError` will
        be raised if this package is not available.

        Returns
        -------
        lightcurve : `stingray.Lightcurve`
            An stingray Lightcurve object.
        """
        try:
            from stingray import Lightcurve as StingrayLightcurve
        except ImportError:
            raise ImportError(
                "You need to install Stingray to use "
                "the LightCurve.to_stringray() method."
            )
        return StingrayLightcurve(
            time=self.time.value,
            counts=self.flux,
            err=self.flux_err,
            input_counts=False,
        )

    @staticmethod
    def from_stingray(lc):
        """Create a new `LightCurve` from a `stingray.Lightcurve`.

        Parameters
        ----------
        lc : `stingray.Lightcurve`
            A stingray Lightcurve object.
        """
        return LightCurve(time=lc.time, flux=lc.counts, flux_err=lc.counts_err)

    def to_csv(self, path_or_buf=None, **kwargs):
        """Writes the light curve to a CSV file.

        This method will convert the light curve into the Comma-Separated Values
        (CSV) text format. By default this method will return the result as a
        string, but you can also write the string directly to disk by providing
        a file name or handle via the `path_or_buf` parameter.

        Parameters
        ----------
        path_or_buf : string or file handle
            File path or object. By default, the result is returned as a string.
        **kwargs : dict
            Dictionary of arguments to be passed to
            `astropy`'s `~astropy.timeseries.TimeSeries.write`.

        Returns
        -------
        csv : str or None
            Returns a csv-formatted string if ``path_or_buf=None``.
            Returns `None` otherwise.
        """
        use_stringio = False
        if path_or_buf is None:
            use_stringio = True
            from io import StringIO

            path_or_buf = StringIO()
        result = self.write(path_or_buf, format="ascii.csv", **kwargs)
        if use_stringio:
            return path_or_buf.getvalue()
        return result

    def to_pandas(self, **kwargs):
        """Converts the light curve to a Pandas `~pandas.DataFrame` object.

        The data frame will be indexed by `time` using values corresponding
        to the light curve's time format.  This is different from the
        default behavior of `astropy`'s `~astropy.timeseries.TimeSeries.to_pandas`,
        which converts time values into ISO timestamps.

        Returns
        -------
        dataframe : `pandas.DataFrame`
            A data frame indexed by `time`.
        """
        df = super().to_pandas(**kwargs)
        # Default AstroPy behavior is to change the time column into ``np.datetime64``
        # We override it here because it confuses Kepler/TESS users who are used
        # to working in BTJD and BKJD rather than ISO timestamps.
        df.index = self.time.value
        df.index.name = "time"
        return df

    def to_excel(self, path_or_buf, **kwargs) -> None:
        """Shorthand for `to_pandas().to_excel()`.

        Parameters
        ----------
        path_or_buf : string or file handle
            File path or object.
        **kwargs : dict
            Dictionary of arguments to be passed to `to_pandas().to_excel(**kwargs)`.
        """
        try:
            import openpyxl  # optional dependency
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to install `openpyxl` to use this feature, e.g. use `pip install openpyxl`."
            )
        self.to_pandas().to_excel(path_or_buf, **kwargs)

    def to_periodogram(self, method="lombscargle", **kwargs):
        """Converts the light curve to a `~lightkurve.periodogram.Periodogram`
        power spectrum object.

        This method will call either
        `LombScarglePeriodogram.from_lightcurve() <lightkurve.periodogram.LombScarglePeriodogram.from_lightcurve>` or
        `BoxLeastSquaresPeriodogram.from_lightcurve() <lightkurve.periodogram.BoxLeastSquaresPeriodogram.from_lightcurve>`,
        which in turn wrap `astropy`'s `~astropy.timeseries.LombScargle` and `~astropy.timeseries.BoxLeastSquares`.

        Optional keywords accepted if ``method='lombscargle'`` are:
        ``minimum_frequency``, ``maximum_frequency``, ``mininum_period``,
        ``maximum_period``, ``frequency``, ``period``, ``nterms``,
        ``nyquist_factor``, ``oversample_factor``, ``freq_unit``,
        ``normalization``, ``ls_method``.

        Optional keywords accepted if ``method='bls'`` are
        ``minimum_period``, ``maximum_period``, ``period``,
        ``frequency_factor``, ``duration``.

        Parameters
        ----------
        method : {'lombscargle', 'boxleastsquares', 'ls', 'bls'}
            Use the Lomb Scargle or Box Least Squares (BLS) method to
            extract the power spectrum. Defaults to ``'lombscargle'``.
            ``'ls'`` and ``'bls'`` are shorthands for ``'lombscargle'``
            and ``'boxleastsquares'``.
        kwargs : dict
            Keyword arguments passed to either
            `LombScarglePeriodogram <lightkurve.periodogram.LombScarglePeriodogram.from_lightcurve>` or
            `BoxLeastSquaresPeriodogram <lightkurve.periodogram.BoxLeastSquaresPeriodogram.from_lightcurve>`.

        Returns
        -------
        Periodogram : `~lightkurve.periodogram.Periodogram` object
            The power spectrum object extracted from the light curve.
        """
        supported_methods = ["ls", "bls", "lombscargle", "boxleastsquares"]
        method = validate_method(method.replace(" ", ""), supported_methods)
        if method in ["bls", "boxleastsquares"]:
            from .periodogram import BoxLeastSquaresPeriodogram

            return BoxLeastSquaresPeriodogram.from_lightcurve(lc=self, **kwargs)
        else:
            from .periodogram import LombScarglePeriodogram

            return LombScarglePeriodogram.from_lightcurve(lc=self, **kwargs)

    def to_seismology(self, **kwargs):
        """Returns a `~lightkurve.seismology.Seismology` object for estimating
        quick-look asteroseismic quantities.

        All `**kwargs` will be passed to the `to_periodogram()` method.

        Returns
        -------
        seismology : `~lightkurve.seismology.Seismology` object
            Object which can be used to estimate quick-look asteroseismic quantities.
        """
        from .seismology import Seismology

        return Seismology.from_lightcurve(self, **kwargs)

    def to_fits(
        self, path=None, overwrite=False, flux_column_name="FLUX", **extra_data
    ):
        """Converts the light curve to a FITS file in the Kepler/TESS file format.

        The FITS file will be returned as a `~astropy.io.fits.HDUList` object.
        If a `path` is specified then the file will also be written to disk.

        Parameters
        ----------
        path : str or None
            Location where the FITS file will be written, which is optional.
        overwrite : bool
            Whether or not to overwrite the file, if `path` is set.
        flux_column_name : str
            The column name in the FITS file where the light curve flux data
            should be stored.  Typical values are `FLUX` or `SAP_FLUX`.
        extra_data : dict
            Extra keywords or columns to include in the FITS file.
            Arguments of type str, int, float, or bool will be stored as
            keywords in the primary header.
            Arguments of type np.array or list will be stored as columns
            in the first extension.

        Returns
        -------
        hdu : `~astropy.io.fits.HDUList`
            Returns an `~astropy.io.fits.HDUList` object.
        """
        typedir = {
            int: "J",
            str: "A",
            float: "D",
            bool: "L",
            np.int32: "J",
            np.int32: "K",
            np.float32: "E",
            np.float64: "D",
        }

        def _header_template(extension):
            """Returns a template `fits.Header` object for a given extension."""
            template_fn = os.path.join(
                PACKAGEDIR, "data", "lc-ext{}-header.txt".format(extension)
            )
            return fits.Header.fromtextfile(template_fn)

        def _make_primary_hdu(extra_data=None):
            """Returns the primary extension (#0)."""
            if extra_data is None:
                extra_data = {}
            hdu = fits.PrimaryHDU()
            # Copy the default keywords from a template file from the MAST archive
            tmpl = _header_template(0)
            for kw in tmpl:
                hdu.header[kw] = (tmpl[kw], tmpl.comments[kw])

            # Override the defaults where necessary
            from . import __version__

            default = {
                "ORIGIN": "Unofficial data product",
                "DATE": datetime.datetime.now().strftime("%Y-%m-%d"),
                "CREATOR": "lightkurve.LightCurve.to_fits()",
                "PROCVER": str(__version__),
            }

            for kw in default:
                hdu.header["{}".format(kw).upper()] = default[kw]
                if default[kw] is None:
                    log.warning("Value for {} is None.".format(kw))

            for kw in extra_data:
                if isinstance(extra_data[kw], (str, float, int, bool, type(None))):
                    hdu.header["{}".format(kw).upper()] = extra_data[kw]
                    if extra_data[kw] is None:
                        log.warning("Value for {} is None.".format(kw))
            return hdu

        def _make_lightcurve_extension(extra_data=None):
            """Create the 'LIGHTCURVE' extension (i.e. extension #1)."""
            # Turn the data arrays into fits columns and initialize the HDU
            if extra_data is None:
                extra_data = {}
            cols = []
            if ~np.asarray(["TIME" in k.upper() for k in extra_data.keys()]).any():
                cols.append(
                    fits.Column(
                        name="TIME",
                        format="D",
                        unit=self.time.format,
                        array=self.time.value,
                    )
                )
            if ~np.asarray(
                [flux_column_name in k.upper() for k in extra_data.keys()]
            ).any():
                cols.append(
                    fits.Column(
                        name=flux_column_name,
                        format="E",
                        unit=self.flux.unit.to_string(),
                        array=self.flux,
                    )
                )
            if hasattr(self,'flux_err'):
                if ~(flux_column_name.upper() + "_ERR" in extra_data.keys()):
                    cols.append(
                        fits.Column(
                            name=flux_column_name.upper() + "_ERR",
                            format="E",
                            unit=self.flux_err.unit.to_string(),
                            array=self.flux_err,
                        )
                    )
            if hasattr(self,'cadenceno'):
                if ~np.asarray(
                    ["CADENCENO" in k.upper() for k in extra_data.keys()]
                ).any():
                    cols.append(
                        fits.Column(name="CADENCENO", format="J", array=self.cadenceno)
                    )
            for kw in extra_data:
                if isinstance(extra_data[kw], (np.ndarray, list)):
                    cols.append(
                        fits.Column(
                            name="{}".format(kw).upper(),
                            format=typedir[extra_data[kw].dtype.type],
                            array=extra_data[kw],
                        )
                    )
            if "SAP_QUALITY" not in extra_data:
                cols.append(
                    fits.Column(
                        name="SAP_QUALITY", format="J", array=np.zeros(len(self.flux))
                    )
                )

            coldefs = fits.ColDefs(cols)
            hdu = fits.BinTableHDU.from_columns(coldefs)
            hdu.header["EXTNAME"] = "LIGHTCURVE"
            return hdu

        def _hdulist(**extra_data):
            """Returns an astropy.io.fits.HDUList object."""
            list_out = fits.HDUList(
                [
                    _make_primary_hdu(extra_data=extra_data),
                    _make_lightcurve_extension(extra_data=extra_data),
                ]
            )
            return list_out

        hdu = _hdulist(**extra_data)
        if path is not None:
            hdu.writeto(path, overwrite=overwrite, checksum=True)
        return hdu

    def to_corrector(self, method="sff", **kwargs):
        """Returns a corrector object to remove instrument systematics.

        Parameters
        ----------
        methods : string
            Currently, "sff" and "cbv" are supported.  This will return a
            `~correctors.SFFCorrector` and `~correctors.CBVCorrector`
            class instance respectively.
        **kwargs : dict
            Extra keyword arguments to be passed to the corrector class.

        Returns
        -------
        correcter : `~correctors.corrector.Corrector`
            Instance of a Corrector class, which typically provides
            `~correctors.corrector.Corrector.correct()`
            and `~correctors.corrector.Corrector.diagnose()` methods.
        """
        if method == "pld":
            raise ValueError(
                "The 'pld' method can only be used on "
                "`TargetPixelFile` objects, not `LightCurve` objects."
            )
        method = validate_method(method, supported_methods=["sff", "cbv"])
        if method == "sff":
            from .correctors import SFFCorrector

            return SFFCorrector(self, **kwargs)
        elif method == "cbv":
            from .correctors import CBVCorrector

            return CBVCorrector(self, **kwargs)

    @deprecated_renamed_argument(
        "t0", "epoch_time", "2.0", warning_type=LightkurveDeprecationWarning
    )
    def plot_river(
        self,
        period,
        epoch_time=None,
        ax=None,
        bin_points=1,
        minimum_phase=-0.5,
        maximum_phase=0.5,
        method="mean",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the light curve as a river plot.

        A river plot uses colors to represent the light curve values in
        chronological order, relative to the period of an interesting signal.
        Each row in the plot represents a full period cycle, and each column
        represents a fixed phase.  This type of plot is often used to visualize
        Transit Timing Variations (TTVs) in the light curves of exoplanets, but
        it can be used to visualize periodic signals of any origin.

        All extra keywords supplied are passed on to Matplotlib's
        `~matplotlib.pyplot.pcolormesh` function.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        period: float
            Period at which to fold the light curve
        epoch_time : float
            Phase mid point for plotting. Defaults to the first time value.
        bin_points : int
            How many points should be in each bin.
        minimum_phase : float
            The minimum phase to plot.
        maximum_phase : float
            The maximum phase to plot.
        method : str
            The river method. Choose from `'mean'` or `'median'` or `'sigma'`.
            If `'mean'` or `'median'`, the plot will display the average value in each bin.
            If `'sigma'`, the plot will display the average in the bin divided by
            the error in each bin, in order to show the data in terms of standard
            deviation.
        kwargs : dict
            Dictionary of arguments to be passed on to Matplotlib's
            `~matplotlib.pyplot.pcolormesh` function.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if hasattr(self, "time_original"):  # folded light curve
            time = self.time_original
        else:
            time = self.time

        # epoch_time defaults to the first time value
        if epoch_time is None:
            epoch_time = time[0]

        # Lightkurve v1.x assumed that `period` was given in days if no unit
        # was specified.  We maintain this behavior for backwards-compatibility.
        if period is not None and not isinstance(period, Quantity):
            period *= u.day
        if epoch_time is not None and not isinstance(epoch_time, (Time, Quantity)):
            epoch_time = Time(epoch_time, format=time.format, scale=time.scale)

        method = validate_method(method, supported_methods=["mean", "median", "sigma"])
        if (bin_points == 1) and (method in ["mean", "median"]):
            bin_func = lambda y, e: (y[0], e[0])
        elif (bin_points == 1) and (method in ["sigma"]):
            bin_func = lambda y, e: ((y[0] - 1) / e[0], np.nan)
        elif method == "mean":
            bin_func = lambda y, e: (np.nanmean(y), np.nansum(e ** 2) ** 0.5 / len(e))
        elif method == "median":
            bin_func = lambda y, e: (np.nanmedian(y), np.nansum(e ** 2) ** 0.5 / len(e))
        elif method == "sigma":
            bin_func = lambda y, e: (
                (np.nanmean(y) - 1) / (np.nansum(e ** 2) ** 0.5 / len(e)),
                np.nan,
            )

        s = np.argsort(time.value)
        x, y, e = time.value[s], self.flux[s], self.flux_err[s]
        med = np.nanmedian(self.flux)
        e /= med
        y /= med

        # Here `ph` is the phase of each time point x
        # cyc is the number of cycles that have occured at each time point x
        # since the phase 0 before x[0]
        n = int(
            period.value
            / np.nanmedian(np.diff(x))
            * (maximum_phase - minimum_phase)
            / bin_points
        )
        if n == 1:
            bin_points = int(maximum_phase - minimum_phase) / (
                2 / int(period.value / np.nanmedian(np.diff(x)))
            )
            warnings.warn(
                "`bin_points` is too high to plot a phase curve, resetting to {}".format(
                    bin_points
                ),
                LightkurveWarning,
            )
            n = 2
        ph = x / period.value % 1
        cyc = np.asarray((x - x % period.value) / period.value, int)
        cyc -= np.min(cyc)

        phase = (epoch_time.value % period.value) / period.value
        ph = ((x - (phase * period.value)) / period.value) % 1
        cyc = np.asarray(
            (x - ((x - phase * period.value) % period.value)) / period.value, int
        )
        cyc -= np.min(cyc)
        ph[ph > 0.5] -= 1

        ar = np.empty((n, np.max(cyc) + 1))
        ar[:] = np.nan
        bs = np.linspace(minimum_phase, maximum_phase, n + 1)
        cycs = np.arange(0, np.max(cyc) + 2)

        ph_masks = [(ph > bs[jdx]) & (ph <= bs[jdx + 1]) for jdx in range(n)]
        qual_mask = np.isfinite(y)
        for cyc1 in np.unique(cyc):
            cyc_mask = cyc == cyc1
            if not np.any(cyc_mask):
                continue
            for jdx, ph_mask in enumerate(ph_masks):
                if not np.any(cyc_mask & ph_mask & qual_mask):
                    ar[jdx, cyc1] = np.nan
                else:
                    ar[jdx, cyc1] = bin_func(
                        y[cyc_mask & ph_mask], e[cyc_mask & ph_mask]
                    )[0]

        # If the method is average we need to denormalize the plot
        if method in ["mean", "median"]:
            median = np.nanmedian(self.flux.value)
            if hasattr(median, 'mask'):
                median = median.filled(np.nan)
            ar *= median

        d = np.max(
            [
                np.abs(np.nanmedian(ar) - np.nanpercentile(ar, 5)),
                np.abs(np.nanmedian(ar) - np.nanpercentile(ar, 95)),
            ]
        )
        vmin = kwargs.pop("vmin", np.nanmedian(ar) - d)
        vmax = kwargs.pop("vmax", np.nanmedian(ar) + d)
        if method in ["mean", "median"]:
            cmap = kwargs.pop("cmap", "viridis")
        elif method == "sigma":
            cmap = kwargs.pop("cmap", "coolwarm")

        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots(figsize=(12, cyc.max() * 0.1))

            im = ax.pcolormesh(
                bs, cycs, ar.T, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
            )
            cbar = plt.colorbar(im, ax=ax)
            if method in ["mean", "median"]:
                unit = "[Normalized Flux]"
                if self.flux.unit is not None:
                    if self.flux.unit != u.dimensionless_unscaled:
                        unit = "[{}]".format(self.flux.unit.to_string("latex"))
                if bin_points == 1:
                    cbar.set_label("Flux {}".format(unit))
                else:
                    cbar.set_label("Average Flux in Bin {}".format(unit))
            elif method == "sigma":
                if bin_points == 1:
                    cbar.set_label(
                        "Flux in units of Standard Deviation "
                        r"$(f - \overline{f})/(\sigma_f)$"
                    )
                else:
                    cbar.set_label(
                        "Average Flux in Bin in units of Standard Deviation "
                        r"$(f - \overline{f})/(\sigma_f)$"
                    )

            ax.set_xlabel("Phase")
            ax.set_ylabel("Cycle")
            ax.set_ylim(cyc.max(), 0)
            ax.set_title(self.meta.get("LABEL"))
            a = cyc.max() * 0.1 / 12.0
            b = (cyc.max() - cyc.min()) / (bs.max() - bs.min())
            ax.set_aspect(a / b)
        return ax

    def create_transit_mask(self, period, transit_time, duration):
        """Returns a boolean array that is ``True`` during transits and
        ``False`` elsewhere.

        This method supports multi-planet systems by allowing ``period``,
        ``transit_time``, and ``duration`` to be array-like lists of parameters.

        Parameters
        ----------
        period : `~astropy.units.Quantity`, float, or array-like
            Period(s) of the transits.
        duration : `~astropy.units.Quantity`, float, or array-like
            Duration(s) of the transits.
        transit_time : `~astropy.time.Time`, float, or array-like
            Transit midpoint(s) of the transits.

        Returns
        -------
        transit_mask : np.array of bool
            Mask that flags transits. Mask is ``True`` where there are transits.

        Examples
        --------
        You can create a transit mask for a single-planet system as follows::

            >>> import lightkurve as lk
            >>> lc = lk.LightCurve({'time': [1, 2, 3, 4, 5], 'flux': [1, 1, 1, 1, 1]})
            >>> lc.create_transit_mask(transit_time=2., period=2., duration=0.1)
            array([False,  True, False,  True, False])

        The method accepts lists of parameters to support multi-planet systems::

            >>> lc.create_transit_mask(transit_time=[2., 3.], period=[2., 10.], duration=[0.1, 0.1])
            array([False,  True,  True,  True, False])
        """
        # Convert Quantity objects to floats in units "day"
        period = _to_unitless_day(period)
        duration = _to_unitless_day(duration)

        # If ``transit_time`` is a ``Quantity```, attempt converting it to a ``Time`` object
        if isinstance(transit_time, Quantity):
            transit_time = Time(transit_time, format=self.time.format, scale=self.time.scale)

        # Ensure all parameters are 1D-arrays
        period = np.atleast_1d(period)
        duration = np.atleast_1d(duration)
        transit_time = np.atleast_1d(transit_time)

        # Make sure all params have the same number of entries
        n_planets = len(period)
        if any(len(param) != n_planets for param in [duration, transit_time]):
            raise ValueError(
                "period, duration, and transit_time must have "
                "the same number of values."
            )

        # Initialize an empty cadence mask
        in_transit = np.empty(len(self), dtype=bool)
        in_transit[:] = False

        # Create the transit mask
        for per, dur, tt in zip(period, duration, transit_time):
            if isinstance(tt, Time):
                # If a `Time` is passed, ensure it has the right format & scale
                tt = Time(tt, format=self.time.format, scale=self.time.scale).value
            hp = per / 2.0
            in_transit |= np.abs((self.time.value - tt + hp) % per - hp) < 0.5 * dur

        return in_transit

    def search_neighbors(
        self, limit: int = 10, radius: float = 3600.0, **search_criteria
    ):
        """Search the data archive at MAST for the most nearby light curves.

        By default, the 10 nearest neighbors located within 3600 arcseconds
        are returned. You can override these defaults by changing the `limit`
        and `radius` parameters.

        If the LightCurve object is a Kepler, K2, or TESS light curve,
        the default behavior of this method is to only return light curves
        obtained during the exact same quarter, campaign, or sector.
        This is useful to enable coeval light curves to be inspected for
        spurious noise signals in common between multiple neighboring targets.
        You can override this default behavior by passing a `mission`,
        `quarter`, `campaign`, or `sector` argument yourself.

        Please refer to the docstring of `search_lightcurve` for a complete
        list of search parameters accepted.

        Parameters
        ----------
        limit : int
            Maximum number of results to return.
        radius : float or `astropy.units.Quantity` object
            Conesearch radius.  If a float is given it will be assumed to be in
            units of arcseconds.
        **search_criteria : kwargs
            Extra criteria to be passed to `search_lightcurve`.

        Returns
        -------
        result : :class:`SearchResult` object
            Object detailing the neighbor light curves found, sorted by
            distance from the current light curve.
        """
        # Local import to avoid circular dependency
        from .search import search_lightcurve

        # By default, only return results from the same sector/quarter/campaign
        if (
            "mission" not in search_criteria
            and "sector" not in search_criteria
            and "quarter" not in search_criteria
            and "campaign" not in search_criteria
        ):
            mission = self.meta.get("MISSION", None)
            if mission == "TESS":
                search_criteria["sector"] = self.sector
            elif mission == "Kepler":
                search_criteria["quarter"] = self.quarter
            elif mission == "K2":
                search_criteria["campaign"] = self.campaign

        # Note: we increase `limit` by one below to account for the fact that the
        # current light curve will be returned by the search operation
        log.info(
            f"Started searching for up to {limit} neighbors within {radius} arcseconds."
        )
        result = search_lightcurve(
            f"{self.ra} {self.dec}", radius=radius, limit=limit + 1, **search_criteria
        )

        # Filter by distance > 0 to avoid returning the current light curve
        result = result[result.distance > 0]
        log.info(f"Found {len(result)} neighbors.")
        return result

    def head(self, n: int = 5):
        """Return the first n rows.

        Parameters
        ----------
        n : int
            Number of rows to return.

        Returns
        -------
        lc : LightCurve
            Light curve containing the first n rows.
        """
        return self[:n]

    def tail(self, n: int = 5):
        """Return the last n rows.

        Parameters
        ----------
        n : int
            Number of rows to return.

        Returns
        -------
        lc : LightCurve
            Light curve containing the last n rows.
        """
        return self[-n:]

    def truncate(self, before: float = None, after: float = None, column: str = "time"):
        """Truncates the light curve before and after some time value.

        Parameters
        ----------
        before : float
            Truncate all rows before this time value.
        after : float
            Truncate all rows after this time value.
        column : str, optional
            The name of the column on which the truncation is based. Defaults to 'time'.

        Returns
        -------
        truncated_lc : LightCurve
            The truncated light curve.
        """
        def _to_unitless(data):
            return np.asarray(getattr(data, "value", data))

        mask = np.ones(len(self), dtype=bool)
        if before:
            mask &= _to_unitless(getattr(self, column)) >= before
        if after:
            mask &= _to_unitless(getattr(self, column)) <= after
        return self[mask]


class FoldedLightCurve(LightCurve):
    """Subclass of `LightCurve` in which the ``time`` parameter represents phase values.

    Compared to the `~lightkurve.lightcurve.LightCurve` base class, this class
    has extra meta data entries (``period``, ``epoch_time``, ``epoch_phase``,
    ``wrap_phase``, ``normalize_phase``), an extra column (``time_original``),
    extra properties (``phase``, ``odd_mask``, ``even_mask``),
    and implements different plotting defaults.
    """

    @property
    def phase(self):
        """Alias for `LightCurve.time`."""
        return self.time

    @property
    def cycle(self):
        """The cycle of the correspond `time_original`.
        The first cycle is cycle 0, irrespective of whether it is a complete one or not.
        """
        epoch_time = self.meta.get("EPOCH_TIME")
        if epoch_time is None:
            # explicit check needed (cannot be the default value in get() function call above)
            # because Lightcurve.fold() will put an explicit None in meta, if epoch_time is not specified.
            epoch_time = self.time.min()
        cycle_epoch_start = epoch_time - self.period / 2
        result = np.asarray(np.floor(((self.time_original - cycle_epoch_start) / self.period).value), dtype=int)
        result = result - result.min()
        return result

    @property
    def odd_mask(self):
        """Boolean mask which flags the odd-numbered cycles (1, 3, 5, etc).

        This is useful for studying every second occurence of a signal.
        For example, in exoplanet searches, comparisons of odd and even transits
        can help confirm the planetary nature of a signal. Differences in the
        depth, duration, or shape of the odd- and even-numbered transits would
        indicate that the 'transits' are being caused by a near-equal mass
        eclipsing background binary, rather than a true transiting exoplanet.

        Examples
        --------
        You can can visualize the odd- and even-centered transits separately as
        follows:

            >>> f = lc.fold(...)  # doctest: +SKIP
            >>> f[f.odd_mask].scatter()  # doctest: +SKIP
            >>> f[f.even_mask].scatter()  # doctest: +SKIP
        """
        return self.cycle % 2 == 1

    @property
    def even_mask(self):
        """Boolean mask which flags the even-numbered cycles (2, 4, 6, etc).

        See the documentation of `odd_mask` for examples.
        """
        return ~self.odd_mask

    def _set_xlabel(self, kwargs):
        """Helper function for plot, scatter, and errorbar.
        Ensures the xlabel is correctly set for folded light curves.
        """
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "Phase"
            if isinstance(self.time, TimeDelta):
                kwargs["xlabel"] += f" [{self.time.format.upper()}]"
        return kwargs

    def plot(self, **kwargs):
        """Plot the folded light curve using matplotlib's
        `~matplotlib.pyplot.plot` method.

        See `LightCurve.plot` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments to be passed to `LightCurve.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        kwargs = self._set_xlabel(kwargs)
        return super(FoldedLightCurve, self).plot(**kwargs)

    def scatter(self, **kwargs):
        """Plot the folded light curve using matplotlib's `~matplotlib.pyplot.scatter` method.

        See `LightCurve.scatter` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments to be passed to `LightCurve.scatter`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        kwargs = self._set_xlabel(kwargs)
        return super(FoldedLightCurve, self).scatter(**kwargs)

    def errorbar(self, **kwargs):
        """Plot the folded light curve using matplotlib's
        `~matplotlib.pyplot.errorbar` method.

        See `LightCurve.scatter` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments to be passed to `LightCurve.scatter`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        kwargs = self._set_xlabel(kwargs)
        return super(FoldedLightCurve, self).errorbar(**kwargs)

    def plot_river(self, **kwargs):
        """Plot the folded light curve in a river style.

        See `~LightCurve.plot_river` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments to be passed to `~LightCurve.plot_river`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        ax = super(FoldedLightCurve, self).plot_river(
            period=self.period, epoch_time=self.epoch_time, **kwargs
        )
        return ax


class KeplerLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    to represent data from NASA's Kepler and K2 mission."""

    _deprecated_keywords = (
        "targetid",
        "label",
        "time_format",
        "time_scale",
        "flux_unit",
        "quality_bitmask",
        "channel",
        "campaign",
        "quarter",
        "mission",
        "ra",
        "dec",
    )

    _default_time_format = "bkjd"

    @classmethod
    def read(cls, *args, **kwargs):
        """Returns a `KeplerLightCurve` by reading the given file.

        Parameters
        ----------
        filename : str
            Local path or remote url of a Kepler light curve FITS file.
        flux_column : str, optional
            The column in the FITS file to be read as `flux`. Defaults to 'pdcsap_flux'.
            Typically 'pdcsap_flux' or 'sap_flux'.
        quality_bitmask : str or int, optional
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored
                * "default": cadences with severe quality issues will be ignored
                * "hard": more conservative choice of flags to ignore
                  This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  This mask is not recommended.

            See the :class:`KeplerQualityFlags <lightkurve.utils.KeplerQualityFlags>` class for details on the bitmasks.
        format : str, optional
            The format of the Kepler FITS file. Should be one of 'kepler', 'k2sff', 'everest'. Defaults to 'kepler'.
        """
        # Default to Kepler file format
        if kwargs.get("format") is None:
            kwargs["format"] = "kepler"
        return super().read(*args, **kwargs)

    def to_fits(
        self,
        path=None,
        overwrite=False,
        flux_column_name="FLUX",
        aperture_mask=None,
        **extra_data,
    ):
        """Writes the KeplerLightCurve to a FITS file.

        Parameters
        ----------
        path : string, default None
            File path, if `None` returns an astropy.io.fits.HDUList object.
        overwrite : bool
            Whether or not to overwrite the file
        flux_column_name : str
            The name of the label for the FITS extension, e.g. SAP_FLUX or FLUX
        aperture_mask : array-like
            Optional 2D aperture mask to save with this lightcurve object, if
            defined.  The mask can be either a boolean mask or an integer mask
            mimicking the Kepler/TESS convention; boolean masks are
            automatically converted to the Kepler/TESS conventions
        extra_data : dict
            Extra keywords or columns to include in the FITS file.
            Arguments of type str, int, float, or bool will be stored as
            keywords in the primary header.
            Arguments of type np.array or list will be stored as columns
            in the first extension.

        Returns
        -------
        hdu : astropy.io.fits
            Returns an astropy.io.fits object if path is None
        """
        kepler_specific_data = {
            "TELESCOP": "KEPLER",
            "INSTRUME": "Kepler Photometer",
            "OBJECT": "{}".format(self.targetid),
            "KEPLERID": self.targetid,
            "CHANNEL": self.channel,
            "MISSION": self.mission,
            "RA_OBJ": self.ra,
            "DEC_OBJ": self.dec,
            "EQUINOX": 2000,
            "DATE-OBS": Time(self.time[0] + 2454833.0, format=("jd")).isot,
            "SAP_QUALITY": self.quality,
            "MOM_CENTR1": self.centroid_col,
            "MOM_CENTR2": self.centroid_row,
        }

        for kw in kepler_specific_data:
            if ~np.asarray([kw.lower == k.lower() for k in extra_data]).any():
                extra_data[kw] = kepler_specific_data[kw]
        hdu = super(KeplerLightCurve, self).to_fits(
            path=None, overwrite=overwrite, **extra_data
        )

        hdu[0].header["QUARTER"] = self.meta.get("QUARTER")
        hdu[0].header["CAMPAIGN"] = self.meta.get("CAMPAIGN")

        hdu = _make_aperture_extension(hdu, aperture_mask)

        if path is not None:
            hdu.writeto(path, overwrite=overwrite, checksum=True)
        else:
            return hdu


class TessLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    to represent data from NASA's TESS mission."""

    _deprecated_keywords = (
        "targetid",
        "label",
        "time_format",
        "time_scale",
        "flux_unit",
        "quality_bitmask",
        "sector",
        "camera",
        "ccd",
        "mission",
        "ra",
        "dec",
    )

    _default_time_format = "btjd"

    @classmethod
    def read(cls, *args, **kwargs):
        """Returns a `TessLightCurve` by reading the given file.

        Parameters
        ----------
        filename : str
            Local path or remote url of a TESS light curve FITS file.
        flux_column : str, optional
            The column in the FITS file to be read as `flux`. Defaults to 'pdcsap_flux'.
            Typically 'pdcsap_flux' or 'sap_flux'.
        quality_bitmask : str or int, optional
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored
                * "default": cadences with severe quality issues will be ignored
                * "hard": more conservative choice of flags to ignore
                  This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  This mask is not recommended.

            See the :class:`TessQualityFlags <lightkurve.utils.TessQualityFlags>` class for details on the bitmasks.
        """
        # Default to TESS file format
        if kwargs.get("format") is None:
            kwargs["format"] = "tess"
        return super().read(*args, **kwargs)

    def to_fits(
        self,
        path=None,
        overwrite=False,
        flux_column_name="FLUX",
        aperture_mask=None,
        **extra_data,
    ):
        """Writes the KeplerLightCurve to a FITS file.

        Parameters
        ----------
        path : string, default None
            File path, if `None` returns an astropy.io.fits.HDUList object.
        overwrite : bool
            Whether or not to overwrite the file
        flux_column_name : str
            The name of the label for the FITS extension, e.g. SAP_FLUX or FLUX
        aperture_mask : array-like
            Optional 2D aperture mask to save with this lightcurve object, if
            defined.  The mask can be either a boolean mask or an integer mask
            mimicking the Kepler/TESS convention; boolean masks are
            automatically converted to the Kepler/TESS conventions
        extra_data : dict
            Extra keywords or columns to include in the FITS file.
            Arguments of type str, int, float, or bool will be stored as
            keywords in the primary header.
            Arguments of type np.array or list will be stored as columns
            in the first extension.

        Returns
        -------
        hdu : astropy.io.fits
            Returns an astropy.io.fits object if path is None
        """
        tess_specific_data = {
            "OBJECT": "{}".format(self.targetid),
            "MISSION": self.meta.get("MISSION"),
            "RA_OBJ": self.meta.get("RA"),
            "TELESCOP": self.meta.get("MISSION"),
            "CAMERA": self.meta.get("CAMERA"),
            "CCD": self.meta.get("CCD"),
            "SECTOR": self.meta.get("SECTOR"),
            "TARGETID": self.meta.get("TARGETID"),
            "DEC_OBJ": self.meta.get("DEC"),
            "MOM_CENTR1": self.centroid_col,
            "MOM_CENTR2": self.centroid_row,
        }

        for kw in tess_specific_data:
            if ~np.asarray([kw.lower == k.lower() for k in extra_data]).any():
                extra_data[kw] = tess_specific_data[kw]
        hdu = super(TessLightCurve, self).to_fits(
            path=None, overwrite=overwrite, **extra_data
        )

        # We do this because the TESS file format is subtly different in the
        #    name of this column.
        hdu[1].columns.change_name("SAP_QUALITY", "QUALITY")

        hdu = _make_aperture_extension(hdu, aperture_mask)

        if path is not None:
            hdu.writeto(path, overwrite=overwrite, checksum=True)
        else:
            return hdu


# Helper functions


def _boolean_mask_to_bitmask(aperture_mask):
    """Takes in an aperture_mask and returns a Kepler-style bitmask

    Parameters
    ----------
    aperture_mask : array-like
        2D aperture mask. The mask can be either a boolean mask or an integer
        mask mimicking the Kepler/TESS convention; boolean or boolean-like masks
        are converted to the Kepler/TESS conventions.  Kepler bitmasks are
        returned unchanged except for possible datatype conversion.

    Returns
    -------
    bitmask : numpy uint8 array
        A bitmask incompletely mimicking the Kepler/TESS convention: Bit 2,
        value = 3, means "pixel was part of the custom aperture".  The other
        bits have no meaning and are currently assigned a value of 1.
    """
    # Masks can either be boolean input or Kepler pipeline style
    clean_mask = np.nan_to_num(aperture_mask)

    contains_bit2 = (clean_mask.astype(np.int_) & 2).any()
    all_zeros_or_ones = (clean_mask.dtype in ["float", "int"]) & (
        (set(np.unique(clean_mask)) - {0, 1}) == set()
    )
    is_bool_mask = (aperture_mask.dtype == "bool") | all_zeros_or_ones

    if is_bool_mask:
        out_mask = np.ones(aperture_mask.shape, dtype=np.uint8)
        out_mask[aperture_mask == 1] = 3
        out_mask = out_mask.astype(np.uint8)
    elif contains_bit2:
        out_mask = aperture_mask.astype(np.uint8)
    else:
        log.warn(
            "The input aperture mask must be boolean or follow the "
            "Kepler-pipeline standard; returning None."
        )
        out_mask = None
    return out_mask


def _make_aperture_extension(hdu_list, aperture_mask):
    """Returns an `ImageHDU` object containing the 'APERTURE' extension
    of a light curve file."""
    if aperture_mask is not None:
        bitmask = _boolean_mask_to_bitmask(aperture_mask)
        hdu = fits.ImageHDU(bitmask)
        hdu.header["EXTNAME"] = "APERTURE"
        hdu_list.append(hdu)
    return hdu_list
