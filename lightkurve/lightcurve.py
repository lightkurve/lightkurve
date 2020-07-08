"""Defines LightCurve, KeplerLightCurve, and TessLightCurve."""
import os
import datetime
import logging
import warnings

from typing import Iterable

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib
from matplotlib import pyplot as plt
from copy import deepcopy

from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.units import Quantity
from astropy.timeseries import TimeSeries, aggregate_downsample
from astropy.table import vstack
from astropy.utils.decorators import deprecated, deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning

from . import PACKAGEDIR, MPLSTYLE
from .utils import (running_mean, bkjd_to_astropy_time, btjd_to_astropy_time,
    validate_method, _query_solar_system_objects
)
from .utils import LightkurveWarning, LightkurveDeprecationWarning


__all__ = ['LightCurve', 'KeplerLightCurve', 'TessLightCurve', 'FoldedLightCurve']

log = logging.getLogger(__name__)


class LightCurve(TimeSeries):
    """Class to hold time series brightness data for an astronomical object.

    Compared to the generic `~astropy.timeseries.TimeSeries` class, `LightCurve`
    ensures that each object has `time`, `flux`, and `flux_err` columns.
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
    _required_columns = ['time', 'flux', 'flux_err']

    # The following keywords were removed in Lightkurve v2.0.
    # Their use will trigger a warning.
    _deprecated_keywords = ('targetid', 'label', 'time_format', 'time_scale',
                            'flux_unit')
    _deprecated_column_keywords = ['centroid_col', 'centroid_row',
                                   'cadenceno', 'quality']

    # If an iterable is passed for ``time``, we will initialize an AstroPy
    # ``Time`` object using the following format and scale:
    _default_time_format = "jd"
    _default_time_scale = "tdb"

    def __init__(self, data=None, *args, time=None, flux=None, flux_err=None, **kwargs):
        # Delay checking for required columns until the end
        self._required_columns_relax = True

        # Lightkurve v1.x supported passing time, flux, and flux_err as
        # positional arguments. We support it here for backwards compatibility.
        if len(args) in [1, 2]:
            warnings.warn("passing flux as a positional argument is deprecated"
                          ", please use ``flux=...`` instead.",
                          LightkurveDeprecationWarning)
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

        self._required_columns = kwargs.pop("_required_columns",
                                            self._required_columns)

        # We are tolerant of missing time if flux is given
        if time is None and flux is not None:
            time = np.arange(len(flux))
        # We are tolerant of missing time format
        if time is not None and not isinstance(time, Time):
            # Lightkurve v1.x supported specifying the time_format
            # as a constructor kwarg
            time = Time(time,
                        format=deprecated_kws.get("time_format", self._default_time_format),
                        scale=deprecated_kws.get("time_scale", self._default_time_scale))

        super().__init__(data=data, time=time, **kwargs)

        # For some operations, an empty time series needs to be created, then
        # columns added one by one. We should check that when columns are added
        # manually, time is added first and is of the right type.
        if data is None and time is None and flux is None and flux_err is None:
            self._required_columns_relax = True
            return

        # Ensure the required columns are available
        if flux is None:
            flux = np.empty(len(self))
            flux[:] = np.nan
        if not isinstance(flux, Quantity):
            flux = Quantity(flux, deprecated_kws.get("flux_unit"))
        if "flux" not in self.columns:
            self.add_column(flux, name="flux", index=1)

        if flux_err is None:
            flux_err = np.empty(len(self))
            flux_err[:] = np.nan
        if not isinstance(flux_err, Quantity):
            flux_err = Quantity(flux_err, deprecated_kws.get("flux_unit"))
        if "flux_err" not in self.columns:
            self.add_column(flux_err, name="flux_err", index=2)

        # Backwards compatibility with Lightkurve v1.x
        # Ensure attributes are set if passed via deprecated kwargs
        for kw in deprecated_kws:
            if kw not in self.meta:
                self.meta[kw] = deprecated_kws[kw]
        # Ensure columns are set if passed via deprecated kwargs
        for kw in deprecated_column_kws:
            if kw not in self.meta and kw not in self.columns:
                self.add_column(deprecated_column_kws[kw], name=kw)

        # Ensure all columns are Quantity objects
        for col in self.columns:
            if not isinstance(self[col], (Quantity, Time)):
                self.replace_column(col, Quantity(self[col], dtype=self[col].dtype))

        # Ensure flux and flux_err have the same units
        if self['flux'].unit != self['flux'].unit:
            raise ValueError("flux and flux_err must have the same units")

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
        elif ('_meta' in self.__dict__) and (name in self.__dict__['_meta']):
            return self.__dict__['_meta'][name]
        raise AttributeError(f"object has no attribute {name}")

    def __setattr__(self, name, value, **kwargs):
        """To get copied, attributes have to be stored in the meta dictionary!"""
        if ('columns' in self.__dict__) and (name in self.__dict__['columns']):
            if not isinstance(value, Quantity):
                value = Quantity(value, dtype=value.dtype)
            self.replace_column(name, value)
        elif ('_meta' in self.__dict__) and (name in self.__dict__['_meta']):
            self.__dict__['_meta'][name] = value
        else:
            super().__setattr__(name, value, **kwargs)

    def _base_repr_(self, descr_vals=None, **kwargs):
        """Defines the description shown by `__repr__` and `_html_repr_`."""
        if descr_vals is None:
            descr_vals = [self.__class__.__name__]
            if self.masked:
                descr_vals.append('masked=True')
            if hasattr(self, "targetid"):
                descr_vals.append(f'targetid={self.targetid}')
            descr_vals.append('length={}'.format(len(self)))
        return super()._base_repr_(descr_vals=descr_vals, **kwargs)

    # Define `time`, `flux`, `flux_err` as class attributes to enable IDE
    # of these required columns auto-completion.

    @property
    def time(self):
        """The time values."""
        return self['time']

    @time.setter
    def time(self, time):
        self['time'] = time

    @property
    def flux(self):
        """The brightness values."""
        return self['flux']

    @flux.setter
    def flux(self, flux):
        self['flux'] = flux

    @property
    def flux_err(self):
        """The brightness uncertainty."""
        return self['flux_err']

    @flux_err.setter
    def flux_err(self, flux_err):
        self['flux_err'] = flux_err

    # Define deprecated attributes for compatibility with Lightkurve v1.x:

    @property
    @deprecated("2.0", alternative="time.format",
                warning_type=LightkurveDeprecationWarning)
    def time_format(self):
        return self.time.format

    @property
    @deprecated("2.0", alternative="time.scale",
                warning_type=LightkurveDeprecationWarning)
    def time_scale(self):
        return self.time.scale

    @property
    @deprecated("2.0", alternative="time",
                warning_type=LightkurveDeprecationWarning)
    def astropy_time(self):
        return self.time

    @property
    @deprecated("2.0", alternative="flux.unit",
                warning_type=LightkurveDeprecationWarning)
    def flux_unit(self):
        return self.flux.unit

    @property
    @deprecated("2.0", alternative="flux",
                warning_type=LightkurveDeprecationWarning)
    def flux_quantity(self):
        return self.flux

    @property
    @deprecated("2.0", alternative="fits.open(lc.filename)",
                warning_type=LightkurveDeprecationWarning)
    def hdu(self):
        return fits.open(self.filename)

    @property
    @deprecated("2.0", warning_type=LightkurveDeprecationWarning)
    def SAP_FLUX(self):
        """A copy of the light curve in which `lc.flux = lc.sap_flux`
        and `lc.flux_err = lc.sap_flux_err`.  It is provided for backwards-
        compatibility with Lightkurve v1.x and will be removed soon."""
        lc = self.copy()
        lc['flux'] = lc['sap_flux']
        lc['flux_err'] = lc['sap_flux_err']
        return lc

    @property
    @deprecated("2.0", warning_type=LightkurveDeprecationWarning)
    def PDCSAP_FLUX(self):
        """A copy of the light curve in which `lc.flux = lc.pdcsap_flux`
        and `lc.flux_err = lc.pdcsap_flux_err`.  It is provided for backwards-
        compatibility with Lightkurve v1.x and will be removed soon."""
        lc = self.copy()
        lc['flux'] = lc['pdcsap_flux']
        lc['flux_err'] = lc['pdcsap_flux_err']
        return lc

    def __add__(self, other):
        newlc = self.copy()
        if isinstance(other, LightCurve):
            if len(self) != len(other):
                raise ValueError("Cannot add LightCurve objects because "
                                 "they do not have equal length ({} vs {})."
                                 "".format(len(self), len(other)))
            if np.any(self.time != other.time):
                warnings.warn("Two LightCurve objects with inconsistent time "
                              "values are being added.",
                              LightkurveWarning)
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
                raise ValueError("Cannot multiply LightCurve objects because "
                                 "they do not have equal length ({} vs {})."
                                 "".format(len(self), len(other)))
            if np.any(self.time != other.time):
                warnings.warn("Two LightCurve objects with inconsistent time "
                              "values are being multiplied.",
                              LightkurveWarning)
            newlc.flux = self.flux * other.flux
            # Applying standard uncertainty propagation, cf.
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
            newlc.flux_err = abs(newlc.flux) * np.hypot(self.flux_err / self.flux, other.flux_err / other.flux)
        else:
            newlc.flux = other * self.flux
            newlc.flux_err = abs(other) * self.flux_err
        return newlc

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1. / other)

    def __rtruediv__(self, other):
        newlc = self.copy()
        if isinstance(other, LightCurve):
            if len(self) != len(other):
                raise ValueError("Cannot divide LightCurve objects because "
                                 "they do not have equal length ({} vs {})."
                                 "".format(len(self), len(other)))
            if np.any(self.time != other.time):
                warnings.warn("Two LightCurve objects with inconsistent time "
                              "values are being divided.",
                              LightkurveWarning)
            newlc.flux = other.flux / self.flux
            newlc.flux_err = abs(newlc.flux) * np.hypot(self.flux_err / self.flux, other.flux_err / other.flux)
        else:
            newlc.flux = other / self.flux
            newlc.flux_err = abs((other * self.flux_err) / (self.flux**2))
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
        deprecated_properties += ['flux_quantity', 'SAP_FLUX', 'PDCSAP_FLUX',
                                  'astropy_time', 'hdu']
        for attr in dir(self):
            if not attr.startswith('_') and attr not in deprecated_properties:
                try:
                    res = getattr(self, attr)
                except Exception:
                    continue
                if callable(res):
                    continue
                attrs[attr] = {'res': res}
                if isinstance(res, int):
                    attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'int'
                elif isinstance(res, np.ndarray):
                    attrs[attr]['print'] = 'array {}'.format(res.shape)
                    attrs[attr]['type'] = 'array'
                elif isinstance(res, list):
                    attrs[attr]['print'] = 'list length {}'.format(len(res))
                    attrs[attr]['type'] = 'list'
                elif isinstance(res, str):
                    if res == '':
                        attrs[attr]['print'] = '{}'.format('None')
                    else:
                        attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'str'
                elif attr == 'wcs':
                    attrs[attr]['print'] = 'astropy.wcs.wcs.WCS'
                    attrs[attr]['type'] = 'other'
                else:
                    attrs[attr]['print'] = '{}'.format(type(res))
                    attrs[attr]['type'] = 'other'
        output = Table(names=['Attribute', 'Description'], dtype=[object, object])
        idx = 0
        types = ['int', 'str', 'list', 'array', 'other']
        for typ in types:
            for attr, dic in attrs.items():
                if dic['type'] == typ:
                    output.add_row([attr, dic['print']])
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
            raise ValueError("the `inplace` parameter is no longer supported "
                             "as of Lightkurve v2.0")
        if not hasattr(others, '__iter__'):
            others = (others,)
        # Need `join_type='inner'` until AstroPy supports masked Quantities
        return vstack((self, *others), join_type='inner', metadata_conflicts='silent')

    def flatten(self, window_length=101, polyorder=2, return_trend=False,
                break_tolerance=5, niters=3, sigma=3, mask=None, **kwargs):
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
        # No NaNs
        mask &= np.isfinite(self.flux)
        # No outliers
        mask &= np.nan_to_num(np.abs(self.flux - np.nanmedian(self.flux))) <= (np.nanstd(self.flux) * sigma)
        for iter in np.arange(0, niters):
            if break_tolerance is None:
                break_tolerance = np.nan
            if polyorder >= window_length:
                polyorder = window_length - 1
                log.warning("polyorder must be smaller than window_length, "
                            "using polyorder={}.".format(polyorder))
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
                        warnings.simplefilter('ignore', FutureWarning)
                        trsig = signal.savgol_filter(x=self.flux.value[mask][l:h],
                                                                 window_length=window_length,
                                                                 polyorder=polyorder,
                                                                 **kwargs)
                        trend_signal[l:h] = Quantity(trsig, trend_signal.unit)
            # Ignore outliers; note we add `1e-14` below to avoid detecting
            # outliers which are merely caused by numerical noise.
            mask1 = np.nan_to_num(np.abs(self.flux[mask] - trend_signal)) <\
                    (np.nanstd(self.flux[mask] - trend_signal) * sigma + Quantity(1e-14, self.flux.unit))
            f = interp1d(self.time.value[mask][mask1], trend_signal[mask1], fill_value='extrapolate')
            trend_signal = f(self.time.value)
            mask[mask] &= mask1

        flatten_lc = self.copy()
        with warnings.catch_warnings():
            # ignore invalid division warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            flatten_lc.flux = flatten_lc.flux / trend_signal
            flatten_lc.flux_err = flatten_lc.flux_err / trend_signal
        if return_trend:
            trend_lc = self.copy()
            trend_lc.flux = trend_signal
            return flatten_lc, trend_lc
        return flatten_lc

    @deprecated_renamed_argument('transit_midpoint', 'epoch_time', '2.0',
                                 warning_type=LightkurveDeprecationWarning)
    @deprecated_renamed_argument('t0', 'epoch_time', '2.0',
                                 warning_type=LightkurveDeprecationWarning)
    def fold(self, period=None, epoch_time=None, epoch_phase=0,
             wrap_phase=None, normalize_phase=False):
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
            epoch_time = Time(epoch_time, format=self.time.format, scale=self.time.scale)
        if epoch_phase is not None and not isinstance(epoch_phase, Quantity) and not normalize_phase:
            epoch_phase *= u.day
        if wrap_phase is not None and not isinstance(wrap_phase, Quantity):
            wrap_phase *= u.day

        # Warn if `epoch_time` appears to use the wrong format
        if epoch_time is not None and epoch_time.value > 2450000:
            if self.time.format == 'bkjd':
                warnings.warn('`epoch_time` appears to be given in JD, '
                              'however the light curve time uses BKJD '
                              '(i.e. JD - 2454833).', LightkurveWarning)
            elif self.time.format == 'btjd':
                warnings.warn('`epoch_time` appears to be given in JD, '
                              'however the light curve time uses BTJD '
                              '(i.e. JD - 2457000).', LightkurveWarning)

        ts = super().fold(period=period, epoch_time=epoch_time,
                          epoch_phase=epoch_phase, wrap_phase=wrap_phase,
                          normalize_phase=normalize_phase)

        # The folded time would pass the `TimeSeries` validation check if
        # `normalize_phase=True`, so creating a `FoldedLightCurve` object
        # requires the following three-step workaround:
        # 1. Give the folded light curve a valid time column again
        with ts._delay_required_column_checks():
            folded_time = ts.time.copy()
            ts.remove_column('time')
            ts.add_column(self.time, name='time', index=0)
        # 2. Create the folded object
        lc = FoldedLightCurve(data=ts)
        # 3. Restore the folded time
        with lc._delay_required_column_checks():
            lc.remove_column('time')
            lc.add_column(folded_time, name='time', index=0)

        # Add extra column and meta data specific to FoldedLightCurve
        lc.add_column(self.time.copy(), name="time_original", index=len(self._required_columns))
        lc.meta['period'] = period
        lc.meta['epoch_time'] = epoch_time
        lc.meta['epoch_phase'] = epoch_phase
        lc.meta['wrap_phase'] = wrap_phase
        lc.meta['normalize_phase'] = normalize_phase
        lc.sort('time')

        return lc

    def normalize(self, unit='unscaled'):
        """Returns a normalized version of the light curve.

        The normalized light curve is obtained by dividing the ``flux`` and
        ``flux_err`` object attributes by the by the median flux.
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
        validate_method(unit, ['unscaled', 'percent', 'ppt', 'ppm'])
        median_flux = np.nanmedian(self.flux)
        std_flux = np.nanstd(self.flux)

        # If the median flux is within half a standard deviation from zero, the
        # light curve is likely zero-centered and normalization makes no sense.
        if (median_flux == 0) or (np.isfinite(std_flux) and (np.abs(median_flux) < 0.5*std_flux)):
            warnings.warn("The light curve appears to be zero-centered "
                          "(median={:.2e} +/- {:.2e}); `normalize()` will divide "
                          "the light curve by a value close to zero, which is "
                          "probably not what you want."
                          "".format(median_flux, std_flux),
                          LightkurveWarning)
        # If the median flux is negative, normalization will invert the light
        # curve and makes no sense.
        if median_flux < 0:
            warnings.warn("The light curve has a negative median flux ({:.2e});"
                          " `normalize()` will therefore divide by a negative "
                          "number and invert the light curve, which is probably"
                          "not what you want".format(median_flux),
                          LightkurveWarning)
        # Warn if the light curve was already normalized before
        if self.meta.get("normalized"):
            warnings.warn("The light curve already appears to be in relative "
                          "units; `normalize()` will convert the light curve "
                          "into relative units for a second time, which is "
                          "probably not what you want.".format(self.flux.unit),
                          LightkurveWarning)

        # Create a new light curve instance and normalize its values
        lc = self.copy()
        lc.flux = lc.flux / median_flux
        lc.flux_err = lc.flux_err / median_flux
        if not lc.flux.unit:
            lc.flux *= u.dimensionless_unscaled
        if not lc.flux_err.unit:
            lc.flux_err *= u.dimensionless_unscaled

        # Set the desired relative (dimensionless) units
        if unit == 'percent':
            lc.flux = lc.flux.to(u.percent)
            lc.flux_err = lc.flux_err.to(u.percent)
        elif unit == 'ppt':  # parts per thousand
            # ppt is not included in astropy, so we define it here
            ppt = u.def_unit(['ppt', 'parts per thousand'], u.Unit(1e-3))
            lc.flux = lc.flux.to(ppt)
        elif unit == 'ppm':  # parts per million
            lc.flux = lc.flux.to(u.cds.ppm)

        lc.meta['normalized'] = True
        return lc

    def remove_nans(self):
        """Removes cadences where the flux is NaN.

        Returns
        -------
        clean_lightcurve : `LightCurve`
            A new light curve object from which NaNs fluxes have been removed.
        """
        return self[~np.isnan(self.flux)]  # This will return a sliced copy

    def fill_gaps(self, method: str = 'gaussian_noise'):
        """Fill in gaps in time.

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
        #nlc = lc.copy()
        newdata = {}

        # Find missing time points
        # Most precise method, taking into account time variation due to orbit
        if hasattr(lc, 'cadenceno'):
            dt = lc.time.value - np.median(np.diff(lc.time.value)) * lc.cadenceno
            ncad = np.arange(lc.cadenceno[0], lc.cadenceno[-1] + 1, 1)
            in_original = np.in1d(ncad, lc.cadenceno)
            ncad = ncad[~in_original]
            ndt = np.interp(ncad, lc.cadenceno, dt)

            ncad = np.append(ncad, lc.cadenceno)
            ndt = np.append(ndt, dt)
            ncad, ndt = ncad[np.argsort(ncad)], ndt[np.argsort(ncad)]
            ntime = ndt + np.median(np.diff(lc.time.value)) * ncad
            newdata['cadenceno'] = ncad
        else:
            # Less precise method
            dt = np.nanmedian(lc.time.value[1::] - lc.time.value[:-1:])
            ntime = [lc.time.value[0]]
            for t in lc.time.value[1::]:
                prevtime = ntime[-1]
                while (t - prevtime) > 1.2*dt:
                    ntime.append(prevtime + dt)
                    prevtime = ntime[-1]
                ntime.append(t)
            ntime = np.asarray(ntime, float)
            in_original = np.in1d(ntime, lc.time.value)
        
        # Fill in time points
        newdata['time'] = Time(ntime, format=lc.time.format, scale=lc.time.scale)
        f = np.zeros(len(ntime))
        f[in_original] = np.copy(lc.flux)
        fe = np.zeros(len(ntime))
        fe[in_original] = np.copy(lc.flux_err)

        fe[~in_original] = np.interp(ntime[~in_original], lc.time.value, lc.flux_err)
        if method == 'gaussian_noise':
            try:
                std = lc.estimate_cdpp()*1e-6
            except:
                std = lc.flux.std()
            f[~in_original] = np.random.normal(lc.flux.mean(), std.value, (~in_original).sum())
        else:
            raise NotImplementedError("No such method as {}".format(method))

        newdata['flux'] = f
        newdata['flux_err'] = fe

        if hasattr(lc, 'quality'):
            quality = np.zeros(len(ntime), dtype=lc.quality.dtype)
            quality[in_original] = np.copy(lc.quality)
            quality[~in_original] += 65536
            newdata['quality'] = quality
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

    def remove_outliers(self, sigma=5., sigma_lower=None, sigma_upper=None,
                        return_mask=False, **kwargs):
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
        # First, we create the outlier mask using AstroPy's sigma_clip function
        with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
            warnings.simplefilter("ignore")
            outlier_mask = sigma_clip(data=self.flux,
                                      sigma=sigma,
                                      sigma_lower=sigma_lower,
                                      sigma_upper=sigma_upper,
                                      **kwargs).mask
        # Second, we return the masked light curve and optionally the mask itself
        if return_mask:
            return self.copy()[~outlier_mask], outlier_mask
        return self.copy()[~outlier_mask]

    @deprecated_renamed_argument('binsize', new_name=None, since='2.0',
                                 warning_type=LightkurveDeprecationWarning,
                                 alternative='time_bin_size')
    def bin(self, time_bin_size=None, time_bin_start=None, n_bins=None,
            aggregate_func=None, binsize=None):
        """Bins a lightcurve in equally-spaced bins in time.

        If the original light curve contains flux uncertainties (``flux_err``),
        the binned lightcurve will report the root-mean-square error.
        If no uncertainties are included, the binned curve will return the
        standard deviation of the data.

        Parameters
        ----------
        time_bin_size : `~astropy.units.Quantity`, float
            The time interval for the binned time series. (Default unit: days.)
        time_bin_start : `~astropy.time.Time`, optional
            The start time for the binned time series. Defaults to the first
            time in the sampled time series.
        n_bins : int, optional
            The number of bins to use. Defaults to the number needed to fit all
            the original points.
        aggregate_func : callable, optional
            The function to use for combining points in the same bin. Defaults
            to np.nanmean.
        binsize : int
            DEPRECATED.

        Returns
        -------
        binned_lc : `LightCurve`
            A new light curve which has been binned.
        """
        # Backwards compatibility with Lightkurve v1.x
        if time_bin_size is None and binsize is not None:
            time_bin_size = (self.time[binsize] - self.time[0]).to(u.day)

        if time_bin_size is None:
            time_bin_size = 0.5*u.day
        if not isinstance(time_bin_size, Quantity):
            time_bin_size *= u.day
        if time_bin_start is None:
            time_bin_start = self.time[0]
        if not isinstance(time_bin_start, Time):
            time_bin_start = Time(time_bin_start, format=self.time.format,
                                  scale=self.time.scale)

        # Call AstroPy's aggregate_downsample
        with warnings.catch_warnings():
            # ignore uninteresting empty slice warnings
            warnings.simplefilter("ignore", (RuntimeWarning, AstropyUserWarning))
            ts = aggregate_downsample(self,
                                      time_bin_size=time_bin_size,
                                      n_bins=n_bins,
                                      time_bin_start=time_bin_start,
                                      aggregate_func=aggregate_func)

            # If `flux_err` is populated, assume the errors combine as the root-mean-square
            if np.any(np.isfinite(self.flux_err)):
                rmse_func = lambda x: np.sqrt(np.nansum(x**2))/len(np.atleast_1d(x)) \
                                    if np.any(np.isfinite(x)) else np.nan
                ts_err = aggregate_downsample(self,
                                              time_bin_size=time_bin_size,
                                              n_bins=n_bins,
                                              time_bin_start=time_bin_start,
                                              aggregate_func=rmse_func)
                ts['flux_err'] = ts_err['flux_err']
            # If `flux_err` is unavailable, populate `flux_err` as nanstd(flux)
            else:
                ts_err = aggregate_downsample(self,
                                              time_bin_size=time_bin_size,
                                              n_bins=n_bins,
                                              time_bin_start=time_bin_start,
                                              aggregate_func=np.nanstd)
                ts['flux_err'] = ts_err['flux']

        # Prepare a LightCurve object by ensuring there is a time column
        ts._required_columns = []
        ts.add_column(ts.time_bin_start + ts.time_bin_size / 2., name="time")

        # Ensure the required columns appear in the correct order
        for idx, colname in enumerate(self.__class__._required_columns):
            tmpcol = ts[colname]
            ts.remove_column(colname)
            ts.add_column(tmpcol, name=colname, index=idx)

        return self.__class__(ts)

    def estimate_cdpp(self, transit_duration=13, savgol_window=101,
                      savgol_polyorder=2, sigma=5.) -> float:
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
            raise ValueError("transit_duration must be an integer in units "
                             "number of cadences, got {}.".format(transit_duration))

        detrended_lc = self.flatten(window_length=savgol_window,
                                    polyorder=savgol_polyorder)
        cleaned_lc = detrended_lc.remove_outliers(sigma=sigma)
        with warnings.catch_warnings():  # ignore "already normalized" message
            warnings.filterwarnings("ignore", message=".*already.*")
            normalized_lc = cleaned_lc.normalize("ppm")
        mean = running_mean(data=normalized_lc.flux, window_size=transit_duration)
        return np.std(mean)

    def query_solar_system_objects(self, cadence_mask='outliers', radius=None,
                                   sigma=3, location=None, cache=True, return_mask=False):
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

        Notes:
        * This method will use the `ra` and `dec` properties of the `LightCurve`
          object to determine the position of the search cone.
        * The size of the search cone is 15 spacecraft pixels by default. You
          can change this by passing the `radius` parameter (unit: degrees).
        * This method will only search points in time during which he light
          curve showed 3-sigma outliers in flux. You can override this behavior
          and search all times by passing the `cadence_mask='all'` argument,
          but this will be much slower.

        Parameters
        ----------
        cadence_mask : str or bool
            mask in time to select which frames or points should be searched for SSOs.
            Default "outliers" will search for SSOs at points that are `sigma` from the mean.
            "all" will search all cadences. Pass a boolean array with values of "True"
            for times to search for SSOs.
        radius : optional, float
            Radius in degrees to search for bodies. If None, will search for
            SSOs within 15 pixels.
        sigma : optional, float
            If `cadence_mask` is set to `"outlier"`, `sigma` will be used to identify
            outliers.
        location : str
            Spacecraft location. Options include `'kepler'` and `'tess'`.
        cache : optional, bool
            If True will cache the search result in the astropy cache. Set to False
            to request the search again.
        return_mask: bool
            If True will return a boolean mask in time alongside the result

        Returns
        -------
        result : `pandas.DataFrame`
            DataFrame object which lists the Solar System objects in frames
            that were identified to contain SSOs.  Returns `None` if no objects
            were found.
        """
        for attr in ['ra', 'dec']:
            if not hasattr(self, '{}'.format(attr)):
                raise ValueError('Input does not have a `{}` attribute.'.format(attr))

        # Validate `cadence_mask`
        if isinstance(cadence_mask, str):
            if cadence_mask == 'outliers':
                cadence_mask = self.remove_outliers(sigma=sigma, return_mask=True)[1]
            elif cadence_mask == 'all':
                cadence_mask = np.ones(len(self.time)).astype(bool)
        elif not isinstance(cadence_mask, np.ndarray):
            raise ValueError('the `cadence_mask` argument is missing or invalid')
        # Avoid searching times with NaN flux; this is necessary because e.g.
        # `remove_outliers` includes NaNs in its mask.
        cadence_mask &= ~np.isnan(self.flux)

        # Validate `location`
        if location is None:
            if hasattr(self, 'mission') and self.mission:
                location = self.mission.lower()
            else:
                raise ValueError('you must pass a value for `location`.')

        # Validate `radius`
        if radius is None:
            # 15 pixels has been chosen as a reasonable default.
            # Comets have long tails which have tripped up users.
            if (location == 'kepler') | (location == 'k2'):
                radius = (4*15)*u.arcsecond.to(u.deg)
            elif location == 'tess':
                radius = (27*15)*u.arcsecond.to(u.deg)
            else:
                radius = 15*u.arcsecond.to(u.deg)

        res = _query_solar_system_objects(ra=self.ra, dec=self.dec,
                                          times=self.astropy_time.jd[cadence_mask],
                                          location=location, radius=radius, cache=cache)
        if return_mask:
            return res, np.in1d(self.astropy_time.jd, res.epoch)
        return res

    def _create_plot(self, column='flux', method='plot', ax=None, normalize=False,
                     xlabel=None, ylabel=None, title='', style='lightkurve',
                     show_colorbar=True, colorbar_label='',
                     **kwargs) -> matplotlib.axes.Axes:
        """Implements `plot()`, `scatter()`, and `errorbar()` to avoid code duplication.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        # Configure the default style
        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        # Default xlabel
        if xlabel is None:
            if not hasattr(self.time, 'format'):
                xlabel = "Phase"
            elif self.time.format == 'bkjd':
                xlabel = 'Time - 2454833 [BKJD days]'
            elif self.time.format == 'btjd':
                xlabel = 'Time - 2457000 [BTJD days]'
            elif self.time.format == 'jd':
                xlabel = 'Time [JD]'
            else:
                xlabel = 'Time'

        # Default ylabel
        if ylabel is None:
            if "flux" in column:
                ylabel = "Flux"
            else:
                ylabel = f"{column}"
            if normalize or self.meta.get("normalized"):
                ylabel = "Normalized " + ylabel
            elif (self[column].unit) and (self[column].unit.to_string() != ''):
                ylabel += f" [{self[column].unit.to_string('latex_inline')}]"

        # Default legend label
        if ('label' not in kwargs):
            kwargs['label'] = self.meta.get('label')

        flux = self[column]
        try:
            flux_err = self[f'{column}_err']
        except KeyError:
            flux_err = np.full(len(flux), np.nan)

        # Normalize the data if requested
        if normalize:
            if column == "flux":
                lc_normed = self.normalize()
            else:
                lc_tmp = self.copy()
                lc_tmp['flux'] = flux
                lc_tmp['flux_err'] = flux_err
                lc_normed = lc_tmp.normalize()
            flux, flux_err = lc_normed.flux, lc_normed.flux_err

        # Make the plot
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(1)
            if method == 'scatter':
                sc = ax.scatter(self.time.value, flux, **kwargs)
                # Colorbars should only be plotted if the user specifies, and there is
                # a color specified that is not a string (e.g. 'C1') and is iterable.
                if show_colorbar and ('c' in kwargs) and \
                   (not isinstance(kwargs['c'], str)) and hasattr(kwargs['c'], '__iter__'):
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label(colorbar_label)
                    cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
                    cbar.ax.minorticks_off()
            elif method == 'errorbar':
                if np.any(~np.isnan(flux_err)):
                    ax.errorbar(x=self.time.value, y=flux.value, yerr=flux_err.value, **kwargs)
                else:
                    log.warning(f"Column `{column}` has no associated errors.")
            else:
                ax.plot(self.time.value, flux.value, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if (np.sum([len(a) for a in legend_labels]) != 0):
                ax.legend(loc='best')

        return ax

    def plot(self, **kwargs) -> matplotlib.axes.Axes:
        """Plot the light curve using Matplotlib's `~matplotlib.pyplot.plot` method.

        Parameters
        ----------
        column : str
            Name of data column to plot. Default `flux`.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._create_plot(method='plot', **kwargs)

    def scatter(self, colorbar_label='', show_colorbar=True, **kwargs) -> matplotlib.axes.Axes:
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
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        colorbar_label : str
            Label to show next to the colorbar (if `c` is given).
        show_colorbar : boolean
            Show the colorbar if colors are given using the `c` argument?
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.scatter`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._create_plot(method='scatter', colorbar_label=colorbar_label,
                                 show_colorbar=show_colorbar, **kwargs)

    def errorbar(self, linestyle='', **kwargs) -> matplotlib.axes.Axes:
        """Plots the light curve using Matplotlib's `~matplotlib.pyplot.errorbar` method.

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
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        linestyle : str
            Connect the error bars using a line?
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.scatter`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if 'ls' not in kwargs:
            kwargs['linestyle'] = linestyle
        return self._create_plot(method='errorbar', **kwargs)

    def interact_bls(self, notebook_url='localhost:8888', minimum_period=None,
                     maximum_period=None, resolution=2000):
        """Display an interactive Jupyter Notebook widget to find planets.

        The Box Least Squares (BLS) periodogram is a statistical tool used
        for detecting transiting exoplanets and eclipsing binaries in
        light curves.  This method will display a Jupyter Notebook Widget
        which enables the BLS algorithm to be used interactively.
        Behind the scenes, the widget uses the AstroPy implementation of BLS [1]_.

        This feature only works inside an active Jupyter Notebook.
        It requires Bokeh v1.0 (or later) and AstroPy v3.1 (or later),
        which are optional dependencies. An error message will be shown
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
        minimum_period : float or None
            Minimum period to assess the BLS to. If None, default value of 0.3 days
            will be used.
        maximum_period : float or None
            Maximum period to evaluate the BLS to. If None, the time coverage of the
            lightcurve / 4 will be used.
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
        .. [1] http://docs.astropy.org/en/latest/stats/bls.html
        """
        from .interact_bls import show_interact_widget
        clean = self.remove_nans()
        return show_interact_widget(clean, notebook_url=notebook_url, minimum_period=minimum_period,
                                    maximum_period=maximum_period, resolution=resolution)

    def to_table(self) -> Table:
        return Table(self)

    @deprecated("2.0",
                message='`to_timeseries()` has been deprecated. `LightCurve` is a '
                        'sub-class of Astropy TimeSeries as of Lightkurve v2.0 '
                        'and no longer needs to be converted.',
                warning_type=LightkurveDeprecationWarning)
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
        return LightCurve(time=ts['time'].value, flux=ts['flux'], flux_err=ts['flux_err'])

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
            raise ImportError("You need to install Stingray to use "
                              "the LightCurve.to_stringray() method.")
        return StingrayLightcurve(time=self.time.value, counts=self.flux,
                                  err=self.flux_err, input_counts=False)

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
            Dictionary of arguments to be passed to `TimeSeries.write()`.

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

    def to_periodogram(self, method="lombscargle", **kwargs):
        """Converts the light curve to a `~lightkurve.periodogram.Periodogram`
        power spectrum object.

        This method will call either
        `lightkurve.periodogram.LombScarglePeriodogram.from_lightcurve()` or
        `lightkurve.periodogram.BoxLeastSquaresPeriodogram.from_lightcurve()`,
        which in turn wrap `astropy.stats.LombScargle` and `astropy.stats.BoxLeastSquares`.

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
            `~lightkurve.periodogram.LombScarglePeriodogram` or
            `~lightkurve.periodogram.BoxLeastSquaresPeriodogram`.

        Returns
        -------
        Periodogram : `~lightkurve.periodogram.Periodogram` object
            The power spectrum object extracted from the light curve.
        """
        supported_methods = ["ls", "bls", "lombscargle", "boxleastsquares"]
        method = validate_method(method.replace(' ', ''), supported_methods)
        if method in ["bls", "boxleastsquares"]:
            from . import BoxLeastSquaresPeriodogram
            return BoxLeastSquaresPeriodogram.from_lightcurve(lc=self, **kwargs)
        else:
            from . import LombScarglePeriodogram
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

    def to_fits(self, path=None, overwrite=False, flux_column_name='FLUX',
                **extra_data):
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
        typedir = {int: 'J', str: 'A', float: 'D', bool: 'L',
                   np.int32: 'J', np.int32: 'K', np.float32: 'E', np.float64: 'D'}

        def _header_template(extension):
            """Returns a template `fits.Header` object for a given extension."""
            template_fn = os.path.join(PACKAGEDIR, "data",
                                       "lc-ext{}-header.txt".format(extension))
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
            default = {'ORIGIN': "Unofficial data product",
                         'DATE': datetime.datetime.now().strftime("%Y-%m-%d"),
                         'CREATOR': "lightkurve.LightCurve.to_fits()",
                         'PROCVER': str(__version__)}

            for kw in default:
                hdu.header['{}'.format(kw).upper()] = default[kw]
                if default[kw] is None:
                    log.warning('Value for {} is None.'.format(kw))

            for kw in extra_data:
                if isinstance(extra_data[kw], (str, float, int, bool, type(None))):
                    hdu.header['{}'.format(kw).upper()] = extra_data[kw]
                    if extra_data[kw] is None:
                        log.warning('Value for {} is None.'.format(kw))
            return hdu

        def _make_lightcurve_extension(extra_data=None):
            """Create the 'LIGHTCURVE' extension (i.e. extension #1)."""
            # Turn the data arrays into fits columns and initialize the HDU
            if extra_data is None:
                extra_data = {}
            cols = []
            if ~np.asarray(['TIME' in k.upper() for k in extra_data.keys()]).any():
                cols.append(fits.Column(name='TIME', format='D', unit=self.time.format,
                                        array=self.time.value))
            if ~np.asarray([flux_column_name in k.upper() for k in extra_data.keys()]).any():
                cols.append(fits.Column(name=flux_column_name, format='E',
                                        unit='e-/s', array=self.flux))
            if 'flux_err' in dir(self):
                if ~(flux_column_name.upper() + '_ERR' in extra_data.keys()):
                    cols.append(fits.Column(name=flux_column_name.upper() + '_ERR', format='E',
                                            unit='e-/s', array=self.flux_err))
            if 'cadenceno' in dir(self):
                if ~np.asarray(['CADENCENO' in k.upper() for k in extra_data.keys()]).any():
                    cols.append(fits.Column(name='CADENCENO', format='J',
                                            array=self.cadenceno))
            for kw in extra_data:
                if isinstance(extra_data[kw], (np.ndarray, list)):
                    cols.append(fits.Column(name='{}'.format(kw).upper(),
                                            format=typedir[extra_data[kw].dtype.type],
                                            array=extra_data[kw]))
            if 'SAP_QUALITY' not in extra_data:
                cols.append(fits.Column(name='SAP_QUALITY',
                                        format='J',
                                        array=np.zeros(len(self.flux))))

            coldefs = fits.ColDefs(cols)
            hdu = fits.BinTableHDU.from_columns(coldefs)
            hdu.header['EXTNAME'] = 'LIGHTCURVE'
            return hdu

        def _hdulist(**extra_data):
            """Returns an astropy.io.fits.HDUList object."""
            list_out = fits.HDUList([_make_primary_hdu(extra_data=extra_data),
                             _make_lightcurve_extension(extra_data=extra_data)])
            return list_out

        hdu = _hdulist(**extra_data)
        if path is not None:
            hdu.writeto(path, overwrite=overwrite, checksum=True)
        return hdu

    def to_corrector(self, method="sff"):
        """Returns a corrector object to remove instrument systematics.

        Parameters
        ----------
        methods : string
            Currently, only "sff" is supported.  This will return a
            `SFFCorrector` class instance.

        Returns
        -------
        correcter : `lightkurve.Correcter`
            Instance of a Corrector class, which typically provides `correct()`
            and `diagnose()` methods.
        """
        allowed_methods = ["sff"]
        if method == "pld":
            raise ValueError("The 'pld' method can only be used on "
                             "`TargetPixelFile` objects, not `LightCurve` objects.")
        if method not in allowed_methods:
            raise ValueError(("Unrecognized method '{0}'\n"
                              "allowed methods are: {1}")
                             .format(method, allowed_methods))
        if method == "sff":
            from .correctors import SFFCorrector
            return SFFCorrector(self)

    @deprecated_renamed_argument('t0', 'epoch_time', '2.0',
                                 warning_type=LightkurveDeprecationWarning)
    def plot_river(self, period, epoch_time=None, ax=None, bin_points=1,
                   minimum_phase=-0.5, maximum_phase=0.5, method='mean',
                   **kwargs) -> matplotlib.axes.Axes:
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
        if hasattr(self, 'time_original'):  # folded light curve
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

        method = validate_method(method, supported_methods=['mean', 'median', 'sigma'])
        if (bin_points == 1) and (method in ['mean', 'median']):
            bin_func = lambda y, e: (y[0], e[0])
        elif (bin_points == 1) and (method in ['sigma']):
            bin_func = lambda y, e: ((y[0] - 1)/e[0], np.nan)
        elif method == 'mean':
            bin_func = lambda y, e: (np.nanmean(y), np.nansum(e**2)**0.5/len(e))
        elif method == 'median':
            bin_func = lambda y, e: (np.nanmedian(y), np.nansum(e**2)**0.5/len(e))
        elif method == 'sigma':
            bin_func = lambda y, e: ((np.nanmean(y) - 1)/(np.nansum(e**2)**0.5/len(e)), np.nan)

        s = np.argsort(time.value)
        x, y, e = time.value[s], self.flux[s], self.flux_err[s]
        med = np.nanmedian(self.flux)
        e /= med
        y /= med

        # Here `ph` is the phase of each time point x
        # cyc is the number of cycles that have occured at each time point x
        # since the phase 0 before x[0]
        n = int(period.value/np.nanmedian(np.diff(x)) * (maximum_phase - minimum_phase)/bin_points)
        if n == 1:
            bin_points = int(maximum_phase - minimum_phase)/(2 / int(period.value/np.nanmedian(np.diff(x))))
            warnings.warn('`bin_points` is too high to plot a phase curve, resetting to {}'.format(bin_points),
                          LightkurveWarning)
            n = 2
        ph = x/period.value % 1
        cyc = np.asarray((x - x % period.value)/period.value, int)
        cyc -= np.min(cyc)

        phase = (epoch_time.value % period.value) / period.value
        ph = ((x - (phase * period.value)) / period.value) % 1
        cyc = np.asarray((x - ((x - phase * period.value) % period.value))/period.value, int)
        cyc -= np.min(cyc)
        ph[ph > 0.5] -= 1

        ar = np.empty((n, np.max(cyc) + 1))
        ar[:] = np.nan
        bs = np.linspace(minimum_phase, maximum_phase, n)
        cycs = np.arange(0, np.max(cyc) + 1)

        ph_masks = [(ph > bs[jdx]) & (ph <= bs[jdx+1]) for jdx in range(n-1)]
        qual_mask = np.isfinite(y)
        for cyc1 in np.unique(cyc):
            cyc_mask = cyc == cyc1
            if not np.any(cyc_mask):
                continue
            for jdx, ph_mask in enumerate(ph_masks):
                if not np.any(cyc_mask & ph_mask & qual_mask):
                    ar[jdx, cyc1] = np.nan
                else:
                    ar[jdx, cyc1] = bin_func(y[cyc_mask & ph_mask],
                                             e[cyc_mask & ph_mask])[0]

        # If the method is average we need to denormalize the plot
        if method in ['mean', 'median']:
            ar *= np.nanmedian(self.flux.value)

        d = np.max([np.abs(np.nanmedian(ar) - np.nanpercentile(ar, 5)),
                    np.abs(np.nanmedian(ar) - np.nanpercentile(ar, 95))])
        vmin = kwargs.pop('vmin', np.nanmedian(ar) - d)
        vmax = kwargs.pop('vmax', np.nanmedian(ar) + d)
        if method in ['mean', 'median']:
            cmap = kwargs.pop('cmap', 'viridis')
        elif method == 'sigma':
            cmap = kwargs.pop('cmap', 'coolwarm')

        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots(figsize=(12, cyc.max()*0.1))

            im = ax.pcolormesh(bs, cycs, ar.T, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            cbar = plt.colorbar(im, ax=ax)
            if method in ['mean', 'median']:
                unit = '[Normalized Flux]'
                if self.flux.unit is not None:
                    if (self.flux.unit != u.dimensionless_unscaled):
                        unit = '[{}]'.format(self.flux.unit.to_string('latex'))
                if bin_points == 1:
                    cbar.set_label("Flux {}".format(unit))
                else:
                    cbar.set_label("Average Flux in Bin {}".format(unit))
            elif method == 'sigma':
                if bin_points == 1:
                    cbar.set_label("Flux in units of Standard Deviation "
                                   "$(f - \overline{f})/(\sigma_f)$")
                else:
                    cbar.set_label("Average Flux in Bin in units of Standard Deviation "
                                   "$(f - \overline{f})/(\sigma_f)$")

            ax.set_xlabel("Phase")
            ax.set_ylabel("Cycle")
            ax.set_ylim(cyc.max(), 0)
            ax.set_title(self.meta.get("label"))
            a = cyc.max() * 0.1 / 12.
            b = (cyc.max() - cyc.min()) / (bs.max() - bs.min())
            ax.set_aspect(a/b)
        return ax


class FoldedLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    in which the ``time`` parameter represents phase values.

    Compared to the `~lightkurve.lightcurve.LightCurve` base class, this class
    has extra meta data entries (``period``, ``epoch_time``, ``epoch_phase``,
    ``wrap_phase``, ``normalize_phase``), an extra column (``time_original``),
    extra properties (``phase``, ``odd_mask``, ``even_mask``),
    and implements different plotting defaults.
    """

    @property
    def phase(self):
        return self.time

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
        cycle = (self.time_original - self.time.value * (self.period) - self.period * 0.5) / (self.period * 2)
        return (cycle.value % 1) < 0.5

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
        if 'xlabel' not in kwargs:
            kwargs['xlabel'] = "Phase"
            if isinstance(self.time, TimeDelta):
                kwargs['xlabel'] += f" [{self.time.format.upper()}]"
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
        ax = super(FoldedLightCurve, self).plot_river(period=self.period, epoch_time=self.epoch_time, **kwargs)
        return ax


class KeplerLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    to represent data from NASA's Kepler and K2 mission."""

    _deprecated_keywords = ('targetid', 'label',  'time_format', 'time_scale',
                            'flux_unit', 'quality_bitmask', 'channel',
                            'campaign', 'quarter', 'mission', 'ra', 'dec')

    _default_time_format = 'bkjd'

    @classmethod
    def read(cls, *args, **kwargs):
        # Default to Kepler file format
        if kwargs.get("format") is None:
            kwargs['format'] = "kepler"
        return super().read(*args, **kwargs)

    def to_fits(self, path=None, overwrite=False, flux_column_name='FLUX',
                aperture_mask=None,**extra_data):
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
            'TELESCOP': "KEPLER",
            'INSTRUME': "Kepler Photometer",
            'OBJECT': '{}'.format(self.targetid),
            'KEPLERID': self.targetid,
            'CHANNEL': self.channel,
            'MISSION': self.mission,
            'RA_OBJ': self.ra,
            'DEC_OBJ': self.dec,
            'EQUINOX': 2000,
            'DATE-OBS': Time(self.time[0]+2454833., format=('jd')).isot,
            'SAP_QUALITY': self.quality,
            'MOM_CENTR1': self.centroid_col,
            'MOM_CENTR2': self.centroid_row}

        for kw in kepler_specific_data:
            if ~np.asarray([kw.lower == k.lower() for k in extra_data]).any():
                extra_data[kw] = kepler_specific_data[kw]
        hdu = super(KeplerLightCurve, self).to_fits(path=None,
                                                    overwrite=overwrite,
                                                    **extra_data)

        hdu[0].header['QUARTER'] = self.meta.get('quarter')
        hdu[0].header['CAMPAIGN'] = self.meta.get('campaign')

        hdu = _make_aperture_extension(hdu, aperture_mask)

        if path is not None:
            hdu.writeto(path, overwrite=overwrite, checksum=True)
        else:
            return hdu


class TessLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    to represent data from NASA's TESS mission."""

    _deprecated_keywords = ('targetid', 'label',  'time_format', 'time_scale',
                            'flux_unit', 'quality_bitmask', 'sector',
                            'camera', 'ccd', 'mission', 'ra', 'dec')

    _default_time_format = 'btjd'

    @classmethod
    def read(cls, *args, **kwargs):
        # Default to TESS file format
        if kwargs.get("format") is None:
            kwargs['format'] = "tess"
        return super().read(*args, **kwargs)

    def to_fits(self, path=None, overwrite=False, flux_column_name='FLUX',
                aperture_mask=None, **extra_data):
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
            'OBJECT': '{}'.format(self.targetid),
            'MISSION': self.meta.get('mission'),
            'RA_OBJ': self.meta.get('ra'),
            'TELESCOP': self.meta.get('mission'),
            'CAMERA': self.meta.get('camera'),
            'CCD': self.meta.get('ccd'),
            'SECTOR': self.meta.get('sector'),
            'TARGETID': self.meta.get('targetid'),
            'DEC_OBJ': self.meta.get('dec'),
            'MOM_CENTR1': self.centroid_col,
            'MOM_CENTR2': self.centroid_row}

        for kw in tess_specific_data:
            if ~np.asarray([kw.lower == k.lower() for k in extra_data]).any():
                extra_data[kw] = tess_specific_data[kw]
        hdu = super(TessLightCurve, self).to_fits(path=None,
                                                    overwrite=overwrite,
                                                    **extra_data)

        # We do this because the TESS file format is subtly different in the
        #    name of this column.
        hdu[1].columns.change_name('SAP_QUALITY', 'QUALITY')

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

    contains_bit2 = (clean_mask.astype(np.int) & 2).any()
    all_zeros_or_ones = ( (clean_mask.dtype in ['float', 'int']) &
                            ((set(np.unique(clean_mask)) - {0,1}) == set()) )
    is_bool_mask = ( (aperture_mask.dtype == 'bool') | all_zeros_or_ones )

    if is_bool_mask:
        out_mask = np.ones(aperture_mask.shape, dtype=np.uint8)
        out_mask[aperture_mask == 1] = 3
        out_mask = out_mask.astype(np.uint8)
    elif contains_bit2:
        out_mask = aperture_mask.astype(np.uint8)
    else:
        log.warn("The input aperture mask must be boolean or follow the \
                Kepler-pipeline standard; returning None.")
        out_mask = None
    return out_mask

def _make_aperture_extension(hdu_list, aperture_mask):
    """Returns an `ImageHDU` object containing the 'APERTURE' extension
    of a light curve file."""
    if aperture_mask is not None:
        bitmask = _boolean_mask_to_bitmask(aperture_mask)
        hdu = fits.ImageHDU(bitmask)
        hdu.header['EXTNAME'] = 'APERTURE'
        hdu_list.append(hdu)
    return hdu_list
