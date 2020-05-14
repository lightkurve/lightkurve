"""Defines LightCurve, KeplerLightCurve, and TessLightCurve."""

from __future__ import division, print_function

import copy
import os
import datetime
import logging
import warnings

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astropy import units as u

from . import PACKAGEDIR, MPLSTYLE
from .utils import (running_mean, bkjd_to_astropy_time, btjd_to_astropy_time,
    LightkurveWarning, validate_method, _query_solar_system_objects
)


__all__ = ['LightCurve', 'KeplerLightCurve', 'TessLightCurve', 'FoldedLightCurve']

log = logging.getLogger(__name__)


class LightCurve(object):
    """Generic light curve object to hold time series photometry for one target.

    Attributes
    ----------
    time : array-like
        Time values.
    flux : array-like
        Flux values for every time point.
    flux_err : array-like
        Uncertainty on each flux data point.
    flux_unit : `~astropy.units.Unit` or str
        Unit of the flux values.  If a string is passed, it will be passed
        on to `~astropy.units.Unit`.
    time_format : str
        String specifying how an instant of time is represented,
        e.g. 'bkjd' or 'jd'.
    time_scale : str
        String which specifies how the time is measured,
        e.g. 'tdb', 'tt', 'ut1', or 'utc'.
    targetid : str
        Identifier of the target.
    label : str
        Human-friendly object label, e.g. "KIC 123456789".
    meta : dict
        Free-form metadata associated with the LightCurve.

    Examples
    --------
    >>> import lightkurve as lk
    >>> lc = lk.LightCurve(time=[1, 2, 3, 4], flux=[0.97, 1.01, 1.03, 0.99])
    >>> lc.time
    array([1, 2, 3, 4])
    >>> lc.flux
    array([0.97, 1.01, 1.03, 0.99])
    >>> lc.bin(binsize=2).flux
    array([0.99, 1.01])
    """
    extra_columns = ()

    def __init__(self, time=None, flux=None, flux_err=None, flux_unit=None,
                 time_format=None, time_scale=None, targetid=None, label=None,
                 meta=None):
        if time is None and flux is None:
            raise ValueError('either time or flux must be given')
        if time is None:
            self.time = np.arange(len(flux))
        else:
            self.time = self._validate_time(time)
        self.flux = self._validate_array(flux, name='flux')
        self.flux_err = self._validate_array(flux_err, name='flux_err')
        # If `time` or `flux` are astropy objects, we will retrieve
        # `time_format`, `time_scale,` and `flux_unit` from them.
        if isinstance(flux, u.Quantity):
            flux_unit = flux.unit
        if isinstance(time, Time):
            time_format = time.format
            time_scale = time.scale
        self.flux_unit = flux_unit  # @flux_unit.setter will validate this
        self.time_format = time_format
        self.time_scale = time_scale
        self.targetid = targetid
        self.label = label
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

    @classmethod
    def _validate_time(cls, time):
        """Ensure the `time` user input is valid."""
        if isinstance(time, Time):  # Support Astropy Time objects
            time = time.value
        time = np.asarray(time)
        # Trigger warning if time=NaN are present
        if np.isnan(time).any():
            warnings.warn('LightCurve object contains NaN times', LightkurveWarning)
        return time

    def _validate_array(self, arr, name='array'):
        """Ensure the input flux/centroid/quality/etc arrays are valid and have
        the exact same length as `self.time`."""
        if arr is None:  # arrays default to NaN arrays of length time
            arr = np.nan * np.ones_like(self.time)
        else:
            arr = np.asarray(arr)

        if not (len(self.time) == len(arr)):
            raise ValueError("Input arrays have different lengths."
                             " len(time)={}, len({})={}"
                             .format(len(self.time), name, len(arr)))
        return arr

    def __getitem__(self, key):
        copy_self = self.copy()
        copy_self.time = self.time[key]
        copy_self.flux = self.flux[key]
        copy_self.flux_err = self.flux_err[key]
        for k in self.extra_columns:
            setattr(copy_self, k, getattr(self, k)[key])
        return copy_self

    def __len__(self):
        return len(self.time)

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

    @property
    def flux_unit(self):
        return self._flux_unit

    @flux_unit.setter
    def flux_unit(self, flux_unit):
        # Validate user input for `flux_unit`
        if flux_unit is None:
            self._flux_unit = None
        else:
            try:
                self._flux_unit = u.Unit(flux_unit)
            except ValueError as e:
                raise ValueError("invalid `flux_unit`: {}".format(e))

    @property
    def flux_quantity(self):
        """Returns the flux as an Astropy `~astropy.units.Quantity` object."""
        if isinstance(self.flux_unit, u.UnitBase):
            return self.flux * self.flux_unit
        else:
            return self.flux * u.dimensionless_unscaled

    @property
    def astropy_time(self):
        """Returns the time values as an Astropy `~astropy.time.Time` object.

        The Time object will be created based on the values of the light curve's
        `time`, `time_format`, and `time_scale` attributes.

        Examples
        --------
        The section below demonstrates working with time values using the TESS
        light curve of Pi Mensae as an example, which we obtained as follows::

            >>> import lightkurve as lk
            >>> lc = lk.search_lightcurvefile("Pi Mensae", mission="TESS", sector=1).download().PDCSAP_FLUX  # doctest: +SKIP
            >>> lc  # doctest: +SKIP
            TessLightCurve(TICID: 261136679)

        Every `LightCurve` object has a `time` attribute, which provides access
        to the original array of time values given in the native format and
        scale used by the data product from which the light curve was obtained::

            >>> lc.time  # doctest: +SKIP
            array([1325.29698328, 1325.29837215, 1325.29976102, ..., 1353.17431099,
                   1353.17569985, 1353.17708871])
            >>> lc.time_format  # doctest: +SKIP
            'btjd'
            >>> lc.time_scale  # doctest: +SKIP
            'tdb'

        To enable users to convert these time values to different formats or
        scales, Lightkurve provides an easy way to access the time values
        as an `AstroPy Time object <http://docs.astropy.org/en/stable/time/>`_::

            >>> lc.astropy_time  # doctest: +SKIP
            <Time object: scale='tdb' format='jd' value=[2458325.29698328 2458325.29837215 2458325.29976102 ... 2458353.17431099
            2458353.17569985 2458353.17708871]>

        This is convenient because AstroPy Time objects provide a lot of useful
        features. For example, we can now obtain the Julian Day or ISO values
        that correspond to the raw time values::

            >>> lc.astropy_time.iso  # doctest: +SKIP
            array(['2018-07-25 19:07:39.356', '2018-07-25 19:09:39.354',
                   '2018-07-25 19:11:39.352', ..., '2018-08-22 16:11:00.470',
                   '2018-08-22 16:13:00.467', '2018-08-22 16:15:00.464'], dtype='<U23')
            >>> lc.astropy_time.jd   # doctest: +SKIP
            array([2458325.29698328, 2458325.29837215, 2458325.29976102, ...,
                   2458353.17431099, 2458353.17569985, 2458353.17708871])


        Raises
        ------
        ValueError
            If the ``time_format`` attribute is not set or not one of the formats
            allowed by AstroPy.
        """
        if self.time_format is None:
            raise ValueError("To retrieve a `Time` object the `time_format` "
                             "attribute must be set on the LightCurve object, "
                             "e.g. `lightcurve.time_format = 'jd'`.")
        # AstroPy does not support BKJD, so we call a function to convert to JD.
        # In the future, we should think about making an AstroPy-compatible
        # `TimeFormat` class for BKJD.
        if self.time_format == 'bkjd':
            return bkjd_to_astropy_time(self.time)
        elif self.time_format == 'btjd':  # TESS
            return btjd_to_astropy_time(self.time)
        return Time(self.time, format=self.time_format, scale=self.time_scale)

    def show_properties(self):
        """Prints a description of all non-callable attributes.

        Prints in order of type (ints, strings, lists, arrays, others).
        """
        attrs = {}
        for attr in dir(self):
            if not attr.startswith('_'):
                try:
                    res = getattr(self, attr)
                except Exception:
                    continue
                if callable(res):
                    continue
                if attr == 'hdu':
                    attrs[attr] = {'res': res, 'type': 'list'}
                    for idx, r in enumerate(res):
                        if idx == 0:
                            attrs[attr]['print'] = '{}'.format(r.header['EXTNAME'])
                        else:
                            attrs[attr]['print'] = '{}, {}'.format(
                                attrs[attr]['print'], '{}'.format(r.header['EXTNAME']))
                    continue
                else:
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
        if not hasattr(others, '__iter__'):
            others = [others]
        if inplace:
            new_lc = self
        else:
            new_lc = self.copy()

        # Find the intersection of all the extra_columns values
        extra_columns = set(new_lc.extra_columns)
        flag = True
        for other in others:
            next_columns = set(other.extra_columns)
            extra_columns &= next_columns
            flag &= extra_columns == next_columns
        flag &= set(new_lc.extra_columns) == extra_columns
        if not flag:
            warnings.warn(
                "append is being applied to LightCurve objects with "
                "inconsistent values for `extra_columns`",
                LightkurveWarning
            )

        for i in range(len(others)):
            new_lc.time = np.append(new_lc.time, others[i].time)
            new_lc.flux = np.append(new_lc.flux, others[i].flux)
            new_lc.flux_err = np.append(new_lc.flux_err, others[i].flux_err)

            for column in extra_columns:
                setattr(
                    new_lc,
                    column,
                    np.append(
                        getattr(new_lc, column),
                        getattr(others[i], column)
                    )
                )

        return new_lc

    def copy(self):
        """Returns a copy of this `LightCurve` object.

        This method uses Python's `copy.deepcopy` function to ensure that all
        objects stored within the LightCurve instance are fully copied.

        Returns
        -------
        lc_copy : `LightCurve`
            A new light curve object which is a copy of the original.
        """
        return copy.deepcopy(self)

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
            dt = self.time[mask][1:] - self.time[mask][0:-1]
            with warnings.catch_warnings():  # Ignore warnings due to NaNs
                warnings.simplefilter("ignore", RuntimeWarning)
                cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
            low = np.append([0], cut)
            high = np.append(cut, len(self.time[mask]))
            # Then, apply the savgol_filter to each segment separately
            trend_signal = np.zeros(len(self.time[mask]))
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
                        trend_signal[l:h] = signal.savgol_filter(x=self.flux[mask][l:h],
                                                                 window_length=window_length,
                                                                 polyorder=polyorder,
                                                                 **kwargs)
            # Ignore outliers; note we add `1e-14` below to avoid detecting
            # outliers which are merely caused by numerical noise.
            mask1 = np.nan_to_num(np.abs(self.flux[mask] - trend_signal)) <\
                    (np.nanstd(self.flux[mask] - trend_signal) * sigma + 1e-14)
            f = interp1d(self.time[mask][mask1], trend_signal[mask1], fill_value='extrapolate')
            trend_signal = f(self.time)
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

    def fold(self, period, t0=None, transit_midpoint=None):
        """Folds the lightcurve at a specified `period` and reference time `t0`.

        This method returns a `~lightkurve.lightcurve.FoldedLightCurve` object
        in which the time values range between -0.5 to +0.5 (i.e. the phase).
        Data points which occur exactly at ``t0`` or an integer multiple of
        ``t0 + n*period`` will have phase value 0.0.

        Examples
        --------
        The example below shows a light curve with a period dip which occurs near
        time value 1001 and has a period of 5 days. Calling the `fold` method
        will transform the light curve into a
        `~lightkurve.lightcurve.FoldedLightCurve` object::

            >>> import lightkurve as lk
            >>> lc = lk.LightCurve(time=range(1001, 1012), flux=[0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5])
            >>> folded_lc = lc.fold(period=5., t0=1006.)
            >>> folded_lc   # doctest: +SKIP
            <lightkurve.lightcurve.FoldedLightCurve>

        An object of type `~lightkurve.lightcurve.FoldedLightCurve` is useful
        because it provides convenient access to the phase values and the
        phase-folded fluxes::

            >>> folded_lc.phase
            array([-0.4, -0.4, -0.2, -0.2,  0. ,  0. ,  0. ,  0.2,  0.2,  0.4,  0.4])
            >>> folded_lc.flux
            array([1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. ])

        We can still access the original time values as well::

            >>> folded_lc.time_original
            array([1004, 1009, 1005, 1010, 1001, 1006, 1011, 1002, 1007, 1003, 1008])

        A `~lightkurve.lightcurve.FoldedLightCurve` inherits all the features
        of a standard `LightCurve` object. For example, we can very quickly
        obtain a phase-folded plot using:

            >>> folded_lc.plot()    # doctest: +SKIP


        Parameters
        ----------
        period : float
            The period upon which to fold, in the same units as this
            LightCurve's ``time`` attribute.
        t0 : float, optional
            Time corresponding to zero phase, in the same units as this
            LightCurve's ``time`` attribute.  Defaults to 0 if not set.
        transit_midpoint : float, optional
            Deprecated.  Use `t0` instead.

        Returns
        -------
        folded_lightcurve : `~lightkurve.lightcurve.FoldedLightCurve`
            A new light curve object in which the data are folded and sorted by
            phase. The object contains an extra ``phase`` attribute.
        """
        # Input validation.  (Note: Quantities are simply ignored for now;
        # we should consider adding extra validation here.)
        if isinstance(period, u.quantity.Quantity):
            period = period.value
        if isinstance(t0, u.quantity.Quantity):
            t0 = t0.value

        # `transit_midpoint` is deprecated
        if transit_midpoint is not None:
            warnings.warn('`transit_midpoint` is deprecated, please use `t0` instead.',
                          LightkurveWarning)
            if t0 is None:
                t0 = transit_midpoint

        if t0 is None:
            t0 = 0.

        if (t0 > 2450000):
            if self.time_format == 'bkjd':
                warnings.warn('`t0` appears to be given in JD, '
                              'however the light curve time uses BKJD '
                              '(i.e. JD - 2454833).', LightkurveWarning)
            elif self.time_format == 'btjd':
                warnings.warn('`t0` appears to be given in JD, '
                              'however the light curve time uses BTJD '
                              '(i.e. JD - 2457000).', LightkurveWarning)
        phase = (t0 % period) / period
        fold_time = (((self.time - phase * period) / period) % 1)
        # fold time domain from -.5 to .5
        fold_time[fold_time > 0.5] -= 1
        sorted_args = np.argsort(fold_time)
        return FoldedLightCurve(time=fold_time[sorted_args],
                                flux=self.flux[sorted_args],
                                flux_err=self.flux_err[sorted_args],
                                time_original=self.time[sorted_args],
                                targetid=self.targetid,
                                period=period,
                                t0=t0,
                                label=self.label,
                                flux_unit=self.flux_unit,
                                meta=self.meta)

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
            array([1.00055917, 0.99885466, 1.        ])
            >>> normalized_lc.flux_err
            array([0.00026223, 0.00017739, 0.00023909])

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
        # Warn if the light curve is already in relative units.
        if isinstance(self._flux_unit, u.UnitBase) and \
            self._flux_unit.is_equivalent(u.dimensionless_unscaled):
            warnings.warn("The light curve already appears to be in relative "
                          "units; `normalize()` will convert the light curve "
                          "into relative units for a second time, which is "
                          "probably not what you want.".format(self._flux_unit),
                          LightkurveWarning)

        # Create a new light curve instance and normalize its values
        lc = self.copy()
        lc.flux = lc.flux / median_flux
        lc.flux_err = lc.flux_err / median_flux
        lc.flux_unit = u.dimensionless_unscaled

        # Set the desired relative (dimensionless) units
        if unit == 'unscaled':
            lc.flux_unit = u.dimensionless_unscaled
        elif unit == 'percent':
            lc.flux_unit = u.percent
            lc.flux *= 100
            lc.flux_err *= 100
        elif unit == 'ppt':  # parts per thousand
            # ppt is not included in astropy, so we define it here
            lc.flux_unit = u.def_unit(['ppt', 'parts per thousand'], u.Unit(1e-3))
            lc.flux *= 1000
            lc.flux_err *= 1000
        elif unit == 'ppm':  # parts per million
            lc.flux_unit = u.cds.ppm
            lc.flux *= 1000000
            lc.flux_err *= 1000000

        return lc

    def remove_nans(self):
        """Removes cadences where the flux is NaN.

        Returns
        -------
        clean_lightcurve : `LightCurve`
            A new light curve object from which NaNs fluxes have been removed.
        """
        return self[~np.isnan(self.flux)]  # This will return a sliced copy

    def fill_gaps(self, method='gaussian_noise'):
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
        nlc = lc.copy()

        # Find missing time points
        # Most precise method, taking into account time variation due to orbit
        if hasattr(lc, 'cadenceno'):
            dt = lc.time - np.median(np.diff(lc.time)) * lc.cadenceno
            ncad = np.arange(lc.cadenceno[0], lc.cadenceno[-1] + 1, 1)
            in_original = np.in1d(ncad, lc.cadenceno)
            ncad = ncad[~in_original]
            ndt = np.interp(ncad, lc.cadenceno, dt)

            ncad = np.append(ncad, lc.cadenceno)
            ndt = np.append(ndt, dt)
            ncad, ndt = ncad[np.argsort(ncad)], ndt[np.argsort(ncad)]
            ntime = ndt + np.median(np.diff(lc.time)) * ncad
            nlc.cadenceno = ncad
        else:
            # Less precise method
            dt = np.nanmedian(lc.time[1::] - lc.time[:-1:])
            ntime = [lc.time[0]]
            for t in lc.time[1::]:
                prevtime = ntime[-1]
                while (t - prevtime) > 1.2*dt:
                    ntime.append(prevtime + dt)
                    prevtime = ntime[-1]
                ntime.append(t)
            ntime = np.asarray(ntime, float)
            in_original = np.in1d(ntime, lc.time)
        # Fill in time points

        nlc.time = ntime
        f = np.zeros(len(ntime))
        f[in_original] = np.copy(lc.flux)
        fe = np.zeros(len(ntime))
        fe[in_original] = np.copy(lc.flux_err)

        fe[~in_original] = np.interp(ntime[~in_original], lc.time, lc.flux_err)
        if method == 'gaussian_noise':
            try:
                std = lc.estimate_cdpp()*1e-6
            except:
                std = lc.flux.std()
            f[~in_original] = np.random.normal(lc.flux.mean(), std, (~in_original).sum())
        else:
            raise NotImplementedError("No such method as {}".format(method))

        nlc.flux = f
        nlc.flux_err = fe

        if hasattr(lc, 'quality'):
            quality = np.zeros(len(ntime), dtype=lc.quality.dtype)
            quality[in_original] = np.copy(lc.quality)
            quality[~in_original] += 65536
            nlc.quality = quality
        for column in lc.extra_columns:
            if column == "quality":
                continue
            old_values = getattr(lc, column)
            new_values = np.empty(len(ntime), dtype=old_values.dtype)
            new_values[~in_original] = np.nan
            new_values[in_original] = np.copy(old_values)
            setattr(nlc, column, new_values)

        return nlc

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
            array([1, 3, 5])
            >>> lc_clean.flux
            array([1, 1, 1])

        Instead of specifying `sigma`, you may specify separate `sigma_lower`
        and `sigma_upper` parameters to remove only outliers above or below
        the median. For example::

            >>> lc = LightCurve(time=[1, 2, 3, 4, 5], flux=[1, 1000, 1, -1000, 1])
            >>> lc_clean = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper=1)
            >>> lc_clean.time
            array([1, 3, 4, 5])
            >>> lc_clean.flux
            array([    1,     1, -1000,     1])

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
            return self[~outlier_mask], outlier_mask
        return self[~outlier_mask]

    def bin(self, binsize=None, bins=None, method='mean'):
        """Bins a lightcurve in chunks defined by ``binsize`` or ``bins``.

        The flux value of the bins will be computed by taking the mean
        (``method='mean'``) or the median (``method='median'``) of the flux.
        The default is mean.

        Parameters
        ----------
        binsize : int or None
            Number of cadences to include in every bin.  The default
            is 13 if neither `bins` nor `binsize` is assigned.
        bins : int, list of int, str, or None
            Requires Astropy version >3.1 and >2.10
            Instruction for how to assign bin locations grouping by the time of
            samples rather than index; overrides the `binsize=` if given.
            If ``bins`` is an int, it is the number of bins. If it is a list
            it is taken to be the bin edges. If it is a string, it must be one
            of  'blocks', 'knuth', 'scott' or 'freedman'.
            See `~astropy.stats.histogram` for description of these algorithms.
        method: str, one of 'mean' or 'median'
            The summary statistic to return for each bin. Default: 'mean'.

        Returns
        -------
        binned_lc : `LightCurve`
            A new light curve which has been binned.

        Notes
        -----
        - If the ratio between the lightcurve length and the binsize is not
          a whole number, then the remainder of the data points will be
          ignored.
        - If the original light curve contains flux uncertainties (``flux_err``),
          the binned lightcurve will report the root-mean-square error.
          If no uncertainties are included, the binned curve will return the
          standard deviation of the data.
        - If the original lightcurve contains a quality attribute, then the
          bitwise OR of the quality flags will be returned per bin.
        """
        # Validate user input
        method = validate_method(method, supported_methods=['mean', 'median'])
        if (binsize is None) and (bins is None):
            binsize = 13
        elif (binsize is not None) and (bins is not None):
            raise ValueError('Both binsize and bins kwargs were passed to '
                             '`.bin()`.  Must assign only one of these.')

        # Only recent versions of AstroPy (>Dec 2018) provide ``calculate_bin_edges``
        if bins is not None:
            try:
                from astropy.stats import calculate_bin_edges
            except ImportError:
                from astropy import __version__ as astropy_version
                raise ImportError("The `bins=` parameter requires astropy >=3.1 or >=2.10, "
                                  "you currently have astropy version {}. "
                                  "Update astropy or use the `binsize` argument instead."
                                  "".format(astropy_version))

        # Define and map the functions to be applied to each bin
        method_func = np.__dict__['nan' + method]
        quality_func = lambda x: np.bitwise_or.reduce(x) \
                                 if np.issubdtype(x.dtype, np.integer) and np.all(np.isfinite(x)) \
                                 else np.nan
        centroid_func = lambda x: method_func(x) \
                                  if np.any(np.isfinite(x)) else np.nan
        # Assume the errors combine as the root-mean-square
        rmse_func = lambda x: np.sqrt(np.nansum(x**2))/len(x) \
                              if np.any(np.isfinite(x)) else np.nan
        statistic_mapper = {'flux': method_func,
                            'time': method_func,
                            'quality': quality_func,
                            'centroid_row': centroid_func,
                            'centroid_col': centroid_func,
                            'flux_err': rmse_func}
        statistic_mapper = {key: value
                            for key, value in statistic_mapper.items()
                            if hasattr(self, key)}
        for column in self.extra_columns:
            if column in statistic_mapper:
                continue
            statistic_mapper[column] = lambda x: np.nan

        # Now create the new binned light curve object
        binned_lc = self.copy()
        if bins is None:  # use ``binsize```
            n_bins = self.flux.size // binsize
            bin_by_array = np.arange(len(self.time))
            #bin_edges = calculate_bin_edges(bin_by_array, bins=n_bins)
            bin_edges = np.linspace(bin_by_array.min(), bin_by_array.max(),
                        n_bins + 1, endpoint=True)
        else:  # ``bins``` was assigned
            bin_by_array = self.time
            bin_edges = calculate_bin_edges(bin_by_array, bins=bins)
            n_bins = len(bin_edges) - 1
            # Trigger a warning if the bin edges make no sense
            if ((np.max(bin_edges) < np.nanmin(self.time)) or
                    (np.nanmax(self.time) < np.nanmin(bin_edges))):
                warnings.warn("the range of the bin edges ({}-{}) does not "
                              "fall in the light curve's time range ({}-{})"
                              "".format(np.min(bin_edges), np.max(bin_edges),
                                        np.nanmin(self.time), np.nanmax(self.time)),
                               LightkurveWarning)

        for attr, bin_function in statistic_mapper.items():
            values_to_bin = getattr(self, attr)
            # Override error propagation if flux_err is all NaN
            if (attr == 'flux_err') & ~np.any(np.isfinite(self.flux_err)):
                values_to_bin = self.flux
                bin_function = np.nanstd

            with warnings.catch_warnings():  # Ignore empty slice warnings
                warnings.simplefilter("ignore", RuntimeWarning)
                binned_stat = binned_statistic(bin_by_array, values_to_bin,
                    statistic=bin_function, bins=bin_edges).statistic
                setattr(binned_lc, attr, binned_stat)

        return binned_lc

    def estimate_cdpp(self, transit_duration=13, savgol_window=101,
                      savgol_polyorder=2, sigma=5.):
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
        mean = running_mean(data=cleaned_lc.flux, window_size=transit_duration)
        cdpp_ppm = np.std(mean) * 1e6
        return cdpp_ppm

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



    def _create_plot(self, method='plot', ax=None, normalize=False,
                     xlabel=None, ylabel=None, title='', style='lightkurve',
                     show_colorbar=True, colorbar_label='',
                     **kwargs):
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
            if self.time_format == 'bkjd':
                xlabel = 'Time - 2454833 [BKJD days]'
            elif self.time_format == 'btjd':
                xlabel = 'Time - 2457000 [BTJD days]'
            elif self.time_format == 'jd':
                xlabel = 'Time [JD]'
            else:
                xlabel = 'Time'
        # Default ylabel
        if ylabel is None:
            if normalize or (self.flux_unit == u.dimensionless_unscaled):
                ylabel = 'Normalized Flux'
            elif self.flux_unit is None:
                ylabel = 'Flux'
            else:
                ylabel = 'Flux [{}]'.format(self.flux_unit.to_string("latex_inline"))
        # Default legend label
        if ('label' not in kwargs):
            kwargs['label'] = self.label

        # Normalize the data if requested
        if normalize:
            lc_normed = self.normalize()
            flux, flux_err = lc_normed.flux, lc_normed.flux_err
        else:
            flux, flux_err = self.flux, self.flux_err

        # Make the plot
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(1)
            if method == 'scatter':
                sc = ax.scatter(self.time, flux, **kwargs)
                # Colorbars should only be plotted if the user specifies, and there is
                # a color specified that is not a string (e.g. 'C1') and is iterable.
                if show_colorbar and ('c' in kwargs) and \
                   (not isinstance(kwargs['c'], str)) and hasattr(kwargs['c'], '__iter__'):
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label(colorbar_label)
                    cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
                    cbar.ax.minorticks_off()
            elif method == 'errorbar':
                ax.errorbar(x=self.time, y=flux, yerr=flux_err, **kwargs)
            else:
                ax.plot(self.time, flux, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if (np.sum([len(a) for a in legend_labels]) != 0):
                ax.legend(loc='best')

        return ax

    def plot(self, **kwargs):
        """Plot the light curve using Matplotlib's `~matplotlib.pyplot.plot` method.

        Parameters
        ----------
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

    def scatter(self, colorbar_label='', show_colorbar=True, **kwargs):
        """Plots the light curve using Matplotlib's `~matplotlib.pyplot.scatter` method.

        Parameters
        ----------
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

    def errorbar(self, linestyle='', **kwargs):
        """Plots the light curve using Matplotlib's `~matplotlib.pyplot.errorbar` method.

        Parameters
        ----------
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
            >>> lc = lk.search_lightcurvefile('kepler-10', quarter=3).download()  # doctest: +SKIP
            >>> lc = lc.PDCSAP_FLUX.normalize().flatten()  # doctest: +SKIP
            >>> lc.interact_bls()  # doctest: +SKIP

        References
        ----------
        .. [1] http://docs.astropy.org/en/latest/stats/bls.html
        """
        from .interact_bls import show_interact_widget
        clean = self.remove_nans()
        return show_interact_widget(clean, notebook_url=notebook_url, minimum_period=minimum_period,
                                    maximum_period=maximum_period, resolution=resolution)

    def to_table(self):
        """Converts the light curve to an Astropy `~astropy.table.Table` object.

        Returns
        -------
        table : `astropy.table.Table`
            An AstroPy Table with columns 'time', 'flux', and 'flux_err'.
        """
        tbl = Table.from_pandas(self.to_pandas())
        if self.time_format is not None:
            tbl['time'] = self.astropy_time  # Ensure 'time' is an AstroPy `Time` object
        tbl.meta = self.meta
        return tbl

    def to_timeseries(self):
        """Converts the light curve to an AstroPy
        `~astropy.timeseries.TimeSeries` object.

        This feature requires AstroPy v3.2 or later (released in 2019).
        An `ImportError` will be raised if this version is not available.

        Returns
        -------
        timeseries : `~astropy.timeseries.TimeSeries`
            An AstroPy TimeSeries object.
        """
        try:
            from astropy.timeseries import TimeSeries
        except ImportError:
            raise ImportError("You need to install AstroPy v3.2 or later to "
                              "use the LightCurve.to_timeseries() method.")
        return TimeSeries(self.to_table())

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
        return StingrayLightcurve(time=self.time, counts=self.flux,
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

    def to_pandas(self, columns=('time', 'flux', 'flux_err')):
        """Converts the light curve to a Pandas `~pandas.DataFrame` object.

        By default, the object returned will contain the columns 'time', 'flux',
        and 'flux_err'.  This can be changed using the `columns` parameter.

        Parameters
        ----------
        columns : tuple of str
            List of columns to include in the DataFrame.  The names must match
            attributes of the `LightCurve` object (e.g. ``time``, ``flux``).

        Returns
        -------
        dataframe : `pandas.DataFrame`
            A data frame indexed by `time` and containing the columns ``flux``
            and ``flux_err``.
        """
        try:
            import pandas as pd
        # lightkurve does not require pandas, so check for import success.
        except ImportError:
            raise ImportError("You need to install pandas to use the "
                              "LightCurve.to_pandas() method.")
        data = {}
        for col in columns:
            if hasattr(self, col):
                data[col] = vars(self)[col]
                # We need to ensure pandas gets the native byteorder.
                # x86 uses little endian, so it is reasonable to assume that
                # we always want little endian, even though FITS uses big endian!
                # See https://github.com/KeplerGO/lightkurve/issues/188
                if data[col].dtype.byteorder == '>':  # is big endian?
                    data[col] = data[col].byteswap().newbyteorder()
        df = pd.DataFrame(data=data, columns=columns)
        return df

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
            Dictionary of arguments to be passed to `pandas.DataFrame.to_csv()`.

        Returns
        -------
        csv : str or None
            Returns a csv-formatted string if ``path_or_buf=None``.
            Returns `None` otherwise.
        """
        return self.to_pandas().to_csv(path_or_buf=path_or_buf, **kwargs)

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

    def to_fits(self, path=None, overwrite=False, flux_column_name='FLUX', **extra_data):
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
                cols.append(fits.Column(name='TIME', format='D', unit=self.time_format,
                                        array=self.time))
            if ~np.asarray([flux_column_name in k.upper() for k in extra_data.keys()]).any():
                cols.append(fits.Column(name=flux_column_name, format='E',
                                        unit='counts', array=self.flux))
            if 'flux_err' in dir(self):
                if ~(flux_column_name.upper() + '_ERR' in extra_data.keys()):
                    cols.append(fits.Column(name=flux_column_name.upper() + '_ERR', format='E',
                                            unit='counts', array=self.flux_err))
            if 'cadenceno' in dir(self):
                if ~np.asarray(['CADENCENO' in k.upper() for k in extra_data.keys()]).any():
                    cols.append(fits.Column(name='CADENCENO', format='J',
                                            array=self.cadenceno))
            for kw in extra_data:
                if isinstance(extra_data[kw], (np.ndarray, list)):
                    cols.append(fits.Column(name='{}'.format(kw).upper(),
                                            format=typedir[type(extra_data[kw][0])],
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

    def plot_river(self, period, t0=0, ax=None, bin_points=1,
                       minimum_phase=-0.5, maximum_phase=0.5, method='mean',
                       **kwargs):
        """Plot the light curve as a river plot.

        A river plot uses colors to represent the light curve values in
        chronological order, relative to the period of an interesting signal.
        Each row in the plot represents a full period cycle, and each column
        represents a fixed phase.  This type of plot is often used to visualize
        Transit Timing Variations (TTVs) in the light curves of exoplanets.

        All extra keywords supplied are passed on to Matplotlib's
        `~matplotlib.pyplot.pcolormesh` function.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        period: float
            Period at which to fold the light curve
        t0 : float
            Phase mid point for plotting
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

        if hasattr(self, 'time_original'):
            time = self.time_original
        else:
            time = self.time

        s = np.argsort(time)
        x, y, e = time[s], self.flux[s], self.flux_err[s]
        med = np.nanmedian(self.flux)
        e /= med
        y /= med

        # Here `ph` is the phase of each time point x
        # cyc is the number of cycles that have occured at each time point x
        # since the phase 0 before x[0]
        n = int(period/np.nanmedian(np.diff(x)) * (maximum_phase - minimum_phase)/bin_points)
        if n == 1:
            bin_points = int(maximum_phase - minimum_phase)/(2 / int(period/np.nanmedian(np.diff(x))))
            warnings.warn('`bin_points` is too high to plot a phase curve, resetting to {}'.format(bin_points),
                          LightkurveWarning)
            n = 2
        ph = x/period % 1
        cyc = np.asarray((x - x % period)/period, int)
        cyc -= np.min(cyc)

        phase = (t0 % period) / period
        ph = ((x - (phase * period)) / period) % 1
        cyc = np.asarray((x - ((x - phase * period) % period))/period, int)
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
            ar *= np.nanmedian(self.flux)

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
                if self.flux_unit is not None:
                    if (self.flux_unit != u.dimensionless_unscaled):
                        unit = '[{}]'.format(self.flux_unit.to_string('latex'))
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
            ax.set_title(self.label)
            a = cyc.max() * 0.1 / 12.
            b = (cyc.max() - cyc.min()) / (bs.max() - bs.min())
            ax.set_aspect(a/b)
        return ax


class FoldedLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    in which the ``time`` parameter represents phase values.

    Compared to the `~lightkurve.lightcurve.LightCurve` base class, this class
    takes three extra parameters (``period``, ``t0``, ``time_original``),
    offers extra properties (`phase`, `odd_mask`, `even_mask`),
    and implements different plotting defaults.
    """

    def __init__(self, time=None, flux=None, flux_err=None, period=None, t0=None,
                 time_original=None, *args, **kwargs):
        self.period = period
        self.t0 = t0
        self.time_original = time_original
        super(FoldedLightCurve, self).__init__(time=time, flux=flux,
            flux_err=flux_err, *args, **kwargs)

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
        cycle = (self.time_original - self.time * (self.period) - self.period * 0.5) / (self.period * 2)
        return (cycle % 1) < 0.5

    @property
    def even_mask(self):
        """Boolean mask which flags the even-numbered cycles (2, 4, 6, etc).

        See the documentation of `odd_mask` for examples.
        """
        return ~self.odd_mask

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
        ax = super(FoldedLightCurve, self).plot(**kwargs)
        if 'xlabel' not in kwargs:
            ax.set_xlabel("Phase")
        return ax

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
        ax = super(FoldedLightCurve, self).scatter(**kwargs)
        if 'xlabel' not in kwargs:
            ax.set_xlabel("Phase")
        return ax

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
        ax = super(FoldedLightCurve, self).errorbar(**kwargs)
        if 'xlabel' not in kwargs:
            ax.set_xlabel("Phase")
        return ax

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
        ax = super(FoldedLightCurve, self).plot_river(self.period, self.t0, **kwargs)
        return ax


class KeplerLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    which holds extra data specific to the Kepler mission.

    Attributes
    ----------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    flux_unit : `~astropy.units.Unit` or str
        Unit of the flux values.  If a string is passed, it will be passed
        on the the constructor of `~astropy.units.Unit`.
    time_format : str
        String specifying how an instant of time is represented,
        e.g. 'bkjd' or 'jd'.
    time_scale : str
        String which specifies how the time is measured,
        e.g. tdb', 'tt', 'ut1', or 'utc'.
    centroid_col : array-like
        Centroid column coordinates as a function of time
    centroid_row : array-like
        Centroid row coordinates as a function of time
    quality : array-like
        Array indicating the quality of each data point
    quality_bitmask : int
        Bitmask specifying quality flags of cadences that should be ignored
    channel : int
        Channel number
    campaign : int
        Campaign number
    quarter : int
        Quarter number
    mission : str
        Mission name
    cadenceno : array-like
        Cadence numbers corresponding to every time measurement
    targetid : int
        Kepler ID number
    """

    extra_columns = ("quality", "cadenceno", "centroid_col", "centroid_row")


    def __init__(self, time=None, flux=None, flux_err=None,
                 flux_unit=u.Unit('electron/second'), time_format='bkjd', time_scale='tdb',
                 centroid_col=None, centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None,
                 cadenceno=None, targetid=None, ra=None, dec=None, label=None, meta=None):
        super(KeplerLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err, flux_unit=flux_unit,
                                               time_format=time_format, time_scale=time_scale,
                                               targetid=targetid, label=label, meta=meta)
        self.centroid_col = self._validate_array(centroid_col, name='centroid_col')
        self.centroid_row = self._validate_array(centroid_row, name='centroid_row')
        self.quality = self._validate_array(quality, name='quality')
        self.cadenceno = self._validate_array(cadenceno, name='cadenceno')
        self.quality_bitmask = quality_bitmask
        self.channel = channel
        self.campaign = campaign
        self.quarter = quarter
        self.mission = mission
        self.ra = ra
        self.dec = dec

    def __repr__(self):
        return('KeplerLightCurve(ID: {})'.format(self.targetid))

    def to_pandas(self, columns=('time', 'flux', 'flux_err', 'quality',
                                 'centroid_col', 'centroid_row')):
        """Converts the light curve to a Pandas `~pandas.DataFrame` object.

        By default, the object returned will contain the columns 'time', 'flux',
        'flux_err', 'quality', 'centroid_col', and 'centroid_row'.
        This can be changed using the `columns` parameter.

        Parameters
        ----------
        columns : list of str
            List of columns to include in the DataFrame.  The names must match
            attributes of the `LightCurve` object (e.g. `time`, `flux`).

        Returns
        -------
        dataframe : `pandas.DataFrame` object
            A dataframe indexed by `time` and containing the columns `flux`
            and `flux_err`.
        """
        return super(KeplerLightCurve, self).to_pandas(columns=columns)

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

        if ('quarter' in dir(self)) and (self.quarter is not None):
            hdu[0].header['QUARTER'] = self.quarter
        elif ('campaign' in dir(self)) and self.campaign is not None:
            hdu[0].header['CAMPAIGN'] = self.campaign
        else:
            log.warning('Cannot find Campaign or Quarter number.')

        hdu = _make_aperture_extension(hdu, aperture_mask)

        if path is not None:
            hdu.writeto(path, overwrite=overwrite, checksum=True)
        else:
            return hdu


class TessLightCurve(LightCurve):
    """Subclass of :class:`LightCurve <lightkurve.lightcurve.LightCurve>`
    which holds extra data specific to the TESS mission.

    Attributes
    ----------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    flux_unit : `~astropy.units.Unit` or str
        Unit of the flux values.  If a string is passed, it will be passed
        on the the constructor of `~astropy.units.Unit`.
    time_format : str
        String specifying how an instant of time is represented,
        e.g. 'bkjd' or 'jd'.
    time_scale : str
        String which specifies how the time is measured,
        e.g. tdb', 'tt', 'ut1', or 'utc'.
    centroid_col, centroid_row : array-like, array-like
        Centroid column and row coordinates as a function of time
    quality : array-like
        Array indicating the quality of each data point
    quality_bitmask : int
        Bitmask specifying quality flags of cadences that should be ignored
    cadenceno : array-like
        Cadence numbers corresponding to every time measurement
    targetid : int
        Tess Input Catalog ID number
    """

    extra_columns = ("quality", "cadenceno", "centroid_col", "centroid_row")

    def __init__(self, time=None, flux=None, flux_err=None,
                 flux_unit=u.Unit('electron/second'), time_format='btjd', time_scale='tdb',
                 centroid_col=None, centroid_row=None, quality=None, quality_bitmask=None,
                 cadenceno=None, sector=None, camera=None, ccd=None,
                 targetid=None, ra=None, dec=None, label=None, meta=None):
        super(TessLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err, flux_unit=flux_unit,
                                             time_format=time_format, time_scale=time_scale,
                                             targetid=targetid, label=label, meta=meta)
        self.centroid_col = self._validate_array(centroid_col, name='centroid_col')
        self.centroid_row = self._validate_array(centroid_row, name='centroid_row')
        self.quality = self._validate_array(quality, name='quality')
        self.cadenceno = self._validate_array(cadenceno)
        self.quality_bitmask = quality_bitmask
        self.mission = "TESS"
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.ra = ra
        self.dec = dec

    def __repr__(self):
        return('TessLightCurve(TICID: {})'.format(self.targetid))


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
            'MISSION': self.mission,
            'RA_OBJ': self.ra,
            'TELESCOP': self.mission,
            'CAMERA': self.camera,
            'CCD': self.ccd,
            'SECTOR': self.sector,
            'TARGETID': self.targetid,
            'DEC_OBJ': self.dec,
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
