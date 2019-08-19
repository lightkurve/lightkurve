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
from matplotlib import pyplot as plt
from copy import deepcopy

from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astropy import units as u

from . import PACKAGEDIR, MPLSTYLE
from .utils import (
    running_mean, bkjd_to_astropy_time, btjd_to_astropy_time,
    LightkurveWarning, validate_method
)

__all__ = ['LightCurve', 'KeplerLightCurve', 'TessLightCurve']

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
    Create a new `LightCurve` object, access the data,
    and apply binning as follows:

        >>> import lightkurve as lk
        >>> lc = lk.LightCurve(time=[1, 2, 3, 4], flux=[0.97, 1.01, 1.03, 0.99])
        >>> lc.time
        array([1, 2, 3, 4])
        >>> lc.flux
        array([0.97, 1.01, 1.03, 0.99])
        >>> lc.bin(binsize=2).flux
        array([0.99, 1.01])
    """
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None,
                 time_scale=None, targetid=None, label=None, meta=None):
        if time is None and flux is None:
            raise ValueError('either time or flux must be given')
        if time is None:
            self.time = np.arange(len(flux))
        else:
            self.time = np.asarray(time)
            # Trigger warning if time=NaN are present
            if np.isnan(self.time).any():
                warnings.warn('LightCurve object contains NaN times', LightkurveWarning)
        self.flux = self._validate_array(flux, name='flux')
        self.flux_err = self._validate_array(flux_err, name='flux_err')
        self.time_format = time_format
        self.time_scale = time_scale
        self.targetid = targetid
        self.label = label
        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

    def _validate_array(self, arr, name='array'):
        """Ensure the input arrays have the same length as `self.time`."""
        if arr is not None:
            arr = np.asarray(arr)
        else:
            arr = np.nan * np.ones_like(self.time)

        if not (len(self.time) == len(arr)):
            raise ValueError("Input arrays have different lengths."
                             " len(time)={}, len({})={}"
                             .format(len(self.time), name, len(arr)))
        return arr

    def __getitem__(self, key):
        copy_self = copy.copy(self)
        copy_self.time = self.time[key]
        copy_self.flux = self.flux[key]
        copy_self.flux_err = self.flux_err[key]
        return copy_self

    def __add__(self, other):
        copy_self = copy.copy(self)
        copy_self.flux = copy_self.flux + other
        return copy_self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        copy_self = copy.copy(self)
        copy_self.flux = other - copy_self.flux
        return copy_self

    def __mul__(self, other):
        copy_self = copy.copy(self)
        copy_self.flux = other * copy_self.flux
        copy_self.flux_err = abs(other) * copy_self.flux_err
        return copy_self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1./other)

    def __rtruediv__(self, other):
        copy_self = copy.copy(self)
        copy_self.flux = other / copy_self.flux
        return copy_self

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

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
            >>> lc = lk.search_lightcurvefile("Pi Mensae", mission="TESS", sector=1).download().PDCSAP_FLUX
            >>> lc
            TessLightCurve(TICID: 261136679)

        Every `LightCurve` object has a `time` attribute, which provides access
        to the original array of time values given in the native format and
        scale used by the data product from which the light curve was obtained::

            >>> lc.time
            array([1325.29698328, 1325.29837215, 1325.29976102, ..., 1353.17431099,
                   1353.17569985, 1353.17708871])
            >>> lc.time_format
            'btjd'
            >>> lc.time_scale
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
                res = getattr(self, attr)
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
        """
        Append LightCurve objects.

        Parameters
        ----------
        others : LightCurve object or list of LightCurve objects
            Light curves to be appended to the current one.

        Returns
        -------
        new_lc : LightCurve object
            Concatenated light curve.
        """
        if not hasattr(others, '__iter__'):
            others = [others]
        if inplace:
            new_lc = self
        else:
            new_lc = copy.copy(self)

        for i in range(len(others)):
            new_lc.time = np.append(new_lc.time, others[i].time)
            new_lc.flux = np.append(new_lc.flux, others[i].flux)
            new_lc.flux_err = np.append(new_lc.flux_err, others[i].flux_err)
            if hasattr(new_lc, 'cadenceno'):
                new_lc.cadenceno = np.append(new_lc.cadenceno, others[i].cadenceno)  # KJM
            if hasattr(new_lc, 'quality'):
                new_lc.quality = np.append(new_lc.quality, others[i].quality)
            if hasattr(new_lc, 'centroid_col'):
                new_lc.centroid_col = np.append(new_lc.centroid_col, others[i].centroid_col)
            if hasattr(new_lc, 'centroid_row'):
                new_lc.centroid_row = np.append(new_lc.centroid_row, others[i].centroid_row)
        return new_lc

    def copy(self):
        """Returns a copy of the LightCurve object.

        This method uses the `copy.deepcopy` function to ensure that all
        objects stored within the LightCurve are copied (e.g. time and flux).

        Returns
        -------
        lc_copy : LightCurve
            A new `LightCurve` object which is a copy of the original.
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
        flatten_lc : LightCurve object
            Flattened lightcurve.
        If ``return_trend`` is `True`, the method will also return:
        trend_lc : LightCurve object
            Trend in the lightcurve data
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
            # No outliers
            mask1 = np.nan_to_num(np.abs(self.flux[mask] - trend_signal)) <\
                    (np.nanstd(self.flux[mask] - trend_signal) * sigma)
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

        This method returns a `FoldedLightCurve` object in which the time
        values range between -0.5 to +0.5 (i.e. the phase).
        Data points which occur exactly at ``t0`` or an integer multiple of
        ``t0 + n*period`` will have phase value 0.0.

        Examples
        --------
        The example below shows a light curve with a period dip which occurs near
        time value 1001 and has a period of 5 days. Calling the `fold` method
        will transform the light curve into a `FoldedLightCurve` object::

            >>> import lightkurve as lk
            >>> lc = lk.LightCurve(time=range(1001, 1012), flux=[0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5])
            >>> folded_lc = lc.fold(period=5., t0=1006.)
            >>> folded_lc   # doctest: +SKIP
            <lightkurve.lightcurve.FoldedLightCurve>

        An object of type `FoldedLightCurve` is useful because it provides
        convenient access to the phase values and the phase-folded fluxes::

            >>> folded_lc.phase
            array([-0.4, -0.4, -0.2, -0.2,  0. ,  0. ,  0. ,  0.2,  0.2,  0.4,  0.4])
            >>> folded_lc.flux
            array([1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. ])

        We can still access the original time values as well::

            >>> folded_lc.time_original
            array([1004, 1009, 1005, 1010, 1001, 1006, 1011, 1002, 1007, 1003, 1008])

        A `FoldedLightCurve` inherits all the features of a standard `LightCurve`
        object. For example, we can very quickly obtain a phase-folded plot using:

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
        folded_lightcurve : `FoldedLightCurve`
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
                                label=self.label,
                                meta=self.meta)

    def normalize(self):
        """Returns a normalized version of the light curve.

        The normalized light curve is obtained by dividing the ``flux`` and
        ``flux_err`` object attributes by the by the median flux.

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
        """
        lc = self.copy()
        lc.flux_err = lc.flux_err / np.nanmedian(lc.flux)
        lc.flux = lc.flux / np.nanmedian(lc.flux)
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

        Parameters
        ----------
        method : string {'gaussian_noise'}
            Method to use for gap filling. Fills with gaussian noise by default

        Returns
        -------
        filled_lightcurve : `LightCurve`
            A new light curve object in which NaN values and gaps in time
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
            quality = np.zeros(len(ntime))
            quality[in_original] = np.copy(lc.quality)
            quality[~in_original] += 65536
            nlc.quality = quality
        if hasattr(lc, 'centroid_col'):
            col = np.zeros(len(ntime)) * np.nan
            col[in_original] = np.copy(lc.centroid_col)
            nlc.centroid_col = col
        if hasattr(lc, 'centroid_row'):
            row = np.zeros(len(ntime)) * np.nan
            row[in_original] = np.copy(lc.centroid_row)
            nlc.centroid_row = row

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
        sigma_lower : float or `None`
            The number of standard deviations to use as the lower bound for
            the clipping limit. Can be set to float('inf') in order to avoid
            clipping outliers below the median at all. If `None` then the
            value of ``sigma`` is used. Defaults to `None`.
        sigma_upper : float or `None`
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

    def bin(self, binsize=13, method='mean'):
        """Bins a lightcurve in blocks of size ``binsize``.

        The value of the bins will contain the mean (``method='mean'``) or the
        median (``method='median'``) of the original data.  The default is mean.

        Parameters
        ----------
        binsize : int
            Number of cadences to include in every bin.
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
        method = validate_method(method, supported_methods=['mean', 'median'])
        methodf = np.__dict__['nan' + method]

        n_bins = self.flux.size // binsize
        binned_lc = self.copy()
        indexes = np.array_split(np.arange(len(self.time)), n_bins)
        binned_lc.time = np.array([methodf(self.time[a]) for a in indexes])
        binned_lc.flux = np.array([methodf(self.flux[a]) for a in indexes])

        if np.any(np.isfinite(self.flux_err)):
            # root-mean-square error
            binned_lc.flux_err = np.array(
                [np.sqrt(np.nansum(self.flux_err[a]**2))
                 for a in indexes]
            ) / binsize
        else:
            # If the original light curve does not provide `flux_err`,
            # then report the standard deviations of the fluxes in each bin.
            binned_lc.flux_err = np.array([np.nanstd(self.flux[a]) for a in indexes])

        if hasattr(binned_lc, 'quality'):
            # Note: np.bitwise_or only works if there are no NaNs
            binned_lc.quality = np.array(
                [np.bitwise_or.reduce(a) if np.all(np.isfinite(a)) else np.nan
                 for a in np.array_split(self.quality, n_bins)])
        if hasattr(binned_lc, 'cadenceno'):
            binned_lc.cadenceno = np.array([np.nan] * n_bins)
        if hasattr(binned_lc, 'centroid_col'):
            # Note: nanmean/nanmedian yield a RuntimeWarning if a slice is all NaNs
            binned_lc.centroid_col = np.array(
                [methodf(a) if np.any(np.isfinite(a)) else np.nan
                 for a in np.array_split(self.centroid_col, n_bins)])
        if hasattr(binned_lc, 'centroid_row'):
            binned_lc.centroid_row = np.array(
                [methodf(a) if np.any(np.isfinite(a)) else np.nan
                 for a in np.array_split(self.centroid_row, n_bins)])

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

    def _create_plot(self, method='plot', ax=None, normalize=True,
                     xlabel=None, ylabel=None, title='', style='lightkurve',
                     show_colorbar=True, colorbar_label='',
                     **kwargs):
        """Implements `plot()`, `scatter()`, and `errorbar()` to avoid code duplication.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
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
            if normalize:
                ylabel = 'Normalized Flux'
            else:
                ylabel = 'Flux [e$^-$s$^{-1}$]'
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
                ax.legend()

        return ax

    def plot(self, **kwargs):
        """Plot the light curve using Matplotlib's `~matplotlib.pyplot.plot` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
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
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        return self._create_plot(method='plot', **kwargs)

    def scatter(self, colorbar_label='', show_colorbar=True, **kwargs):
        """Plots the light curve using Matplotlib's `~matplotlib.pyplot.scatter` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
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
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        return self._create_plot(method='scatter', colorbar_label=colorbar_label,
                                 show_colorbar=show_colorbar, **kwargs)

    def errorbar(self, linestyle='', **kwargs):
        """Plots the light curve using Matplotlib's `~matplotlib.pyplot.errorbar` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
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
        ax : matplotlib.axes._subplots.AxesSubplot
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
        """Converts the light curve to an `~astropy.timeseries.TimeSeries` object.

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
        """Create a new `LightCurve` from an `~astropy.timeseries.TimeSeries`.

        Parameters
        ----------
        ts : `~astropy.timeseries.TimeSeries`
            An AstroPy TimeSeries object.  The object must contain columns
            named 'time', 'flux', and 'flux_err'.
        """
        return LightCurve(time=ts['time'].value, flux=ts['flux'], flux_err=ts['flux_err'])

    def to_pandas(self, columns=('time', 'flux', 'flux_err')):
        """Converts the light curve to a Pandas `~pandas.DataFrame` object.

        Parameters
        ----------
        columns : list of str
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
        df = pd.DataFrame(data=data, index=self.time, columns=columns)
        df.index.name = 'time'
        return df

    def to_csv(self, path_or_buf=None, **kwargs):
        """Writes the light curve to a csv file.

        Parameters
        ----------
        path_or_buf : string or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.
        **kwargs : dict
            Dictionary of arguments to be passed to `pandas.DataFrame.to_csv()`.

        Returns
        -------
        csv : str or None
            Returns a csv-formatted string if ``path_or_buf=None``,
            returns None otherwise.
        """
        return self.to_pandas().to_csv(path_or_buf=path_or_buf, **kwargs)

    def to_periodogram(self, method="lombscargle", **kwargs):
        """Converts the light curve to a `~lightkurve.periodogram.Periodogram`
        power spectrum object.

        This method will call either `lightkurve.periodogram.LombScarglePeriodogram.from_lightcurve()`
        or `lightkurve.periodogram.BoxLeastSquaresPeriodogram.from_lightcurve()`,
        which in turn wrap `astropy.stats.LombScargle` and `astropy.stats.BoxLeastSquares`.

        Optional keywords accepted if ``method='lombscargle'`` are:
            ``minimum_frequency``, ``maximum_frequency``, ``mininum_period``,
            ``maximum_period``, ``frequency``, ``period``, ``nterms``,
            ``nyquist_factor``, ``oversample_factor``, ``freq_unit``,
            ``normalization``.

        Optional keywords accepted for ``method='bls'`` are:
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

        All **kwargs will be passed to the `to_periodogram()` method.

        Returns
        -------
        seismology : `~lightkurve.seismology.Seismology` object
            Object which can be used to estimate quick-look asteroseismic quantities.
        """
        from .seismology import Seismology
        return Seismology.from_lightcurve(self, **kwargs)

    def to_fits(self, path=None, overwrite=False, flux_column_name='FLUX', **extra_data):
        """Writes the light curve to a FITS file.

        Parameters
        ----------
        path : string, default ``None``
            If set, location where the FITS file will be written.
        overwrite : bool
            Whether or not to overwrite the file
        flux_column_name : str
            The name of the label for the FITS extension, e.g. SAP_FLUX or FLUX
        extra_data : dict
            Extra keywords or columns to include in the FITS file.
            Arguments of type str, int, float, or bool will be stored as
            keywords in the primary header.
            Arguments of type np.array or list will be stored as columns
            in the first extension.

        Returns
        -------
        hdu : `astropy.io.fits.HDUList`
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
                if ~np.asarray(['FLUX_ERR' in k.upper() for k in extra_data.keys()]).any():
                    cols.append(fits.Column(name='FLUX_ERR', format='E',
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


class FoldedLightCurve(LightCurve):
    """Generic class to store and plot phase-folded light curves.

    Compared to the standard `LightCurve` class, this class offers an extra
    `phase` property and implements different plotting defaults.
    """
    def __init__(self, *args, **kwargs):
        self.time_original = kwargs.pop("time_original", None)
        super(FoldedLightCurve, self).__init__(*args, **kwargs)

    @property
    def phase(self):
        return self.time

    def plot(self, **kwargs):
        """Plot the folded light curve usng matplotlib's `plot` method.

        See `LightCurve.plot` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments to be passed to `LightCurve.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        ax = super(FoldedLightCurve, self).plot(**kwargs)
        if 'xlabel' not in kwargs:
            ax.set_xlabel("Phase")
        return ax

    def scatter(self, **kwargs):
        """Plot the folded light curve usng matplotlib's `~matplotlib.pyplot.scatter` method.

        See `LightCurve.scatter` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments to be passed to `LightCurve.scatter`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        ax = super(FoldedLightCurve, self).scatter(**kwargs)
        if 'xlabel' not in kwargs:
            ax.set_xlabel("Phase")
        return ax

    def errorbar(self, **kwargs):
        """Plot the folded light curve usng matplotlib's `errorbar` method.

        See `LightCurve.scatter` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments to be passed to `LightCurve.scatter`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        ax = super(FoldedLightCurve, self).errorbar(**kwargs)
        if 'xlabel' not in kwargs:
            ax.set_xlabel("Phase")
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
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None, time_scale=None,
                 centroid_col=None, centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None,
                 cadenceno=None, targetid=None, ra=None, dec=None, label=None, meta=None):
        super(KeplerLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err,
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

    def __getitem__(self, key):
        lc = super(KeplerLightCurve, self).__getitem__(key)
        # Compared to `LightCurve`, we need to slice a few additional arrays:
        lc.quality = self.quality[key]
        lc.cadenceno = self.cadenceno[key]
        lc.centroid_col = self.centroid_col[key]
        lc.centroid_row = self.centroid_row[key]
        return lc

    def __repr__(self):
        return('KeplerLightCurve(ID: {})'.format(self.targetid))

    def to_pandas(self, columns=('time', 'flux', 'flux_err', 'quality',
                                 'centroid_col', 'centroid_row')):
        """Converts the light curve to a Pandas `~pandas.DataFrame` object.

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
            'SAP_QUALITY': self.quality}

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
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None, time_scale=None,
                 centroid_col=None, centroid_row=None, quality=None, quality_bitmask=None,
                 cadenceno=None, sector=None, camera=None, ccd=None,
                 targetid=None, ra=None, dec=None, label=None, meta=None):
        super(TessLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err,
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

    def __getitem__(self, key):
        lc = super(TessLightCurve, self).__getitem__(key)
        # Compared to `LightCurve`, we need to slice a few additional arrays:
        lc.quality = self.quality[key]
        lc.cadenceno = self.cadenceno[key]
        lc.centroid_col = self.centroid_col[key]
        lc.centroid_row = self.centroid_row[key]
        return lc

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
            'DEC_OBJ': self.dec}

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
