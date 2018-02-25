"""Defines LightCurve, KeplerLightCurve, TessLightCurve, etc."""

from __future__ import division, print_function

import copy
from tqdm import tqdm

import oktopus
import numpy as np
from scipy import signal
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from astropy.stats import sigma_clip
from astropy.table import Table

from .utils import running_mean


__all__ = ['LightCurve', 'KeplerLightCurve', 'TessLightCurve',
           'iterative_box_period_search']


class LightCurve(object):
    """
    Implements a simple class for a generic light curve.

    Attributes
    ----------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    meta : dict
        Free-form metadata associated with the LightCurve.
    """

    def __init__(self, time, flux, flux_err=None, meta={}):
        self.time, self.flux, self.flux_err = self._validate_inputs(time, flux, flux_err)
        self.meta = meta

    def _validate_inputs(self, time, flux, flux_err):
        if flux_err is not None:
            flux_err = np.asarray(flux_err)
        else:
            flux_err = np.nan * np.ones_like(flux)

        if not (len(time) == len(flux)):
            raise ValueError("Input arrays have different lengths."
                             " len(time)={}, len(flux)={}, len(flux_err)={}"
                             .format(len(time), len(flux), len(flux_err)))

        return np.asarray(time), np.asarray(flux), flux_err

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

    def stitch(self, others):
        """
        Stitches LightCurve objects.

        Parameters
        ----------
        others : LightCurve object or list of LightCurve objects
            Light curves to be stitched.

        Returns
        -------
        stitched_lc : LightCurve object
            Stitched light curve.
        """
        if not hasattr(others, '__iter__'):
            others = [others]
        time = self.time
        flux = self.flux
        flux_err = self.flux_err

        for i in range(len(others)):
            time = np.append(time, others[i].time)
            flux = np.append(flux, others[i].flux)
            flux_err = np.append(flux_err, others[i].flux_err)

        return LightCurve(time=time, flux=flux, flux_err=flux_err)

    def flatten(self, window_length=101, polyorder=3, return_trend=False, **kwargs):
        """
        Removes low frequency trend using scipy's Savitzky-Golay filter.

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
        **kwargs : dict
            Dictionary of arguments to be passed to `scipy.signal.savgol_filter`.

        Returns
        -------
        flatten_lc : LightCurve object
            Flattened lightcurve.
        If `return_trend` is `True`, the method will also return:
        trend_lc : LightCurve object
            Trend in the lightcurve data
        """
        lc_clean = self.remove_nans()  # The SG filter does not allow NaNs
        trend_signal = signal.savgol_filter(x=lc_clean.flux,
                                            window_length=window_length,
                                            polyorder=polyorder, **kwargs)
        flatten_lc = copy.copy(lc_clean)
        flatten_lc.flux = lc_clean.flux / trend_signal
        if flatten_lc.flux_err is not None:
            flatten_lc.flux_err = lc_clean.flux_err / trend_signal

        if return_trend:
            trend_lc = copy.copy(lc_clean)
            trend_lc.flux = trend_signal
            return flatten_lc, trend_lc
        return flatten_lc

    def fold(self, period, phase=0.):
        """Folds the lightcurve at a specified ``period`` and ``phase``.

        This method returns a new ``LightCurve`` object in which the time
        values range between -0.5 to +0.5.  Data points which occur exactly
        at ``phase`` or an integer multiple of `phase + n*period` have time
        value 0.0.

        Parameters
        ----------
        period : float
            The period upon which to fold.
        phase : float, optional
            Time reference point.

        Returns
        -------
        folded_lightcurve : LightCurve object
            A new ``LightCurve`` in which the data are folded and sorted by
            phase.
        """
        fold_time = (((self.time - phase * period) / period) % 1)
        # fold time domain from -.5 to .5
        fold_time[fold_time > 0.5] -= 1
        sorted_args = np.argsort(fold_time)
        if self.flux_err is None:
            return LightCurve(fold_time[sorted_args],
                              self.flux[sorted_args])
        return LightCurve(fold_time[sorted_args],
                          self.flux[sorted_args],
                          flux_err=self.flux_err[sorted_args])

    def normalize(self):
        """Returns a normalized version of the lightcurve.

        The normalized lightcurve is obtained by dividing `flux` and `flux_err`
        by the median flux.

        Returns
        -------
        normalized_lightcurve : LightCurve object
            A new ``LightCurve`` in which `flux` and `flux_err` are divided
            by the median.
        """
        lc = copy.copy(self)
        if lc.flux_err is not None:
            lc.flux_err = lc.flux_err / np.nanmedian(lc.flux)
        lc.flux = lc.flux / np.nanmedian(lc.flux)
        return lc

    def remove_nans(self):
        """Removes cadences where the flux is NaN.

        Returns
        -------
        clean_lightcurve : LightCurve object
            A new ``LightCurve`` from which NaNs fluxes have been removed.
        """
        lc = copy.copy(self)
        nan_mask = np.isnan(lc.flux)
        lc.time = self.time[~nan_mask]
        lc.flux = self.flux[~nan_mask]
        if lc.flux_err is not None:
            lc.flux_err = self.flux_err[~nan_mask]
        return lc

    def remove_outliers(self, sigma=5., return_mask=False, **kwargs):
        """Removes outlier flux values using sigma-clipping.

        This method returns a new LightCurve object from which flux values
        are removed if they are separated from the mean flux by `sigma` times
        the standard deviation.

        Parameters
        ----------
        sigma : float
            The number of standard deviations to use for clipping outliers.
            Defaults to 5.
        return_mask : bool
            Whether or not to return the mask indicating which data points
            were removed. Entries marked as `True` are considered outliers.
        **kwargs : dict
            Dictionary of arguments to be passed to `astropy.stats.sigma_clip`.

        Returns
        -------
        clean_lightcurve : LightCurve object
            A new ``LightCurve`` in which outliers have been removed.
        """
        new_lc = copy.copy(self)
        outlier_mask = sigma_clip(data=new_lc.flux, sigma=sigma, **kwargs).mask
        new_lc.time = self.time[~outlier_mask]
        new_lc.flux = self.flux[~outlier_mask]
        if new_lc.flux_err is not None:
            new_lc.flux_err = self.flux_err[~outlier_mask]
        if hasattr(new_lc, 'centroid_col') and new_lc.centroid_col is not None:
            new_lc.centroid_col = self.centroid_col[~outlier_mask]
        if hasattr(new_lc, 'centroid_row') and new_lc.centroid_row is not None:
            new_lc.centroid_row = self.centroid_row[~outlier_mask]
        if return_mask:
            return new_lc, outlier_mask
        return new_lc

    def bin(self, binsize=13, method='mean'):
        """Bins a lightcurve using a function defined by `method`
        on blocks of samples of size `binsize`.

        Parameters
        ----------
        binsize : int
            Number of cadences to include in every bin.
        method: str, one of 'mean' or 'median'
            The summary statistic to return for each bin. Default: 'mean'.

        Returns
        -------
        binned_lc : LightCurve object
            Binned lightcurve.

        Notes
        -----
        - If the ratio between the lightcurve length and the binsize is not
          a whole number, then the remainder of the data points will be
          ignored.
        - If the original lightcurve contains flux uncertainties (flux_err),
          the binned lightcurve will report the root-mean-square error.
          If no uncertainties are included, the binned curve will return the
          standard deviation of the data.
        """
        available_methods = ['mean', 'median']
        if method not in available_methods:
            raise ValueError("method must be one of: {}".format(available_methods))
        methodf = np.__dict__['nan' + method]

        n_bins = self.flux.size // binsize
        binned_lc = copy.copy(self)
        binned_lc.time = np.array([methodf(a) for a in np.array_split(self.time, n_bins)])
        binned_lc.flux = np.array([methodf(a) for a in np.array_split(self.flux, n_bins)])

        if self.flux_err is not None:
            # root-mean-square error
            binned_lc.flux_err = np.array(
                                    [np.sqrt(np.nansum(a**2))
                                     for a in np.array_split(self.flux_err, n_bins)]
                                 ) / binsize
        else:
            # compute the standard deviation from the data
            binned_lc.flux_err = np.array([np.nanstd(a)
                                           for a in np.array_split(self.flux, n_bins)])

        return binned_lc

    def cdpp(self, transit_duration=13, savgol_window=101, savgol_polyorder=2,
             sigma_clip=5.):
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
               the mean by `sigma_clip` times the standard deviation.
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
        sigma_clip : float, optional
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
        try:
            transit_duration = int(transit_duration)
        except Exception:
            raise Exception("transit_duration must be an integer, got {}."
                            .format(transit_duration))

        detrended_lc = self.flatten(window_length=savgol_window,
                                    polyorder=savgol_polyorder)
        cleaned_lc = detrended_lc.remove_outliers(sigma=sigma_clip)
        mean = running_mean(data=cleaned_lc.flux, window_size=transit_duration)
        cdpp_ppm = np.std(mean) * 1e6
        return cdpp_ppm

    def plot(self, ax=None, normalize=True, xlabel='Time - 2454833 (days)',
             ylabel='Normalized Flux', title=None, color='#363636', linestyle="",
             fill=False, grid=True, **kwargs):
        """Plots the light curve.

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
        color: str
            Color to plot flux points
        fill: bool
            Shade the region between 0 and flux
        grid: bool
            Add a grid to the plot
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        if normalize:
            normalized_lc = self.normalize()
            flux, flux_err = normalized_lc.flux, normalized_lc.flux_err
        else:
            flux, flux_err = self.flux, self.flux_err
        if flux_err is None:
            ax.plot(self.time, flux, marker='o', color=color,
                    linestyle=linestyle, **kwargs)
        else:
            ax.errorbar(self.time, flux, flux_err, color=color,
                        linestyle=linestyle, **kwargs)
        if fill:
            ax.fill(self.time, flux, fc='#a8a7a7', linewidth=0.0, alpha=0.3)
        if grid:
            ax.grid(alpha=0.3)
        if 'label' in kwargs:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel, {'color': 'k'})
        ax.set_ylabel(ylabel, {'color': 'k'})
        return ax

    def to_table(self):
        """Export the LightCurve as an AstroPy Table.

        Returns
        -------
        table : `astropy.table.Table` object
            An AstroPy Table with columns 'time', 'flux', and 'flux_err'.
        """
        return Table(data=(self.time, self.flux, self.flux_err),
                     names=('time', 'flux', 'flux_err'),
                     meta=self.meta)

    def to_pandas(self):
        """Export the LightCurve as a Pandas DataFrame.

        Returns
        -------
        dataframe : `pandas.DataFrame` object
            A dataframe indexed by `time` and containing the columns `flux`
            and `flux_err`.
        """
        try:
            import pandas as pd
        # lightkurve does not require pandas, so check for import success.
        except ImportError:
            raise ImportError("You need to install pandas to use the "
                              "LightCurve.to_pandas() method.")
        df = pd.DataFrame(data={'flux': self.flux, 'flux_err': self.flux_err},
                          index=self.time,
                          columns=['flux', 'flux_err'])
        df.index.name = 'time'
        df.meta = self.meta
        return df

    def to_csv(self, path_or_buf=None, **kwargs):
        """Writes the LightCurve to a csv file.

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
            Returns a csv-formatted string if `path_or_buf=None`,
            returns None otherwise.
        """
        return self.to_pandas().to_csv(path_or_buf=path_or_buf, **kwargs)


class KeplerLightCurve(LightCurve):
    """Defines a light curve class for NASA's Kepler and K2 missions.

    Attributes
    ----------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    centroid_col, centroid_row : array-like, array-like
        Centroid column and row coordinates as a function of time
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
    keplerid : int
        Kepler ID number
    """

    def __init__(self, time, flux, flux_err=None, centroid_col=None,
                 centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None,
                 cadenceno=None, keplerid=None):
        super(KeplerLightCurve, self).__init__(time, flux, flux_err)
        self.centroid_col = centroid_col
        self.centroid_row = centroid_row
        self.quality = quality
        self.quality_bitmask = quality_bitmask
        self.channel = channel
        self.campaign = campaign
        self.quarter = quarter
        self.mission = mission
        self.cadenceno = cadenceno
        self.keplerid = keplerid

    def __repr__(self):
        if self.mission is None:
            return('KeplerLightCurve(ID: {})'.format(self.keplerid))
        elif self.mission.lower() == 'kepler':
            return('KeplerLightCurve(KIC: {})'.format(self.keplerid))
        elif self.mission.lower() == 'k2':
            return('KeplerLightCurve(EPIC: {})'.format(self.keplerid))

    def correct(self, method='sff', **kwargs):
        """Corrects a lightcurve for motion-dependent systematic errors.

        Parameters
        ----------
        method : str
            Method used to correct the lightcurve.
            Right now only 'sff' (Vanderburg's Self-Flat Fielding) is supported.
        kwargs : dict
            Dictionary of keyword arguments to be passed to the function
            defined by `method`.

        Returns
        -------
        new_lc : KeplerLightCurve object
            Corrected lightcurve
        """
        not_nan = np.isfinite(self.flux)
        if method == 'sff':
                from .correctors import SFFCorrector
                self.corrector = SFFCorrector()
                corrected_lc = self.corrector.correct(time=self.time[not_nan],
                                                      flux=self.flux[not_nan],
                                                      centroid_col=self.centroid_col[not_nan],
                                                      centroid_row=self.centroid_row[not_nan],
                                                      **kwargs)
        else:
            raise ValueError("method {} is not available.".format(method))
        new_lc = copy.copy(self)
        new_lc.time = corrected_lc.time
        new_lc.flux = corrected_lc.flux
        new_lc.flux_err = self.normalize().flux_err[not_nan]
        return new_lc

    def to_fits(self):
        raise NotImplementedError()


class TessLightCurve(LightCurve):
    """Defines a light curve class for NASA's TESS mission.

    Attributes
    ----------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    centroid_col, centroid_row : array-like, array-like
        Centroid column and row coordinates as a function of time
    quality : array-like
        Array indicating the quality of each data point
    quality_bitmask : int
        Bitmask specifying quality flags of cadences that should be ignored
    cadenceno : array-like
        Cadence numbers corresponding to every time measurement
    ticid : int
        Tess Input Catalog ID number
    """
    def __init__(self, time, flux, flux_err=None, centroid_col=None,
                 centroid_row=None, quality=None, quality_bitmask=None,
                 cadenceno=None, ticid=None):
        super(TessLightCurve, self).__init__(time, flux, flux_err)
        self.centroid_col = centroid_col
        self.centroid_row = centroid_row
        self.quality = quality
        self.quality_bitmask = quality_bitmask
        self.mission = "TESS"
        self.cadenceno = cadenceno
        self.ticid = ticid

    def __repr__(self):
        return('TessLightCurve(TICID: {})'.format(self.ticid))

    def to_fits(self):
        raise NotImplementedError()


def iterative_box_period_search(lc, niters=2, min_period=0.5, max_period=30,
                                nperiods=501, period_scale='log'):
    """
    Implements a routine to find box-like transit events.
    This function fits a "box" model defined as:

    .. math::

        \Pi (t) = h - d\cdot \mathbb{I}(t_0 \leq t_i \leq t_0 + w)

    in which :math:`\mathbb{I}` is the indicator function.

    It turns out that, in a iid Gaussian noise setting, the parameters
    :math:`h` and :math:`d`, respectively, the amplitude and the depth
    of the box, can be solved analytically.

    Hence, this function iterates between two procedures:
    (i) compute :math:`h` and :math:`d` for given :math:`t_0` and :math:`w`
    (ii) numerically optimize for :math:`t_0` and :math:`w` given :math:`h`
    and :math:`d`.

    This procedure is done to a list of `nperiods` periods between `min_period`
    and `max_period`. It's assumed that the best period is the one that
    maximizes the posterior probability of the fit.

    Parameters
    ----------
    lc : LightCurve object
        An object from KeplerLightCurve or LightCurve.
        Note that flattening the lightcurve beforehand does aid the quest
        for the transit period.
    min_period : float
        Minimum period to search for. Units must be the same as `lc.time`.
    max_period : float
        Maximum period to search for. Units must be the same as `lc.time`.
    nperiods : int
        Number of periods to search between `min_period` and `max_period`.
    period_scale : str
        Type of the scale used to create the grid of periods between `min_period`
        and `max_period` used to search for the best period.
        Options are `linear`, `log`, or `inverse`.

    Returns
    -------
    log_posterior : list
        Log posterior (up to an additive constant) of the fit. The "best"
        period is therefore the one that maximizes the log posterior
        probability.
    trial_periods : numpy array
        List of trial periods.
    best_period : float
        Best period.

    Notes
    -----
    This function is experimental. Changes may be made in both its signature
    and implementation.
    """

    t = np.linspace(-.5, .5, len(lc.time))
    logprior_to = oktopus.prior.UniformPrior(lb=-.5, ub=.5)
    logprior_width = oktopus.prior.UniformPrior(lb=1e-3, ub=.2)
    m = np.nanmean(lc.flux)

    def logposterior(args, amplitude=None, depth=None, data=None):
        """Defines the negative of log of the joint posterior distribution of
        the start time of the transit `to` and the transit duration `w`.
        """
        to, width = args
        out_of_transit = (t < to) + (t >= to + width)
        in_transit = np.logical_not(out_of_transit)
        ll = ((data - amplitude) ** 2 * out_of_transit
              + (data - (amplitude - depth)) ** 2 * in_transit)
        return np.nansum(ll) + logprior_to(to) + logprior_width(width)

    def opt_amplitude(width, depth):
        """The MAP estimator for the amplitude of the lightcurve given that
        `to` and `w` have joint uniform prior distribution.
        """
        return width * depth + m

    def opt_depth(to, width, data):
        """The MAP estimator for the transit depth given that `to` and `w`
        have joint uniform prior distribution.
        """
        in_transit = (t < to + width) * (t >= to)
        n = np.nanmean(data[in_transit])
        return (m - n) / (1 - width)

    if period_scale == 'linear':
        trial_periods = np.linspace(min_period, max_period, nperiods)
    elif period_scale == 'log':
        trial_periods = np.logspace(np.log10(min_period), np.log10(max_period), nperiods)
    elif period_scale == 'inverse':
        trial_periods = 1 / np.linspace(1/max_period, 1/min_period, nperiods)
    else:
        raise ValueError("period_scale must be one of {}. Got {}."
                         .format("{'linear', 'log', 'inverse'}", period_scale))

    log_posterior, snr_d = [], []
    for p in tqdm(trial_periods):
        folded = lc.fold(period=p)
        # heuristically define initial guesses for the parameters
        amplitude_star, depth_star = m, 1e-4
        width_star = .1
        if np.nanmean(folded.flux[t < 0]) > np.nanmean(folded.flux[t > 0]):
            to_star = .25
        else:
            to_star = -.25
        for i in range(niters):
            # optimize the joint log posterior of to and width
            res = minimize(logposterior, x0=(to_star, width_star),
                           args=(amplitude_star, depth_star, folded.flux),
                           method='powell')
            to_star, width_star = res.x
            # compute the depth and amplitude using MAP
            depth_star = opt_depth(to_star, width_star, folded.flux)
            amplitude_star = opt_amplitude(width_star, depth_star)
        log_posterior.append(-res.fun)
        snr_d.append(depth_star * np.sqrt(width_star))

    return log_posterior, trial_periods, trial_periods[np.argmax(log_posterior)]
