"""Defines GPCorrector"""

import logging
log = logging.getLogger(__name__)


from tqdm import tqdm
from copy import deepcopy


from scipy.optimize import minimize
from matplotlib import pyplot as plt
import numpy as np


from ..lightcurve import LightCurve
try:
    import celerite
except ImportError:
    log.error('celerite is not installend. GPCorrector depends on celerite. Please install to continue.')

class GPCorrectorError(Exception):
    """Raised if there is a problem masking the data."""
    pass


class GPCorrector(object):
    '''Remove long term trends by fitting a Gaussian Process using celerite
    Uses `celerite` and a Matern 3/2 Kernel and a Jitter term to model the
    long term noise.
    '''


    def __init__(self, lc):
        if np.any([np.isnan(lc.flux), np.isnan(lc.time), np.isnan(lc.flux_err)]):
            raise GPCorrectorError('Light curve contains NaN values.')
        self.lc = lc.copy().normalize()
        self.break_tolerance = 10
        self.mask = np.ones(len(lc.flux), dtype=bool)
        self.dt = self.lc.time[1:] - self.lc.time[0:-1]
        self.dt_min = np.nanmin(self.dt)
        self.breaks = self._find_breaks()

    def _find_breaks(self):
        '''Find breaks in data collection that are more than the break_tolerance.
        '''
        breaks = np.where(self.dt > self.break_tolerance * np.nanmin(self.dt))[0] + 1
        breaks = np.append([0], breaks)
        breaks = np.append(breaks, len(self.lc.time))
        return breaks

    def _initialize(self, timescale=None, initial_parameters=None, bounds=None):
        '''Initialize the celerite GP model.

        Parameters
        ----------
        timescale : float or None
            If not None, sets the lower bound for optimization of the GP. Will
            not fit trends on timescales near to or shorter than this timescale.
        initial_parameters : list of floats
            Initial guesses for the log(sigma) (amplitude) and log(rho) (timescale)
            parameters. If none, these parameters will be generated for you.
        bounds : list of tuples
            Initial bounds for the log(sigma) (amplitude) and log(rho) (timescale)
            parameters. If none, these bounds will be generated for you.
        '''
        # Set up the kernels and bounds for celerite
        sigma = self.lc[self.mask].flux.std()
        jittersigma = self.lc[self.mask].estimate_cdpp() * 1e-6
        if timescale is not None:
            if bounds is None:
                bounds = [(np.log(0.1*sigma), np.log(10*sigma)),
                          (np.log(timescale/2), np.log(timescale*10))]
            log_rho = np.log(timescale)
        else:
            if bounds is None:
                bounds = [(-15, 15), (-15, 15)]
            log_rho = -np.log(0.26)
        if initial_parameters is None:
            kernel = celerite.terms.Matern32Term(log_sigma=np.log(sigma),
                                                 log_rho=log_rho, bounds=bounds)
        else:
            kernel = celerite.terms.Matern32Term(log_sigma=initial_parameters[0],
                                                 log_rho=initial_parameters[1], bounds=bounds)
        kernel += celerite.terms.JitterTerm(log_sigma=np.log(jittersigma))

        gp = celerite.GP(kernel, mean=1, fit_mean=True)
        self.initial_params = gp.get_parameter_dict()
        gp.compute(self.lc[self.mask].time - self.lc.time[0], self.lc[self.mask].flux_err)

        # Define a log liklihood to minimize
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)

        # Optimize the GP with scipy.minimize
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(neg_log_like, initial_params,
                        method="TNC", bounds=bounds, args=(self.lc[self.mask].flux, gp))
        self.soln = soln
        gp.set_parameter_vector(soln.x)
        self.best_fit_params = gp.get_parameter_dict()
        self.gp = gp


    def correct(self, timescale=2, iters=2, sigma=3, return_trend=False,
                bounds=None, initial_parameters=None, mask=None):
        '''Returns a lightcurve corrected by a GP.
        Runs the GP for times, split into different chunks based on gaps in
        data collection.

        Parameters
        ----------
        timescale : float (default 0.25 days)
            The time scale at which the user wishes to preserve signals.
        iters : int
            Number of times to iterate the correction. Each iteration successively
            masks out data that is poorly fit by the GP.
        return_trend : bool (default False)
            If True, returns a LightCurve object with the correction. If False,
            applies the correction to a copy of the original lc.s

        Returns
        -------
        corrected_lc : lightkurve.LightCurve object
            The corrected light curve with propagated errors
        flat_lc : lightkurve.LightCurve object (optional)
            If return_trend is True, will also return the best fit trend.
        '''

        if mask is None:
            mask = np.ones(len(self.lc.flux), dtype=bool)
        else:
            mask = ~np.copy(mask)

        # Iteratively sigma clip the data
        for iter in range(iters):
            # In the next run, include this new mask.
            self.mask &= mask
            if 100.*np.nansum(~self.mask)/len(self.mask) > 80:
                raise GPCorrectorError('Too large of a fraction of data is masked ({}%).'
                                        'Consider running fewer iterations, with a larger '
                                        'sigma tolerance or a shorter timescale.'
                                        ''.format(int(100.*np.nansum(~self.mask)/len(self.mask))))

            # Start the GP, find the maximum liklihood solution
            self._initialize(timescale=timescale, bounds=bounds, initial_parameters=initial_parameters)
            flat = LightCurve(time=self.lc.time, flux=self.lc.flux*0, flux_err=self.lc.flux_err*0)

            # Run over each 'portion' of the data, defined where breaks are greater
            # than break_tolerance
#            for idx in tqdm(range(len(self.breaks)-1), desc='Iteration {} ({}% Masked)'.format(iter + 1, int(100.*np.nansum(~self.mask)/len(self.mask)))):
            for idx in range(len(self.breaks)-1):
                l = self.lc[self.breaks[idx]:self.breaks[idx+1]]
                x = l.time - self.lc.time[0]
                # Calculate the best fit mean and errors
                pred_mean, pred_var = self.gp.predict(self.lc[self.mask].flux, x, return_var=True)
                pred_mean = pred_mean + self.gp.get_parameter_dict()['mean:value'] - 1
                pred_std = np.sqrt(pred_var)
                flat.flux[self.breaks[idx]:self.breaks[idx+1]] = pred_mean - 1
                flat.flux_err[self.breaks[idx]:self.breaks[idx+1]] = pred_std

            # Mask out any data that is different from the model by `sigma`
            corr = np.abs(self.lc.flux - flat.flux)
            corr -= np.nanmedian(corr)
            mask = corr < sigma * ((self.lc.flux_err**2 + (flat.flux_err)**2)**0.5)
#            mask = (convolve(mask, Box1DKernel(boxwidth)) == 1)

        # Return the best fit trend.
        flat.flux -= np.nanmedian(flat.flux)
        corrected_lc = deepcopy(self.lc)
        corrected_lc.flux -= flat.flux
        corrected_lc.flux_err = (self.lc.flux_err**2 + (flat.flux_err)**2)**0.5
        if return_trend:
            flat.flux += 1
            return corrected_lc, flat
        return corrected_lc
