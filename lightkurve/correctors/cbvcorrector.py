"""Defines Corrector classes that utilize Kepler/K2/TESS Cotrending Basis Vectors.
"""
import logging

from tqdm import tqdm

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from astropy.io import fits as pyfits
from bs4 import BeautifulSoup
import urllib.request
import astropy.units as u

from .. import MPLSTYLE
from ..lightcurve import LightCurve, KeplerLightCurve
from ..search import search_lightcurve
from ..lightcurvefile import KeplerLightCurveFile
from .corrector import Corrector
from ..utils import channel_to_module_output, validate_method, print_dictionary
from .designmatrix import DesignMatrix, DesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from ..collections import LightCurveCollection

log = logging.getLogger(__name__)

# For Kepler/K2/TESS max number of stored CBVs has always been 16
# Nevertheless, set to a ridiculously high number to future-proof the code
MAX_NUMBER_CBVS = 99

# The number of bands for multi-scale MAP has also always been 3
# Still, set to a slightly higher number, just in case
# NOTE: This is a zero-based number
# NOTE: We don't want to set this number too high because we would be attempting
# to retrieve a lot of bands that will never exist, which takes a long time to
# attempt
MAX_NUMBER_BANDS = 5

__all__ = ['CotrendingBasisVectors', 'KeplerCotrendingBasisVectors',
        'TessCotrendingBasisVectors', 'get_kepler_cbvs', 'get_tess_cbvs', 'CBVCorrector']

#*******************************************************************************
# Corrector Class


class CBVCorrector(RegressionCorrector):
    """ Class for removing systematics using CBV correctors for Kepler/K2/TESS

    On construction of this object, the relevent CBVs will be downloaded from
    MAST appropriate for the lightcurve object passed to the constructor.

    For TESS there multiple CBV types. All are loaded and the user must specify
    which to use in the correction.

    Attributes
    ----------
    lc  : LightCurve
        The light curve to correct, stored zero-centered median normalized
    denormalized_lc  : LightCurve
        The light curve in electrons / second
    cbvs    : CetrendingBasisVectors list
        The retrieved CBVs, can contain multiple types
    cbv_design_matrix : DesignMatrix
        The retrieved CBVs ported into a DesignMatrix object
    extra_design_matrix : DesignMatrix
        An extra design matrix to include in the fit with the CBVs
    design_matrix_collection   : DesignMatrixCollection
        The design matrix collection composed of cbv_design_matrix and extra_design_matrix
    corrected_lc : LightCurve
        The returned light curve from regression_corrector.correct
        stored zero-centered median normalized
    denormalized_corrected_lc  : LightCurve
        The light curve in electrons / second
    coefficients : float ndarray
        The fit coefficients corresponding to the design_matric_collection
        As computed by regressioncorrection.correct
    coefficients_err : float ndarray
        The error estimates for the coefficients, see regressioncorrection
    model_lc : LightCurve
        The model fit to the lightcurve 'lc'
    diagnostic_lightcurves : dict
        Model fits for each of the sub design matrices fit in model_lc
    lc_neighborhood : LightCurveCollection
        SAP light curves of all targets within the defined neighborhood of the
        target understudy for use with the under-fitting metric
    cadence_mask : np.ndarray of bool
        Mask, where True indicates a cadence that was used in
        regressioncorrector.correct.
        Note: The saved cadence_mask is overwritten for each call to correct.
    over_fitting_score : float
        Over-fitting score from the most recent run of correct_optimizer
    under_fitting_score : float
        Under-fitting score from the most recent run of correct_optimizer
    alpha : float
        L2-norm regularization term used in most recent fit
        equal to:
            designmatrix prior sigma = np.median(self.lc.flux_err) / np.sqrt(alpha)


    """

    def __init__(self, lc, do_not_load_cbvs=False):
        """ Constructor for CBVClass objects

        This constructor will retrieve all relevant CBVs from MAST and then
        align them with the passed-in light curve.

        Parameters
        ----------
        lc  : LightCurve
            The light curve to correct
        do_not_load_cbvs : bool
            If True then the CBVs will NOT be loaded from MAST. 
            Use this option if you wish to use the CBV corrector methods with only a 
            custom design matrix (via the ext_dm argument in the corrector methods)

        Examples
        --------
        """

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')

        # This class wants zero-centered median normalized flux
        # Store the median value so we can denormalize
        assert  lc.flux.unit==u.Unit('electron / second'), \
            'cbvCorrector expects light curve to be passed in e-/s units.'        
        self._lc_median = np.nanmedian(lc.flux)
        lc = lc.remove_nans().normalize()
        lc.flux -= 1.0 

        # Call the RegresssionCorrector Constructor
        super(CBVCorrector, self).__init__(lc)

        self.lc_neighborhood = None

        #***
        # Retrieve all relevant CBVs from MAST
        # TODO: create CBV collection class
        cbvs = []

        if (not do_not_load_cbvs):
            if self.lc.mission == 'Kepler':
                cbvs.append(get_kepler_cbvs(mission=self.lc.mission, quarter=self.lc.quarter,
                        channel=self.lc.channel, cbv_indices='ALL'))
            elif self.lc.mission == 'K2':
                cbvs.append(get_kepler_cbvs(mission=self.lc.mission, campaign=self.lc.campaign,
                        channel=self.lc.channel, cbv_indices='ALL'))
            elif self.lc.mission == 'TESS':
                # For TESS we load multiple CBV types
            
                cbvs.append(get_tess_cbvs(sector=self.lc.sector,
                    camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='SingleScale',
                    cbv_indices='ALL'))
            
                for iBand in np.arange(1,MAX_NUMBER_BANDS+1):
                    iBand = int(iBand)
                    cbvObj = get_tess_cbvs(sector=self.lc.sector,
                        camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='MultiScale',
                        band=iBand, cbv_indices='ALL')
                    if (cbvObj.cbvEXTNAME is not None):
                        cbvs.append(cbvObj)
            
                cbvs.append(get_tess_cbvs(sector=self.lc.sector,
                    camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='Spike',
                    cbv_indices='ALL'))
            
            else:
                raise ValueError('Unknown mission type')
            
            for idx in np.arange(len(cbvs)):
                if (not isinstance(cbvs[idx], CotrendingBasisVectors)):
                    raise Exception('CBVs could not be loaded. CBVCorrector must exit')
            
            # Align the CBVs with the lightcurve flux using the cadence numbers
            # This will also trim the lightcurve if it contains cadences not in the
            # CBVs
            for idx in np.arange(len(cbvs)):
                self.lc = cbvs[idx].align(self.lc, trim_lc=True)

        self.cbvs = cbvs

        # Initiate all extra attributes to empty set
        self.cbv_design_matrix = None
        self.extra_design_matrix = None
        self.design_matrix_collection = None
        self.corrected_lc = None
        self.coefficients = None
        self.coefficients_err = None
        self.model_lc = None
        self.diagnostic_lightcurves = None
        self.lc_neighborhood = None
        self.cadence_mask = None
        self.over_fitting_score = None
        self.under_fitting_score = None
        self.alpha = None

    @property 
    def denormalized_lc(self): 
        """ The light curve is stored internally as zero-centered median 
        normalized. This getter returns the denormalized light curve in
        absolute flux units.
        """
        
        lc = self.lc.copy()
        lc.flux = (lc.flux+1.0) * self._lc_median
        lc.flux_err = lc.flux_err * self._lc_median
        assert  lc.flux.unit==u.Unit('electron / second'), \
            'Bookkeeping Error, flux should be in units of e-/s.'        

        return lc 

    @property 
    def denormalized_corrected_lc(self): 
        """ The light curve is stored internally as zero-centered median 
        normalized. This getter returns the denormalized light curve.
        """

        if (self.corrected_lc is None):
            log.warning('A corrected light curve does not exist, please run '
                    'correct first')
            return None
        
        lc = self.corrected_lc.copy()
        lc.flux = (lc.flux+1.0) * self._lc_median
        lc.flux_err = lc.flux_err * self._lc_median
        assert  lc.flux.unit==u.Unit('electron / second'), \
            'Bookkeeping Error, flux should be in units of e-/s.'        

        return lc 

    def correct_gaussian_prior(self, cbv_type=['SingleScale'],
            cbv_indices=[np.arange(1,9)], 
            alpha=1e-20, ext_dm=None, cadence_mask=None):
        """ Performs the correction using RegressionCorrector methods.

        This method will assemble the full design matrix collection composed of
        cbv_design_matrix and extra_design_matrix (ext_dm). It then uses the
        alpha L2-Norm (Ridge Regression) penalty term to set the width on the
        design matrix priors.  Then uses the super-class
        RegressionCorrector.correct to perform the correction.

        The relation between the L2-Norm alpha term and the Gaussian prior sigma
        is: 
        alpha = flux_sigma^2 / sigma^2

        By default this method will use the first 8 "SingleScale" basis vectors.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        alpha : float
            L2-norm regularization penatly term. Default = 1e-20
            {0 => no regularization}
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        `.LightCurve`
            Corrected light curve, with noise removed. In units of electrons / second

        Examples
        --------
        The following example will perform the correction using the
        SingleScale and Spike basis vectors with a weak regularization alpha
        term of 0.1. It also adds in an external design matrix to perfomr a
        joint fit.
            >>> cbv_type = ['SingleScale', 'Spike']
            >>> cbv_indices = [np.arange(1,9), 'ALL']
            >>> corrected_lc = cbvCorrector.correct_gaussian_prior(cbv_type=cbv_type, # doctest: +SKIP
            >>>     cbv_indices=cbv_indices, alpha=0.1, # doctest: +SKIP
            >>>     ext_dm=design_matrix ) # doctest: +SKIP
        """
        
        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm)

        # Add in a width to the Gaussian priors
        # alpha = flux_sigma^2 / sigma^2
        if (alpha == 0.0):
            sigma = None
        else:
            sigma = np.median(self.lc.flux_err) / np.sqrt(np.abs(alpha))
        self._set_prior_width(sigma)
            
        # Use RegressionCorrector.correct for the actual fitting
        super(CBVCorrector, self).correct(self.design_matrix_collection, 
                cadence_mask=cadence_mask)

        self.alpha = alpha

        return self.denormalized_corrected_lc

    def correct_ElasticNet(self, cbv_type='SingleScale', cbv_indices=np.arange(1,9), 
            alpha=1e-20, l1_ratio=0.01, ext_dm=None, cadence_mask=None):
        """ Performs the correction using scikit-learn's ElasticNet which
        utilizes combined L1- and L2-Norm priors as a regularizer.

        This method will assemble the full design matrix collection composed of
        cbv_design_matrix and extra_design_matrix (ext_dm). Then uses
        scikit-learn.linear_model.ElasticNet to perform the correction.

        By default this method will use the first 8 "SingleScale" basis vectors.

        Note that the alpha term in scikit-learn's ElasticNet does not have the
        same scaling as when used in CBVCorrector.correct_gaussian_prior or 
        CBVCorrector.correct_optimizer. Do not assume similar results with a
        similar alpha value.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        alpha : float
            L2-norm regularization pentaly term.
            {0 => no regularization}
        l1_ratio : float
            Elastic-Net mixing parameter
            l1_ratio = 0 => L2 penalty (Ridge). l1_ratio = 1 => L1 penalty (Lasso).
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        `.LightCurve`
            Corrected light curve, with noise removed. In units of electrons / second

        Examples
        --------
        The following example will perform the ElasticNet correction using the
        SingleScale and Spike basis vectors with a strong regualrization alpha
        term of 1.0 and an L1 ratio of 0.9 which means predominantly a Lasso
        regularization but a slight amount of Ridge Regression.
            >>> cbv_type = ['SingleScale', 'Spike']
            >>> cbv_indices = [np.arange(1,9), 'ALL']
            >>> corrected_lc = cbvCorrector.correct_ElasticNet(cbv_type=cbv_type, # doctest: +SKIP
            >>>     cbv_indices=cbv_indices, alpha=1.0, l1_ratio=0.9) # doctest: +SKIP
        """
        
        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm)

        # Default cadence mask
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.flux), bool)
            
        from sklearn import linear_model

        # Use Scikit-learn ElasticNet
        self.regressor = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                fit_intercept=False)

        X = self.design_matrix_collection.values
        y = self.lc.flux

        # Set mask
        # note: ElasticNet has no internal way to do this so we have to just
        # remove the cadences from X and y
        X = X[cadence_mask,:]
        y = y[cadence_mask]

        self.regressor.fit(X, y)

        # Finishing work
        model_flux  = np.dot(X, self.regressor.coef_)
        model_err   = np.zeros(len(model_flux))
        
        self.coefficients = self.regressor.coef_
        
        self.model_lc = LightCurve(time=self.lc.time, flux=model_flux, flux_err=model_err)
        self.corrected_lc = self.lc.copy()
        self.corrected_lc.flux = self.lc.flux - self.model_lc.flux
        self.corrected_lc.flux_err = (self.lc.flux_err**2 + model_err**2)**0.5
        self.diagnostic_lightcurves = self._create_diagnostic_lightcurves()
        self.cadence_mask = cadence_mask
        self.alpha = alpha
            
        return self.denormalized_corrected_lc

    def correct_optimizer(self, cbv_type='SingleScale', cbv_indices=np.arange(1,9), 
            ext_dm=None, cadence_mask=None, alpha_bounds=[1e-4,1e4], 
            target_over_score=0.5, target_under_score=0.5, max_iter=100):
        """ Optimizes the correction by adjusting the L2-Norm (Ridge Regression)
        regularization penalty term, alpha, based on the introduced noise
        (over-fitting) and residual correlation (under-fitting) goodness
        metrics. The optmization is performed using the
        scipy.optimize.minimize_scalar Brent's method.

        The optimizer attempts to maximize the over- and under-fitting goodness
        metrics.  However, once the target_over_score or target_under_score is
        reached, a "Leaky ReLU" is used so that the optimization "pressure"
        concentrates on ther other metric so that both metrics rise above their
        respective target scores, instead of driving a single metric to near
        1.0.

        The optimization parameters used are stored in self.optimization_params
        as a record of how the optimization was performed.

        The optimized correction is performed using LightKurve's 
        RegressionCorrector methods. See correct_gaussian_prior for details.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use in correction {'ALL' => Use all}
        cbv_indices : list of lists
            List of CBV vectors to use in each of cbv_type passed. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        alpha_bounds : float list(len=2)
            upper anbd lowe bounds for alpha
        target_over_score : float
            Target Over-fitting metric score
        target_under_score : float
            Target under-fitting metric score
        max_iter : int
            Maximum number of iterations to optimize goodness metrics

        Returns
        -------
        `.LightCurve`
            Corrected light curve, with noise removed. In units of electrons / second

        Examples
        --------
        The following example will perform the correction using the
        SingleScale and Spike basis vectors. It will use an initial alpha value
        of 1.0. The target over-fitting score is 0.5 and the target
        under-fitting score is 0.8. 
            >>> cbv_type = ['SingleScale', 'Spike']
            >>> cbv_indices = [np.arange(1,9), 'ALL']
            >>> cbvCorrector.correct_optimizer(cbv_type=cbv_type, cbv_indices=cbv_indices,  # doctest: +SKIP
            >>> alpha_bounds=[1.0,1e3],  # doctest: +SKIP
            >>> target_over_score=0.5, target_under_score=0.5) # doctest: +SKIP
            >>> cbvCorrector.diagnose() # doctest: +SKIP
        """

        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm)

        # Create a dictionary for optimization parameters to easily pass to the
        # objective function, and also to save for posterity
        self.optimization_params = {'alpha_bounds': alpha_bounds,
                                    'target_over_score': target_over_score,
                                    'target_under_score': target_under_score,
                                    'max_iter': max_iter,
                                    'cadence_mask': cadence_mask,
                                    'over_metric_nSamples': 1}

        #***
        # Use scipy.optimize.minimize_scalar
        # Minimize the introduced metric
        from scipy.optimize import minimize_scalar
        minimize_result = minimize_scalar(self._goodness_metric_obj_fun, method='Bounded',
                bounds=alpha_bounds,
                options={'maxiter':max_iter, 'disp': False})

        # Re-fit with final alpha value
        # (scipy.optimize.minimize_scalar does not exit with the final fit!)
        self._goodness_metric_obj_fun(minimize_result.x)

        self.over_fitting_score = self.over_fitting_metric(nSamples=10)
        self.under_fitting_score = self.under_fitting_metric()
        self.alpha = minimize_result.x

        print('Optimized Over-fitting metric: {}'.format(self.over_fitting_score))
        print('Optimized Under-fitting metric: {}'.format(self.under_fitting_score))
        print('Optimized Alpha: {0:2.3e}'.format(self.alpha))

        return self.denormalized_corrected_lc


    def over_fitting_metric(self, nSamples=10):
        """ Uses a LombScarglePeriodogram to assess the change in broad-band
        power in a corrected light curve (in self.corrected_lc) to measure 
        degree of over-fitting.

        This function expects a zero-centered median normalized light curve.

        The to_periodogram Lomb-Scargle method is used and the sampling band is
        from one frequency separation to the Nyquist frequency

        This over-fitting goodness metric is callibrated such that a metric
        value 0.5 means the introduced noise due to over-fitting is at the level
        of the uncertainties in the light curve.

        Parameters
        ----------
        nSamples : int
            The number of times to compute and average the metric
            This can stabalize the value, defaut = 10

        Returns
        -------
        over-fitting_metric : float
            A float in the range [0,1] where 0 => Bad, 1 => Good
        """

        # Check if corrected_lc is present
        if (self.corrected_lc is None):
            log.warning('A corrected light curve does not exist, please run '
                    'correct first')
            return None

        # The fit can sometimes result in NaNs
        # Also ignore masked cadences
        orig_lc         = self.lc.copy()
        orig_lc         = orig_lc[self.cadence_mask]
        orig_lc         = orig_lc.remove_nans()
        corrected_lc    = self.corrected_lc.copy()
        corrected_lc    = corrected_lc[self.cadence_mask]
        corrected_lc    = corrected_lc.remove_nans()
        if (len(corrected_lc.flux) == 0):
            return 1.0

        # Perform the measurement multiple times and averge to stabalize the metric

        metric_per_iter = np.full_like(np.arange(nSamples), np.nan, dtype=np.double)
        for idx in np.arange(nSamples):

            pgOrig = orig_lc.to_periodogram()
            # Use the same periods in the corrected flux as just used in the
            # original flux
            pgCorrected = corrected_lc.to_periodogram(frequency=pgOrig.frequency)
            
            # Get an estimate of the PSD at the uncertainties limit
            # The raw and corrected uncertainties should be essentially identical so 
            # use the corrected
            # TODO: the periodogram of WGN should be analytical to compute!
            nNonGappedCadences = len(orig_lc.flux)
            meanCorrectedUncertainties = np.nanmean(corrected_lc.flux_err)
            WGNCorrectedUncert = (np.random.randn(nNonGappedCadences,1) * 
                    meanCorrectedUncertainties).T[0]
            model_err   = np.zeros(nNonGappedCadences)
            noise_lc = LightCurve(time=orig_lc.time, flux=WGNCorrectedUncert, flux_err=model_err)
            pgCorrectedUncert = noise_lc.to_periodogram()
            meanCorrectedUncertPower = np.nanmean(np.array(pgCorrectedUncert.power))
            
            # Compute the change in power
            pgChange = np.array(pgCorrected.power) - np.array(pgOrig.power)

            # ignore nans
            pgChange = pgChange[~np.isnan(pgChange)]
            
            # If no increase in power in ANY bands then return a perfect loss
            # function
            if (len(np.nonzero(pgChange>0.0)[0]) == 0):
                metric_per_iter[idx] = 0.0
                continue
            
            # We are only concerned with bands where the power increased so
            # when(pgCorrected - pgOrig) > 0 
            # Normalize by the noise in the uncertainty
            # We want the goodness to begin to degrade when the introduced 
            # noise is greater than the uncertainties.
            # So, when Sigmoid > 0.5 (given twiceSigmoidInv defn.)
            metric_per_iter[idx] = (np.sum(pgChange[pgChange>0.0]) / 
                    ((len(np.nonzero(pgChange>0.0)[0]))*meanCorrectedUncertPower))

        metric = np.mean(metric_per_iter)

        
        # We want the goodness to span (0,1]
        # Use twice a reversed sigmoid to get a [0,1] range from a [0,inf) range
        # We also want a goodness of 0.8 to mean the introduced noise is just
        # begining to increase beyond the noise of WGN of the mean of the uncertainties
        # 0.8 = sigmoidInv(0.4)
        # So, subtract 0.6 from the metric so that a metric value of 1.0 returns
        # a goodness of 0.8
        def sigmoidInv(x): return 2.0 / (1 + np.exp(x))
        # Make sure maximum score is 1.0
        metric = sigmoidInv(np.max([metric, 0.0]))

        return metric
    
    def under_fitting_metric(self, search_radius=None, min_targets=30,
            max_targets=50, clear_cache=False):
        """ This goodness metric measures the degree of under-fitting of the
        CBVs to the light curve. It does so by measuring the mean residual target to
        target Pearson correlation between the target under study and a selection of
        neighboring targets.

        This function will begin with the given search_radius in arcseconds and
        finds all neighboring targets. If not enough were found (< min_targets)
        the search_radius is increased until a minimum number are found.

        For TESS, the default search_radius is 5000 arcseconds.
        For Kepler/K2, the default search_radius is 1000 arcseconds

        The returned under-fitting goodness metric is callibrated such that a
        value fo 0.95 means the residual correlations in the target is
        equivalent to chance correlations of White Gaussian Noise.

        Parameters
        ----------
        search_radius : float
            Search Radius to find neighboring targets in arcseconds
        min_targets : float
            Minimum number of targets to use in correlation metric
            Using too few can cause unreliable results. Default = 30
        max_targets : float
            Maximum number of targets to use in correlation metric
            Using too many can slow down the metric due to large data
            download. Default = 50
        clear_cache : bool
            If true then force the method to re-find the neighboring targets.
            Otherwise, the neighboring targets are only found once the first
            time this function is called within the object.

        Returns
        -------
        under-fitting_metric : float
            A float in the range [0,1] where 0 => Bad, 1 => Good
        """

        # Check if corrected_lc is present
        if (self.corrected_lc is None):
            raise Exception('A corrected light curve does not exist, please run '
                    'correct first')
            return None

        # Set default search_radius if one is not provided.
        if (search_radius is None):
            if (self.lc.mission == 'TESS'):
                search_radius = 5000
            else:
                search_radius = 1000

        # Make a copy of search_radius because it changes locally
        dynamic_search_radius = search_radius
        # Max search radius is the diagonal distance along a CCD in arcseconds
        # 1 pixel in TESS is 21.09 arcseconds
        # 1 pixel in Kepler/K2 is 3.98 arcseconds
        if (self.lc.mission == 'TESS'):
            # 24 degrees of a TESS CCD array (2 CCD's wide) is 86,400 arcseconds
            max_search_radius = np.sqrt(2) * (86400/2.0)
        else:
            # One Kepler CCD spans 4,096 arcseconds
            max_search_radius = np.sqrt(2) * 4096

        # Retrieve SAP light curves in a neighborhood around the target
        # Only do this once, check if lc set is already cached
        if (self.lc_neighborhood is None or clear_cache):
            continue_searching = True
            while continue_searching:
                if (self.lc.mission == 'TESS'):
                    search_result = search_lightcurve(self.lc.label,
                        mission=self.lc.mission, sector=self.lc.sector,
                        radius=dynamic_search_radius, limit=max_targets)
                elif (self.lc.mission == 'Kepler'):
                    search_result = search_lightcurve(self.lc.label,
                        mission=self.lc.mission, quarter=self.lc.quarter,
                        radius=dynamic_search_radius, limit=max_targets)
                elif (self.lc.mission == 'K2'):
                    search_result = search_lightcurve(self.lc.label,
                        mission=self.lc.mission, campaign=self.lc.campaign,
                        radius=dynamic_search_radius, limit=max_targets)
                # Check if too few found
                if (len(search_result) < min_targets):
                    if (dynamic_search_radius > max_search_radius):
                        # hit the edge fo the CCD, we have to give up
                        raise Exception('Not enough neighboring targets were '
                            'found. under_fitting_metric failed')
                    # Too few found, increase search radius
                    dynamic_search_radius *= 1.5
                else:
                    continue_searching = False
                
            print('Found {} neighboring targets for under-fitting '
                    'metric'.format(len(search_result)))
            print('Downloading... this might take a while')
            lcfCol = search_result.download_all(flux_column='sap_flux')

            lc_neighborhood = []
            # Extract SAP light curves
            # We want zero-centered median normalized light curves
            for lc in lcfCol:
                lcSAP = lc.remove_nans().normalize()
                lcSAP.flux -= 1.0
                lc_neighborhood.append(lcSAP)
                # Align the neighboring target with the target under study
                lc_trim_mask = np.in1d(lc_neighborhood[-1].cadenceno, 
                        self.corrected_lc.cadenceno)
                lc_neighborhood[-1] = lc_neighborhood[-1][lc_trim_mask]

            self.lc_neighborhood = LightCurveCollection(lc_neighborhood)

            
            print('Neighboring targets ready for use')


        # Create fluxMatrix. The last entry is the target under study
        fluxMatrix = np.zeros((len(self.lc_neighborhood[0].flux),
            len(self.lc_neighborhood)+1))
        for idx in np.arange(len(fluxMatrix[0,:])-1):
            fluxMatrix[:,idx] = self.lc_neighborhood[idx].flux
        # Add in target under study
        fluxMatrix[:,-1] = self.corrected_lc.flux

        # Ignore masked cadences and NaNs
        mask = np.logical_and(self.cadence_mask,
                ~np.isnan(self.corrected_lc.flux))
        fluxMatrix = fluxMatrix[mask,:]

        # Determine the target-target correlation between target and
        # neighborhood
        correlationMatrix = compute_correlation(fluxMatrix)

        # The selection basis for targets used for the PDC-MAP SVD  uses median
        # absolute correlation per star.  However, here we wish to overemphasize
        # any residual correlation between a handfull of targets and not the
        # overall correlation (which should almost always be low).

        # We want a residual correlation larger than random correlations of WGN
        # to mean a meaningful correlation. The median Pearson correlation of
        # WGN of nCadences is approximated by the equation: 
        # 0.0010288 + 0.80304 nCadences^ -0.50128
        nCadences = len(fluxMatrix[:,0])
        beta = [0.0007, 0.8083, -0.5023]
        WGNCorrelation = (beta[0]+beta[1]*(nCadences**(beta[2])))

        # badLimit is the goodness value for WGN correlations
        # I.e. anything above this goodness value is equivalent to random correlations
        # I.e. 0.95 = sigmoidInv(WGNCorr * correlationScale)
        badLimit = 0.95
        correlationScale = 1 / (WGNCorrelation) * np.log((2.0 / badLimit) - 1.0)

        # Over-emphasize any individual correlation groups. Note the power of 
        # three after taking the absolute value
        # of the correlation. Also, the mean is used so that outliers are *not* ignored. 
        # Zero diagonal elements
        correlationMatrix = (np.tril(correlationMatrix, k=-1) + 
                                np.triu(correlationMatrix, k=+1))


        # Add up the correlation over all targets ignoring NaNs (no corrected fit)
        correlation = correlationScale*np.nanmean(np.abs(correlationMatrix)**3, axis=0)

        # We only want the last entry, which is for the target under study
        correlation = correlation[-1]

        def sigmoidInv(x): return 2.0 / (1 + np.exp(x))
        metric = sigmoidInv(correlation)

        return metric


    def _correct_initialization(self, cbv_type='SingleScale', cbv_indices='ALL',
            ext_dm=None):
        """ Performs all the preparatory work needed before applying a 'correct' method.

        This helper function is used so that multiple correct methods can be used
        without the need to repeat preparatory code.

        The main thing this method does is set up the design matrix, given the
        requested CBVs and external design matrix.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use
            Can be None if only ext_dm is used
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
            Can be None if only ext_dm is used
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to additionally be used in the fit

        """

        assert not ((cbv_type is None) ^ (cbv_indices is None)), \
                'Both cbv_type and cbv_indices must be None, or neither'

        if (cbv_type is None and cbv_indices is None):
            use_cbvs = False
        else:
            use_cbvs = True

        # If any DesignMatrix was passed then store it
        self.extra_design_matrix = ext_dm
        # Check that extra design matrix is aligned with lc flux
        if ext_dm is not None:
            assert isinstance(ext_dm, DesignMatrix), \
                'ext_dm must be a DesignMatrix'
            if (ext_dm.df.shape[0] != len(self.lc.flux)):
                    raise ValueError(
                        'ext_dm must contain the same number of cadences as lc.flux')
            

        # Create a CBV design matrix for each CBV set requested
        self.cbv_design_matrix = []

        if use_cbvs:
            assert (not isinstance(cbv_type, str) and 
                    not isinstance(cbv_indices[0], int)), \
                    'cbv_type and cbv_indices must be lists of strings'
            
            if (self.lc.mission in ['Kepler', 'K2']):
                assert len(cbv_type) == 1 , \
                    'cbv_type must be only Single-Scale for Kepler and K2 missions'
                assert cbv_type == ['SingleScale'], \
                    'cbv_type must be Single-Scale for Kepler and K2 missions'
            
            if (isinstance(cbv_type, list) and len(cbv_type) != 1):
                assert (self.lc.mission == 'TESS'), \
                    'Multiple CBV types are only allowed for TESS'
            
            assert (len(cbv_type) == len(cbv_indices)), \
                'cbv_type and cbv_indices must be the same list length'


            # Loop through all the stored CBVs and find the ones matching the
            # requested cbv_type list
            for idx in np.arange(len(cbv_type)): 
                for cbvs in self.cbvs:
            
                    # Temporarily copy the cbv_indices requested
                    cbv_idx_loop = cbv_indices[idx]
            
                    # If requesting 'ALL' CBVs then set to max default number
                    # Remember, cbv indices is 1-based!
                    if (isinstance(cbv_idx_loop, str) and (cbv_idx_loop == 'ALL')):
                        cbv_idx_loop=np.arange(1,MAX_NUMBER_CBVS+1)
                    # Trim to nCBVs in cbvs
                    nCBVs = len(cbvs.cbv_array)
                    cbv_idx_loop = cbv_idx_loop[cbv_idx_loop<=nCBVs]
            
                    if cbv_type[idx].find('MultiScale') >= 0:
                        # Find the correct band if this is a multi-scale CBV set
                        band = int(cbv_type[idx][-1])
                        if (cbvs.cbv_type in cbv_type[idx] and cbvs.band == band):
                            cbv_index_names = [cbv_index for cbv_index in
                                    cbv_idx_loop]
                            self.cbv_design_matrix.append(DesignMatrix(
                                cbvs._cbvs_to_matrix(cbv_idx_loop),
                                name=cbv_type[idx], columns=cbv_index_names))
                    else:
                        if (cbvs.cbv_type in cbv_type[idx]):
                            cbv_index_names = [cbv_index for cbv_index in
                                    cbv_idx_loop]
                            self.cbv_design_matrix.append(DesignMatrix(
                                cbvs._cbvs_to_matrix(cbv_idx_loop),
                                name=cbvs.cbv_type, columns=cbv_index_names))

        #***
        # Create the design matrix collection with CBVs, plus extra passed basis vectors

        # Create the full design matrix collection from all the sub-design
        # matrices (I.e 'flatten' the design matrix collection)
        if self.extra_design_matrix is not None and \
            self.cbv_design_matrix != []:
            # Combine cbv_design_matrix and extra_design_matrix 
            dm_to_flatten = [[cbv_dm for cbv_dm in self.cbv_design_matrix], 
                                [self.extra_design_matrix]]
            flattened_dm_list = [item for sublist in dm_to_flatten for item in sublist]
        elif self.cbv_design_matrix != []:
            # Just use cbv_design_matrix 
            dm_to_flatten = [[cbv_dm for cbv_dm in self.cbv_design_matrix]]
            flattened_dm_list = [item for sublist in dm_to_flatten for item in sublist]
        else:
            # Just use extra_design_matrix
            flattened_dm_list = [self.extra_design_matrix]

        self.design_matrix_collection = DesignMatrixCollection(flattened_dm_list)
        self.design_matrix_collection.validate()


    def _set_prior_width(self, sigma):
        """ Sets the Gaussian prior in the design_matrix_collection widths to sigma

        Parameters
        ----------
        sigma : scalar float 
            all widths are set to the same value
            If sigma = None then uniform sigma is set
        """

        if (isinstance(sigma, list)):
            raise Exception("Seperate widths is not yet implemented")

        for dm in self.design_matrix_collection:
            nCBVs = len(dm.prior_sigma)

            if sigma is None:
                dm.prior_sigma = np.ones(nCBVs) * np.inf
            else:
                dm.prior_sigma = np.ones(nCBVs) * sigma


    def _goodness_metric_obj_fun(self, alpha):
        """ The objective function to minimize with
        scipy.optimize.minimize_scalar

        First sets the alpha regularization penalty then runs
        RegressionCorrector.correct and then computes the over- and
        under-fitting goodness metrics to return a scalar penalty term to
        minimize.

        Uses the paramaters in self.optimization_params.

        Parameters
        ----------
        alpha : float
            regualrization penalty term value to set
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        target_over_score : float
            Target Over-fitting metric score
        target_under_score : float
            Target under-fitting metric score

        Returns
        -------
        penalty : float
            Penalty term for minimizer, based on goodness metrics

        """

        # Add in a width to the Gaussian priors
        # alpha = flux_sigma^2 / sigma^2
        sigma = np.median(self.lc.flux_err) / np.sqrt(np.abs(alpha))
        self._set_prior_width(sigma)                
        # Use RegressionCorrector.correct for the actual fitting
        super(CBVCorrector, self).correct(self.design_matrix_collection, 
            cadence_mask=self.optimization_params['cadence_mask'])

        overMetric = self.over_fitting_metric(
                nSamples=self.optimization_params['over_metric_nSamples'])
        underMetric = self.under_fitting_metric()

        # Once we hit the target we want to ease-back on increasing the metric
        # However, we don't want to ease-back to zero pressure, that will
        # unconstrain the penalty term and cause the optmizer to run wild.
        # So, use a "Leaky ReLU"
        # metric' = threshold + (metric - threshold) * leakFactor
        leakFactor = 0.01
        if (overMetric >= self.optimization_params['target_over_score']):
            overMetric = (self.optimization_params['target_over_score'] +
                leakFactor * 
                    (overMetric -
                        self.optimization_params['target_over_score']))

        if (underMetric >= self.optimization_params['target_under_score']):
            underMetric = (self.optimization_params['target_under_score'] +
                leakFactor * 
                    (underMetric -
                        self.optimization_params['target_under_score']))

        penalty = -(overMetric + underMetric)

        return penalty

    def diagnose(self):
        """ Returns diagnostic plots to assess the most recent correction.

        If a correction has not yet been fitted, a ``ValueError`` will be raised.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """

        axs = self._diagnostic_plot()

        plt.title('Alpha = {0:2.3e}'.format(self.alpha))
        
        return axs

    def goodness_metric_scan_plot(self, cbv_type=['SingleScale'],
            cbv_indices=[np.arange(1,9)], alpha_range_log10=[-4, 4],
            ext_dm=None, cadence_mask=None):
        """ Returns a diagnostic plot of the over and under goodness metrics as a
        function of the L2-Norm regularization term, alpha.

        alpha is scanned by default to the range 10^-4 : 10^4 in logspace

        cbvCorrector.correct_gaussian_prior is used to make the correction for
        each alpha. Then the over and under goodness metric are computed.

        If a correction has already been performed (via one of the correct_*
        methods) then the used alpha value is also plotted for reference.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use in correction {'ALL' => Use all}
        cbv_indices : list of lists
            List of CBV vectors to use in each of cbv_type passed. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        alpha_range_log10 : [list of two] The start and end exponent for the logspace scan.
            Default = [-4, 4]
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """

        alphaArray = np.logspace(alpha_range_log10[0], alpha_range_log10[1], num=100)

        # We need to make a copy of self so that the scan's final fit parameters
        # do not over-write any stored fit parameters
        cbvCorrectorCopy = self.copy()

        # Compute both metrics vs. alpha
        overMetric = []
        underMetric = []
        for thisAlpha in alphaArray:
            cbvCorrectorCopy.correct_gaussian_prior(cbv_type=cbv_type, cbv_indices=cbv_indices, 
                                        alpha=thisAlpha, ext_dm=None,
                                        cadence_mask=cadence_mask)
            overMetric.append(cbvCorrectorCopy.over_fitting_metric(nSamples=1))
            underMetric.append(cbvCorrectorCopy.under_fitting_metric())

        # plot both
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.semilogx(alphaArray, underMetric, 'b.', label='UnderFit')
        ax.semilogx(alphaArray, overMetric, 'r.', label='OverFit')

        if (isinstance(self.alpha, float)):
            ax.semilogx([self.alpha, self.alpha], [0, 1.0], 'k-', 
                label='corrected_lc Alpha = {0:2.3e}'.format(self.alpha))


        plt.title('Goodness Metrics vs. L2-Norm Penalty (alpha)')
        plt.xlabel('Regularation Factor Alpha')
        plt.ylabel('Goodness Metric')
        ax.grid(':', alpha=0.3)
        ax.legend()

        return ax

    def copy(self):
        """Returns a copy of this `cbvCorrector` object.

        This method uses Python's `copy.deepcopy` function to ensure that all
        objects stored within the cbvCorrector instance are fully copied.

        Returns
        -------
        cbvCorrector_copy : `cbvCorrector`
            A new object which is a copy of the original.
        """
        return copy.deepcopy(self)

    def __repr__(self):
       #return 'cbvCorrector (ID: {})'.format(self.lc.targetid)
        return print_dictionary(self)

#*******************************************************************************
# Cotrending Basis Vectors Classes and Functions 

class CotrendingBasisVectors:
    """
    Defines a CotrendingBasisVectors class, which is the Superclass for
    KeplerCotrendingBasisVectors and TessCotrendingBasisVectors
    Normally, one would use these latter classes instead of instantiating
    CotrendingBasisVectors directly.

    Stores Cotrending Basis Vectors for the Kepler/K2/TESS missions.

    Use search_cbvs to find the appropriate FITs files and KeplerCBVFile and TessCBVFile to retrieve the CBV FITS files.

    Each CotrendingBasisVectors objects contains only ONE set of CBVs. Instantiate multiple objects to store multiple set of
    CBVS, for example to save each of the three multi-scale bands.

    Attributes
    ----------
    mission         : [str] ('Kepler', 'K2', 'TESS')
    quarter         : only for Kepler mission
    campaign        : only for K2 mission
    sector          : only for TESS mission
    module,output   : only for Kepler/K2
    camera,ccd      : only for TESS
    cbv_type        : [str ('SingleScale', 'MultiScale', 'Spike')
    band            : [int] MultiScale band number (invalid for other CBV types)
    cbv_indices     : [int array] List of CBVs extracted for FITS file, {'ALL' => extract all}
    cbv_array       : [np.ndarray] The basis vectors
    cadenceno       : [int array-like] Cadence indices
    gap_indicators  : [bool array-like] True => cadence is gapped
    cbvEXTNAME      : [str] The EXTNAME extension in the header

    """

    #***
    validMissionOptions = ('Kepler', 'K2', 'TESS')
    validCBVTypes = ('SingleScale', 'MultiScale', 'Spike')

    #***

    def __init__(self, mission, cbv_type='SingleScale', band=None, cbv_indices='ALL'):

        # Check if a valid mission was passed
        try:
            self.validMissionOptions.index(mission)
            self.mission = mission
        except:
            raise ValueError('Invalid mission')

        # Check if a valid cbv_type was passed
        try:
            self.validCBVTypes.index(cbv_type)
            self.cbv_type = cbv_type
        except:
            raise ValueError('Invalid cbv_type')

        self.band = band

        # For Kepler/K2/TESS it's always been 16 CBVs max
        if (isinstance(cbv_indices, str) and (cbv_indices == 'ALL')):
            cbv_indices=np.arange(1,MAX_NUMBER_CBVS+1)
        self.cbv_indices = cbv_indices

    def _cbvs_to_matrix(self, cbv_indices='ALL'):
        """ Converts cbv_array (which is a list of np.ndarrays, one for each
        CBV, to a two-dimensional ndarray where the columns are the CBVs

        Parameters
        ----------
        cbv_indices : list int
            List of CBV vectors to use. 1-based indexing! {'ALL' => Use all}

        Returns
        -------
            cbvMatrix : numpy.ndarray
                2-dim array where each column is a CBV
        """
        if (isinstance(cbv_indices, str) and (cbv_indices == 'ALL')):
            cbv_indices=np.arange(1,MAX_NUMBER_CBVS+1)
        nCBVs = len(self.cbv_array)

        # Only keep CBV indices that are within range
        if (isinstance(cbv_indices, int)):
            # cbv_indices is just a single int
            if (cbv_indices > nCBVs):
                return []
        else:
            cbv_indices = [idx for idx in cbv_indices if idx <= nCBVs] 
    
        # Keep in mind that the CBVs are 1-based indexing
        # So, subtract 1 from indices!
        return np.array(self.cbv_array[(np.array(cbv_indices)-1).tolist()]).T
 
    def plot_cbvs(self, cbv_indices='ALL', ax=None):
        """Plot the requested CBVs

            Does not plot gapped cadences

        Parameters
        ----------
        cbv_indices  : list of ints
                        The list of cotrending basis vectors to plot. For example:
                            [1, 2] will fit the first two basis vectors. 'ALL' => plot all
                            NOTE: 1-base indexing
        ax          : matplotlib.pyplot.Axes.AxesSubplot
                        Matplotlib axis object. If `None`, one will be generated.

        Returns
        -------
        ax      : matplotlib.pyplot.Axes.AxesSubplot
                    Matplotlib axis object
        """
        with plt.style.context(MPLSTYLE):
            if (isinstance(cbv_indices, str) and (cbv_indices == 'ALL')):
                cbv_indices=np.arange(1,len(self.cbv_array)+1)
            cbvChosenLogicalArray = np.in1d(np.arange(1, len(self.cbv_array)+1), np.asarray(cbv_indices))

            if ax is None:
                _, ax = plt.subplots(1)

            for idx, cbv in enumerate(self.cbv_array[cbvChosenLogicalArray, :][:, :]):
                cbvIndex = cbv_indices[idx]
                # Do not plot gaps
                cbv[self.gap_indicators] = np.nan
                ax.plot(self.cadenceno, cbv-idx/10., label='{}'.format(cbvIndex))

            ax.set_yticks([])
            ax.set_xlabel('Cadence Number')

            if self.mission == 'Kepler':
                ax.set_title('Kepler CBVs (Quarter.Module.Output : {}.{}.{})'
                             ''.format(self.quarter, self.module, self.output),
                             fontdict={'fontsize': 10})
            elif self.mission == 'K2':
                ax.set_title('K2 CBVs (Campaign.Module.Output : {}.{}.{})'
                             ''.format( self.campaign, self.module, self.output),
                             fontdict={'fontsize': 10})
            elif self.mission == 'TESS':
                if (self.cbv_type == 'MultiScale'):
                    ax.set_title('TESS CBVs (Sector.Camera.CCD : {}.{}.{}, CBVType.Band : {}.{})'
                             ''.format(self.sector, self.camera, self.ccd, self.cbv_type, self.band),
                             fontdict={'fontsize': 9})
                else:
                    ax.set_title('TESS CBVs (Sector.Camera.CCD : {}.{}.{}, CBVType : {})'
                             ''.format(self.sector, self.camera, self.ccd, self.cbv_type),
                             fontdict={'fontsize': 10})

            ax.grid(':', alpha=0.3)
            ax.legend(fontsize='small', ncol=2)
        return ax

    def align(self, lc, trim_lc=False):
        """ Aligns the CBVs with a light curve. The lightCurve object might not 
        have the same cadences as the CBVs. This will trim the CBVs to be
        aligned with the light curve. 

        This method will preferentially use the cadence number (lc.cadenceno) to
        perform the synchronization, but will revert to using cadence time if
        cadenceno is not available in the light curve, which is more prone to
        errors

        It will report a warning and not synchronize if the light curve contains
        cadences not in the CBVs, unless trim_lc=True, in which case the light
        curve will also be trimmed.

        Parameters
        ----------
            lc : LightCurve object
                The reference light curve to align to
            trim_lc : [bool] If True then also trim the light curve, if needed

        Returns
        -------
            lc : LightCurve object
                If trim_lc = True then the returned light curve is also trimmed,

        """

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')


        if hasattr(lc, 'cadenceno'):

            lc_trim_mask = np.in1d(lc.cadenceno, self.cadenceno)
            if (np.any(np.logical_not(lc_trim_mask))):
                if (trim_lc):
                    # trim the light curve
                    lc = lc[lc_trim_mask]
                else:
                    log.warning('There are cadences in the light curve that are not in the CBVs. NO SYNCHRONIZATION OCCURED')


            trim_mask = np.in1d(self.cadenceno, lc.cadenceno)
            self.cbv_array      = self.cbv_array[:,trim_mask]
            self.cadenceno      = self.cadenceno[trim_mask]
            self.gap_indicators = self.gap_indicators[trim_mask]

        else:
            log.warning('Synchronization with cadence time stamps is not yet implemented. NO SYNCHRONIZATION OCCURED')

        return lc

    @staticmethod
    def _extract_cbvs_from_hdu_data(cbv_data, cbv_indices):
        """ STATIC method: Extracts the CBVs from the HDU[extName].data CBV data set

        Will remove all-zero CBVs.

        For internal use only

        Parameters
        ----------
        cbv_data : HDU extension data
                    The CBV data from the HDU extension
        cbv_indices : int-like
                    List of CBV indices to extract

        Returns
        -------
        cbv_array   : float array
                    The extracted CBVs in a ndarray list, all-zero CBVs removed
        cbv_indices : int array
                    List of CBV indices to extract all-zero CBVs removed
        """
        cbv_array = []
        indices_to_remove = []
        for idx, i in enumerate(cbv_indices):
            try:
                cbv = cbv_data.field('VECTOR_{}'.format(i))
                if (np.all(cbv == 0.0)):
                    # all-zero CBV, remove form list
                    raise Exception()
                cbv_array.append(cbv)
            except:
                # For CBV vectors that do not exist remove from cbv_indices list
                indices_to_remove.append(idx)
        cbv_indices = np.delete(cbv_indices, indices_to_remove)
        cbv_array = np.asarray(cbv_array)

        return cbv_array, cbv_indices
        
#***
class KeplerCotrendingBasisVectors(CotrendingBasisVectors):
    """ Sub-class for Kepler/K2 cotrending basis vectors

    See CotrendingBasisVectors for class details
    """

    #***
    validMissionOptions = ('Kepler', 'K2')
    validCBVTypes = ('SingleScale')

    #***

    def __init__(self, HDU, mission, module, output, cbv_indices='ALL'):
        """ Kepler/K2 CBVs are all in the same FITS file for each quarter/campaign, so, when intantiating the CBV object
        we must specify which module and output we desire. Only Single-Scale CBVs are stored for Kepler.

        Parameters
        ----------
        HDU : astropy.io.fits.hdu.hdulist.HDUList
            A pyfits opened FITS file containing the CBVs
        mission : str
            'Kepler' or 'K2'
        module : int
            Kepler CCD module 2 - 84
        output : int
            Kepler CCD output 1 - 4
        cbv_indices : int array
            List of CBVs extracted from FITS file, 1-Based {'ALL' => extract all}
        """

        super(KeplerCotrendingBasisVectors, self).__init__(mission, cbv_type='SingleScale', band=None, cbv_indices=cbv_indices)
        del mission, cbv_indices # Force use of object attributes

        if (self.mission == 'Kepler'):
            self.quarter = HDU['Primary'].header['QUARTER']
            self.campaign = None
        elif (self.mission == 'K2'):
            self.campaign = HDU['Primary'].header['CAMPAIGN']
            self.quarter = None

        self.module = module
        self.output = output

        extName = 'MODOUT_{0}_{1}'.format(module, output)
        cbv_data = HDU[extName].data

        self.cadenceno      = np.array(HDU[extName].data['CADENCENO'])
        self.gap_indicators = np.array(HDU[extName].data['GAPFLAG'])
        self.cbvEXTNAME     = HDU[extName].header['EXTNAME']

        # Pull out each individual CBV
        self.cbv_array, self.cbv_indices = self._extract_cbvs_from_hdu_data(cbv_data, self.cbv_indices)

    def __repr__(self):

        if self.mission == 'Kepler':
            repr_string = 'Kepler CBVs, Quarter.Module.Output : {}.{}.{}, nCBVs : {}'\
                ''.format(self.quarter, self.module, self.output)
        elif self.mission == 'K2':
            repr_string = 'K2 CBVs, Campaign.Module.Output : {}.{}.{}, nCBVs : {}'\
                ''.format( self.campaign, self.module, self.output)

        return repr_string
#***
class TessCotrendingBasisVectors(CotrendingBasisVectors):
    """ Sub-class for TESS cotrending basis vectors

    See CotrendingBasisVectors for class details
    """


    #***
    validMissionOptions = ('TESS')
    validCBVTypes = ('SingleScale', 'MultiScale', 'Spike')

    #***

    def __init__(self, HDU, cbv_type, band, cbv_indices='ALL'):
        """ TESS CBVs are in seperate FITS files for each camera.CCD, so camera.CCD is already specified in the
        TessCBVFile object, here we need to specify which CBV type and band is desired.

        If the requested CBV type does not exist in the HDU then None is returned

        Parameters
        ----------
        HDU : astropy.io.fits.hdu.hdulist.HDUList
            A pyfits opened FITS file containing the CBVs
        cbv_type : str
            'SingleScale', 'MultiScale' or 'Spike'
        band : int
            Band number for 'MultiScale' CBVs 
            Ignored for 'SingleScale' or 'Spike'
        cbv_indices : int array
            List of CBVs extracted from FITS file, 1-Based {'ALL' => extract all}
        """

        mission = HDU['PRIMARY'].header['TELESCOP']
        assert mission == 'TESS', 'This does not appear to be a TESS FITS HDU'

        super(TessCotrendingBasisVectors, self).__init__(mission, cbv_type=cbv_type, band=band, cbv_indices=cbv_indices)
        del mission, cbv_type, band, cbv_indices # Force use of object attributes

        self.sector = HDU['PRIMARY'].header['SECTOR']
        # Curiosly, camera and CCD are not in the primary header!
        self.camera = HDU[1].header['CAMERA']
        self.ccd = HDU[1].header['CCD']

        # Get the requested cbv_type
        switcher = {
            'SingleScale': 'CBV.single-scale.{}.{}'.format(self.camera, self.ccd),
            'MultiScale': 'CBV.multiscale-band-{}.{}.{}'.format(self.band,
                self.camera, self.ccd),
            'Spike': 'CBV.spike.{}.{}'.format(self.camera, self.ccd),
            'unknown': 'error'
            }
        extName = switcher.get(self.cbv_type, switcher['unknown'])
        if (extName == 'error'):
            raise Exception('Invalide cbv_type')


        try:
            cbv_data = HDU[extName].data

            self.cadenceno      = np.array(HDU[extName].data['CADENCENO'])
            self.gap_indicators = np.array(HDU[extName].data['GAP'])
            self.cbvEXTNAME     = HDU[extName].header['EXTNAME']
            
            # Pull out each individual CBV
            self.cbv_array, self.cbv_indices = self._extract_cbvs_from_hdu_data(
                    cbv_data, self.cbv_indices)
        except:
            self.cadenceno      = None
            self.gap_indicators = None
            self.cbvEXTNAME     = None
            self.cbv_array      = None
            self.cbv_indices    = None


    def __repr__(self):

        if (self.cbv_type == 'MultiScale'):
            repr_string = 'TESS CBVs, Sector.Camera.CCD : {}.{}.{}, CBVType.Band: {}.{}, nCBVs : {}' \
                ''.format(self.sector, self.camera, self.ccd, self.cbv_type, 
                    self.band, len(self.cbv_indices))
        else:
            repr_string = 'TESS CBVs, Sector.Camera.CCD : {}.{}.{}, CBVType : {}, nCBVS : {}'\
                ''.format(self.sector, self.camera, self.ccd, self.cbv_type, len(self.cbv_indices))

        return repr_string

#*******************************************************************************
# Functions

def get_kepler_cbvs (mission=('Kepler', 'K2'), quarter=None, campaign=None,
        channel=None, module=None, output=None, cbv_indices='ALL'):
    """ Searches the public data archive at MAST <https://archive.stsci.edu>
    for Kepler or K2 cotrending basis vectors.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    mission, quarter/campaign and channel or module/output, etc...
    and then extracts the requested basis vectors

    For Kepler/K2, the FITS files contain all channels in a single file per
    quarter/campaing.

    For Kepler extracts the DR25 CBVs

    Parameters
    ----------
    mission     : str, list of str
                    'Kepler' or 'K2'
    quarter or campaign : int, list of ints
                    Kepler Quarter or K2 Campaign.
    channel or module and output : int
                    Kepler/K2  requested channel or module and output
                    Must provide either channel, or module and outout, 
                    but not both
    cbv_indices : int array
                    List of CBVs extracted from FITS file, 1-Based {'ALL' => extract all}

    Returns
    -------
    result : :class:`KeplerCotrendingBasisVectors` object
        Object detailing the data products found.

    Examples
    --------
    This example will read in the CBVs for Kepler quarter 8,
    and then extract the first 8 CBVs for module.output 16.4

        >>> cbvs = get_kepler_cbvs(mission='Kepler', quarter=8, module=16, output=4,   # doctest: +SKIP
        >>>     cbv_indices=np.arange(1,9))                                     # doctest: +SKIP

    """

    #***
    # Validate inputs
    # Make sure only the appropriate arguments are passed
    if (mission == 'Kepler'):
        assert  isinstance(quarter, int), 'quarter must be passed for Kepler mission'
        assert  campaign is None, 'campaign must not be passed for Kepler mission'
    elif (mission == 'K2'):
        assert  isinstance(campaign, int), 'campaign must be passed for K2 mission'
        assert  quarter is None,  'quarter must not be passed for K2 mission'
    else:
        raise Exception('Unknown mission type')

    # CBV FITS files use module/output, not channel
    # So if channel is passed, convert to module/output
    if (isinstance(channel, int)):
        assert  module is None, 'module must NOT be passed if channel is passed'
        assert  output is None, 'output must NOT be passed if channel is passed'
        module, output = channel_to_module_output(channel)
        channel = None
    else:
        assert  module is not None, 'module must be passed'
        assert  output is not None, 'output must be passed'

    if (mission == 'Kepler'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/kepler/cbv/"
    elif (mission == 'K2'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/k2/cbv/"

    try:     
        soup = BeautifulSoup(requests.get(cbvBaseUrl).text, 'html.parser')
        cbv_files = [fn['href'] for fn in soup.find_all('a') if fn['href'].endswith('fits')]
 
        if mission == 'Kepler':
            quarter = 'q{:02}'.format(quarter)
            for cbv_file in cbv_files:
                if quarter + '-d25' in cbv_file:
                    break
        elif mission == 'K2':
            campaign = 'c{:02}'.format(campaign)
            for cbv_file in cbv_files:
                if campaign in cbv_file:
                    break

        kepler_cbv_url = cbvBaseUrl + cbv_file
        hdu = pyfits.open(kepler_cbv_url)

        cbvs = KeplerCotrendingBasisVectors(hdu, mission, module, output, 
                cbv_indices=cbv_indices)

        return cbvs

    except:
        raise Exception('CBVS were not found')

def get_tess_cbvs (sector=None, camera=None,
        CCD=None, cbv_type='SingleScale', band=None, cbv_indices='ALL'):

    """ Searches the `public data archive at MAST <https://archive.stsci.edu>`
    for TESS cotrending basis vectors.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    cotrending basis vectors.

    For TESS, each CCD CBVs are stored in a seperate FITS files.

    Parameters
    ----------
    sector : int, list of ints
                    TESS Sector number.
    camera and CCD : int, list of ints
                    TESS camera and CCD
    cbv_type    : str
                    'SingleScale' or 'MultiScale' or 'Spike'
    band        : int
                    Multi-scale band number
    cbv_indices : int array
                    List of CBVs extracted from FITS file, 1-Based {'ALL' => extract all}

    Returns
    -------
    result : :class:`TessCotrendingBasisVectors` object
        Object detailing the data products found.

    Examples
    --------
    This example will read in the CBVs for TESS Sector 10 Camera.CCD 2.4
    and then extract the first 6 CBVs of multi-scale band 2

        >>> cbvs = get_tess_cbvs(sector=10, camera=2, CCD=4, # doctest: +SKIP
        >>>     cbv_type='MultiScale', band=2, cbv_indices=np.arange(1,7)) # doctest: +SKIP
    """

    # The easiest way to obtain a link to the CBV file for a TESS Sector and
    # camera.CCD is
    # 
    # 1. Download the bulk download curl script (with a predictable url) for the
    # desired sector and search it for the camera.CCD I need 2. Download the CBV
    # FITS file based on the link in the curl script
    #
    # The bulk download curl links have urls such as:
    #
    # https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_17_cbv.sh
    #
    # Then the individual CBV files foudn in the curl file have urls such as:
    #
    # https://archive.stsci.edu/missions/tess/ffi/s0017/2019/279/1-1/tess2019279210107-s0017-1-1-0161-s_cbv.fits

    #***
    # Validate inputs
    # Make sure only the appropriate arguments are passed
    assert  isinstance(sector, int),    'sector must be passed for TESS mission'
    assert  isinstance(camera, int),    'camera must be passed'
    assert  isinstance(CCD, int),       'CCD must be passed'
    if cbv_type == 'MultiScale':
        assert  isinstance(band, int),  'band must be passed for multi-scale CBVs'
    else:
        assert  band is None,  'band must NOT be passed for single-scale or spike CBVs'
        
    curlBaseUrl = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_'
    curlEndUrl = '_cbv.sh'
    curlUrl = curlBaseUrl + str(sector) + curlEndUrl

    # This is the string to search for in the curl script file
    # Pad the sector number with a first '0' if less than 10
    # TODO: figure out a way to pad an interger number with forward zeros
    # without needing a conditional
    sector = int(sector)
    if (sector < 10):
        curlSearchString = 's000' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-'
    elif (sector >= 10 and sector < 100):
        curlSearchString = 's00' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-'
    elif (sector >= 100 and sector < 1000):
        curlSearchString = 's0' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-'
    elif (sector > 999):
        # TESS will be truly blessed if it gets to more than 999 sectors!
        raise Exception('Only up to 999 Sectors is currently supported')
    else:
        raise Exception('Error parsing sector string when getting TESS CBV FITS files')

    try: 

        # 1. Read in the relevent curl script file and find the line for the CBV data we are looking for
        data = urllib.request.urlopen(curlUrl)
        foundIndex = None
        for line in data:
            strLine = str(line)
            try:
                foundIndex = strLine.index(curlSearchString) # str.index will error when not found
                break
            except Exception:
                pass # continue searching
        if (foundIndex is None):
            raise Exception('CBV FITS file not found')

        # extract url from strLine
        htmlStartIndex = strLine.find('https:')
        htmlEndIndex = strLine.rfind('fits')
        tess_cbv_url  = strLine[htmlStartIndex:htmlEndIndex+4] # Add 4 for length of 'fits' string
        
        hdu = pyfits.open(tess_cbv_url)
            
        # Check that this is a TESS CBV FITS file
        mission = hdu['Primary'].header['TELESCOP']
        validate_method(mission, ['tess'])

        cbvs = TessCotrendingBasisVectors(hdu, cbv_type, band, cbv_indices=cbv_indices)

        return cbvs
            

    except:
        raise Exception('CBVS were not found')

        
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
# Helper Functions
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************

def compute_correlation(fluxMatrix):
    """  Finds the empirical target to target flux time series Pearson correlation.

    Parameters
    ----------
    fluxMatrix : float 2-d array[ntargets,ncadences]
        The matrix of target flux. There should be no gaps or Nans

    Returns
    -------
    correlation_matrix : [float 2-d array] (nTargets x nTargets)
        The target-target correlation
    """

    nCadences = len(fluxMatrix[:,0])

    # Scale each flux value by the RMS flux for the given target.
    rmsFlux = np.sqrt(np.sum(fluxMatrix**2.0, axis=0) / nCadences)
    unitNormFlux = fluxMatrix / np.tile(rmsFlux, (nCadences, 1))

    correlation_matrix = unitNormFlux.T.dot(unitNormFlux) / nCadences

    return correlation_matrix


