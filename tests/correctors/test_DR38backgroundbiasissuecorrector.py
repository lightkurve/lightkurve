"""Unit tests for the `DR38BackgroundBiasIssueCorrector` class."""
import pytest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from scipy import ndimage

import lightkurve as lk
from lightkurve.correctors import DR38BackgroundBiasIssueCorrector

def test_DR38BackgroundBiasIssueCorrector_priors():
    """This test will check that the transit depth is being calculated
    corectly
    """

    #Getting information to compare against
    tpf = lk.search_targetpixelfile('TOI-824', sector=11).download()
    lcf = lk.search_lightcurve('TOI-824', author='SPOC', sector=11).download()

    #Exclude pixels in optimal aperture + a 1 pixel halo around it.
    inBackgroundAperture = ~tpf.pipeline_mask
    halo = ndimage.binary_dilation(tpf.pipeline_mask, iterations=1)
    inBackgroundAperture2 = inBackgroundAperture * ~halo

     #Remove any pixels from the background that might be saturated + a 2 pixel halo around them
    svalue = 200000
    tp = np.where(tpf.flux.value >=svalue)

    tp1 = tp[1] #x-axis
    tp2 = tp[2] #y-axis

    sat_mask = np.zeros((tpf.shape[1:]), dtype='bool')
    sat_mask[tp1,tp2] = True

    aper_new2 = ndimage.binary_dilation(sat_mask, iterations=2)
    
    inBackgroundAperture3 = inBackgroundAperture2 * ~aper_new2

    #Get indices of where array is true in this background mask
    bkg = np.where(inBackgroundAperture3==True)
    bkg2 = [bkg[0],bkg[1]]

    #For each pixel above get the median value of its time series - but only good quality data
    medval = []
    
    for a in range(tpf.shape[1]):
        custom_mask = np.zeros((tpf.shape[1:]), dtype='bool')
        custom_mask[bkg2[0][a],bkg2[1][a]] = True
        lc_new = tpf[tpf.quality == 0].to_lightcurve(aperture_mask=custom_mask)
        medval.append(np.nanmedian(lc_new.flux.value))

    #Now sort the data and get indicies
    sorted_index = np.argsort(medval)

    #Grab the 3rd lowest value
    vals2 = sorted_index[2]
    puse = medval[vals2]

    bgBias_true = np.abs(puse)
    
    #This is the corrector
    bg = DR38BackgroundBiasIssueCorrector(tpf, lcf)
    
    corrected_lc = bg.correct()

    bgBias = corrected_lc.bgBias
    pdcCorrection = corrected_lc.pdcCorrection
    transitBias = corrected_lc.transitBias

    lcm = np.nanmedian(corrected_lc.flux.value)
    
    assert_almost_equal(bgBias_true,bgBias)

    #Get the number of pixels in the SPOC optimal aperture
    nPix = np.sum(tpf.pipeline_mask)

    #Get the CROWDSAP function from the TPF
    CROWDSAP = tpf.hdu[1].header['CROWDSAP']

    #Get the FLFRCSAP function from the TPF
    FLFRCSAP = tpf.hdu[1].header['FLFRCSAP']

    #Get the median flux from the input PDCSAP_FLUX light curve
    medFluxPdc = np.nanmedian(lcf[lcf.quality == 0].flux.value)

    #Then calculate the pdcCorrection
    pdcCorrection_true = bgBias_true*nPix*CROWDSAP/FLFRCSAP

    assert_almost_equal(pdcCorrection_true,pdcCorrection)

    #Correct the flux and make a new PDCSAP_FLUX light curve
    flux_corr = lcf.flux.value + pdcCorrection_true
    lc_corr_true = lk.LightCurve(time=lcf.time.value, flux=flux_corr, flux_err=lcf.flux_err.value)

    lcm_true = np.nanmedian(lc_corr_true.flux.value)

    #Calculate the transitBias
    transitBias_true = (pdcCorrection_true/medFluxPdc)*100
    

    assert_almost_equal(transitBias_true,transitBias)

    assert_almost_equal(lcm_true,lcm)
