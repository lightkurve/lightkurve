"""Unit tests for the `CrowdingCorrector` class."""
import pytest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

import lightkurve as lk
from lightkurve.correctors import CrowdingCorrector, PLDCorrector

def test_crowdingcorrector_priors():
    """This test will check that the keywords are being pulled 
    from the headers correctly 
    """
    tpf = lk.search_targetpixelfile('WR21a', sector=36).download(quality_bitmask='hard')
    tpf_lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    pld = PLDCorrector(tpf,aperture_mask=tpf.pipeline_mask)
    pld_lc = pld.correct(pca_components=5, aperture_mask=tpf.pipeline_mask)
    median_flux = np.nanmedian(pld_lc.flux.value)

    #This just checks you have downloaded right data 
    assert_almost_equal(int(median_flux),int(7690.953289489277))

    CROWDSAP = tpf.hdu[1].header['CROWDSAP']
    FLFRCSAP = tpf.hdu[1].header['FLFRCSAP']

    cc = CrowdingCorrector(pld_lc)
    # Is the correct filetype returned?
    lc = cc.correct(crowdsap=CROWDSAP, flfrcsap=FLFRCSAP)
    
    val = np.nanmedian(lc.flux.value)
    val_true = 8574.747088383487
    
    assert_almost_equal(int(val),int(val_true))


