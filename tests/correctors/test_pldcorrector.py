import pytest

import matplotlib.pyplot as plt

from lightkurve import (
    search_targetpixelfile,
    search_tesscut,
    KeplerLightCurve,
    TessLightCurve,
    read,
)
from lightkurve.correctors import PLDCorrector

from astropy.io import fits
import numpy as np


@pytest.mark.remote_data
def test_kepler_pld_corrector():
    tpf = search_targetpixelfile("K2-199")[0].download()
    pld = PLDCorrector(tpf)
    # Is the correct filetype returned?
    clc = pld.correct()
    assert isinstance(clc, KeplerLightCurve)
    # Do the diagnostic plots run?
    pld.diagnose()
    plt.close()
    pld.diagnose_masks()
    plt.close()
    # Does sparse correction work?
    pld.correct(sparse=True)
    # Did the correction with default values help?
    raw_lc = tpf.to_lightcurve(aperture_mask="threshold")
    assert clc.estimate_cdpp() < raw_lc.estimate_cdpp()


@pytest.mark.remote_data
def test_tess_pld_corrector():
    tpf = search_targetpixelfile("TOI 700")[0].download()
    pld = PLDCorrector(tpf)
    # Is the correct filetype returned?
    clc = pld.correct()
    assert isinstance(clc, TessLightCurve)
    # Do the diagnostic plots run?
    pld.diagnose()
    plt.close()
    pld.diagnose_masks()
    plt.close()
    # Does sparse correction work?
    pld.correct(sparse=True)
    # Did the correction with default values help?
    raw_lc = tpf.to_lightcurve(aperture_mask="threshold")
    assert clc.estimate_cdpp() < raw_lc.estimate_cdpp()


@pytest.mark.remote_data
def test_pld_aperture_mask():
    """Test for #523: does PLDCorrector.correct() accept separate apertures for
    PLD pixels?"""
    tpf = search_targetpixelfile("K2-205")[0].download()
    # use only the pixels in the pipeline mask
    lc_pipeline = tpf.to_corrector("pld").correct(
        pld_aperture_mask="pipeline", restore_trend=False
    )
    # use all pixels in the tpf
    lc_all = tpf.to_corrector("pld").correct(
        pld_aperture_mask="all", restore_trend=False
    )
    # does this improve the correction?
    assert lc_all.estimate_cdpp() < lc_pipeline.estimate_cdpp()


@pytest.mark.remote_data
def test_pld_corrector():
    # download tpf data for a target
    k2_target = "EPIC247887989"
    k2_tpf = search_targetpixelfile(k2_target).download()
    # instantiate PLD corrector object
    pld = PLDCorrector(k2_tpf[:500], aperture_mask="threshold")
    # produce a PLD-corrected light curve with a default aperture mask
    corrected_lc = pld.correct()
    # ensure the CDPP was reduced by the corrector
    pld_cdpp = corrected_lc.estimate_cdpp()
    raw_cdpp = k2_tpf.to_lightcurve().estimate_cdpp()
    assert pld_cdpp < raw_cdpp
    # make sure the returned object is the correct type (`KeplerLightCurve`)
    assert isinstance(corrected_lc, KeplerLightCurve)
    # try detrending using a threshold mask
    corrected_lc = pld.correct()
    # reduce using fewer principle components
    corrected_lc = pld.correct(pca_components=20)
    # try PLD on a TESS observation
    from lightkurve import TessTargetPixelFile
    from ..test_targetpixelfile import TESS_SIM

    tess_tpf = TessTargetPixelFile(TESS_SIM)
    # instantiate PLD corrector object
    pld = PLDCorrector(tess_tpf[:500], aperture_mask="pipeline")
    # produce a PLD-corrected light curve with a pipeline aperture mask
    raw_lc = tess_tpf.to_lightcurve(aperture_mask="pipeline")
    corrected_lc = pld.correct(pca_components=20)
    # the corrected light curve should have higher precision
    assert corrected_lc.estimate_cdpp() < raw_lc.estimate_cdpp()
    # make sure the returned object is the correct type (`TessLightCurve`)
    assert isinstance(corrected_lc, TessLightCurve)


@pytest.mark.remote_data
def test_tpf_with_zero_flux_cadence():
    """Regression test for #873."""
    tpf = search_tesscut("TIC 123835353", sector=6).download(cutout_size=5)
    tpf.to_corrector("pld").correct()



@pytest.mark.remote_data
def test_tpf_with_allflux_err_NaN():

    #This test shows that a TPF with all flux_err values equal to NaN fails
    #to be put into PLDCorrector - throwing a ValueError 

    #Download a TPF
    search_tpf = search_targetpixelfile("KT Eri", sector=32)
    tpf = search_tpf[0].download()
    #Get the TPF file path
    fname = tpf.path

    #Open the TPF fits file and populate the flux_err with NaN values
    with fits.open(fname, mode='update') as hdulist:
        a = np.empty((112388,11,11))
        a[:] = np.nan
        hdulist[1].data['FLUX_ERR'] = a
    hdulist.close()

    #Read in the fits file as a TPF object
    tpf = read(fname)

    #Create and apply the PLDCorrector - this should fail as the mask gets rid of all values.
    with pytest.raises(ValueError):
        PLDCorrector(tpf).correct()

@pytest.mark.remote_data
def test_tpf_with_someflux_err_NaN():

    #This test shows that a TPF with some flux_err values equal to NaN 
    #can be put into PLDCorrector 

    #Download a TPF from MAST
    fits_file = "https://archive.stsci.edu/missions/tess/tid/s0026/0000/0001/5832/4245/tess2020160202036-s0026-0000000158324245-0188-s_tp.fits"
    #Read fits file 
    tpf = read(fits_file)
    #The apply the PLDCorrector - this should pass
    #The PLDCorrector should mask out the bad entries in flux_err and set them as NaN
    #The code should then progress as ususal
    pld = tpf.to_corrector('pld')
    
    
    
