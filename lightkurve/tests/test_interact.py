from __future__ import division, print_function

import os
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tempfile
from ..interact import map_cadences, prepare_lightcurve_datasource
from ..targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from ..utils import KeplerQualityFlags


filename_tpf_all_zeros = get_pkg_data_filename("data/test-tpf-all-zeros.fits")
filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")
TABBY_TPF = ("https://archive.stsci.edu/missions/kepler/target_pixel_files"
             "/0084/008462852/kplr008462852-2011073133259_lpd-targ.fits.gz")
TESS_SIM = ("https://archive.stsci.edu/missions/tess/ete-6/tid/00/000"
            "/004/176/tess2019128220341-0000000417699452-0016-s_tp.fits")

@pytest.mark.remote_data
def test_malformed_notebook_url():
    '''Test if malformed notebook_urls raise proper exceptions.'''
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    with pytest.raises(ValueError) as exc:
        tpf.interact(notebook_url='')
    assert('Empty host value' in exc.value.args[0])
    with pytest.raises(AttributeError) as exc:
        tpf.interact(notebook_url=None)
    assert('object has no attribute' in exc.value.args[0])


@pytest.mark.remote_data
def test_graceful_exit_outside_notebook():
    '''Test if running interact outside of a notebook fails gracefully'''
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    result = tpf.interact()
    assert(result is None)


def test_cadence_mapping():
    """Test if the cadence mapping works"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_all_zeros),
                TessTargetPixelFile(filename_tpf_all_zeros)]:
        cadence_dict = map_cadences(tpf)
        assert cadence_dict[tpf.cadenceno[0]] == 0
        extra_cadences = set(cadence_dict.keys()) - set(tpf.cadenceno)
        assert extra_cadences == set()
