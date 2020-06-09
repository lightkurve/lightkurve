import pytest

from astropy.utils.data import get_pkg_data_filename
import numpy as np

from ... import read
from .. import TessPLDCorrector


TESS_DATA = ("https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:"
            "TESS/product/tess2018319095959-s0005-0000000238201762-0125-s_tp.fits")


@pytest.mark.remote_data
def test_basics():
    tpf = read(TESS_DATA)
    tpf = tpf[np.nansum(tpf.flux[:, tpf.pipeline_mask], axis=(1)) != 0]
    tpf = tpf[np.nansum(tpf.flux_err[:, tpf.pipeline_mask], axis=(1)) != 0]
    corrector = TessPLDCorrector(tpf)
    corrector.correct(spline_n_knots=10, spline_degree=3, pixel_components=1)
    corrector.diagnose()
    corrector.correct(spline_n_knots=10, spline_degree=3, pixel_components=1, sparse=True)
