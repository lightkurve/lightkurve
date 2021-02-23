from __future__ import division, print_function

import numpy as np

from .lightcurve import LightCurve


__all__ = ["estimate_cdpp"]


def estimate_cdpp(flux, **kwargs):
    """A convenience function which wraps LightCurve.estimate_cdpp().

    For details on the algorithm used to compute the Combined Differential
    Photometric Precision (CDPP) noise metric, please see the docstring of
    the `LightCurve.estimate_cdpp()` method.

    Parameters
    ----------
    flux : array-like
        Flux values.
    **kwargs : dict
        Dictionary of arguments to be passed to `LightCurve.estimate_cdpp()`.

    Returns
    -------
    cdpp : float
        Savitzky-Golay CDPP noise metric in units parts-per-million (ppm).
    """
    return LightCurve(time=np.arange(len(flux)), flux=flux).estimate_cdpp(**kwargs)
