import lightkurve as lk
import numpy as np
import astropy
import scipy
import astroquery

def test_show_environment(capsys):
    """Show the environment the test suite is running. """
    with capsys.disabled():
        print("Dependencies used: ")
        print("lightkurve", lk.__version__)
        print("astropy", astropy.__version__)
        print("numpy", np.__version__)
        print("scipy", scipy.__version__)
        print("astroquery", astroquery.__version__)
