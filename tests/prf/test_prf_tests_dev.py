import lightkurve as lk
from astropy.utils.data import get_pkg_data_filename

# a local version of TABBY_TPF with ~ 2 days of data; should be sufficient for most tests
filename_tpf_tabby_lite = get_pkg_data_filename("../data/test-tpf-kplr-tabby-100-cadences.fits")


def test_tpf_has_prfattribute():
	tpf = lk.KeplerTargetPixelFile(filename_tpf_tabby_lite)
	assert hasattr(tpf, 'prf')
	print(type(tpf.prf))
	assert isinstance(tpf.prf, lk.prf.prfmodel.PRF)