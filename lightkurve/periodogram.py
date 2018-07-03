import numpy as np
from astropy import units as u
from astropy.stats import LombScargle
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

__all__ = ['Periodogram']

class Periodogram(object):
	"""
	Implements a simple class for a generic periodogram
	"""
	def __init__(self, delta_nu=None, nu_max=None, period=None,
				frequency=None, model=LombScargle, power=None, 
				t=None):
		self.delta_nu = delta_nu
		self.nu_max = nu_max
		self.period = period
		self.t = t
		self.frequency = frequency
		self.model = model
		self.power = power

	def from_lightcurve(lc, model=LombScargle, normalization="psd"):
		m = (lc.quality == 0)
		m &= np.isfinite(lc.time)
		m &= np.isfinite(lc.flux)
		t = lc.time[m]
		y = 1e6 * (lc.flux[m] - 1.0)
		model = model(t, y)

		return Periodogram(model=model, t=t)
	
	def get_power(self, frequency=None, normalization="psd", method="fast"):
		uHz = u.Hz * 1e-6
		if frequency is None:
			#Create some 'reasonable' default for them.
			self.frequency = np.linspace(1, 300, 100000) * uHz
		else:
			#TODO: check units for frequency
			self.frequency = frequency

		self.power = self.model.power(self.frequency.value, method=method, normalization=normalization)
		self.power *= (uHz / len(self.t)).value

	def get_period(self):
		#TODO: implement me
		pass

	def plot(self, ax=None, **kwargs):
		if ax is None:
			_, ax = plt.subplots(1)
		ax.semilogy(self.frequency.value, self.power, "k")
		ax.set_xlim(self.frequency[0].value, self.frequency[-1].value)
		ax.set_xlabel("frequency [$\mu$Hz]")
		ax.set_ylabel("power [ppm$^2$/$\mu$Hz")

		return ax


