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
	def __init__(self, frequency=None, model=LombScargle, 
				power=None, lc=None):

		self.lightcurve = lc

		nyquist_frequency = 0.5 * (1./((np.median(self.lightcurve.time[1:] - self.lightcurve.time[0:-1])*u.day).to(u.second))).to(u.microhertz).value
		frequency = np.linspace(1, nyquist_frequency, len(self.lightcurve.time)//2) * u.microhertz
		self.frequency = frequency

		try:
			self.model = model((self.lightcurve.time*u.day).to(u.second), self.lightcurve.flux*1e6)
		except:
			raise AttributeError("No Ligtcurve given")

		self.power = power

	def from_lightcurve(lc):
		#model = model((lc.time*u.day).to(u.second), lc.flux*1e6)
		return Periodogram(lc=lc)

	def get_period(self):
		#TODO: implement me
		pass

	def generate_power(self, frequency):
		uHz_conv = 1./(((1./u.day).to(u.microhertz)))
		self.power = self.model.power(frequency, method="fast", normalization="psd")
		self.power *= uHz_conv / len(self.lightcurve.time)  # Convert to ppm^2/uHz

	def generate_freq(self, frequency):
		if frequency is None:
			nyquist_frequency = 0.5 * (1./((np.median(self.lightcurve.time[1:] - self.lightcurve.time[0:-1])*u.day).to(u.second))).to(u.microhertz).value
			frequency = np.linspace(1, nyquist_frequency, len(self.lightcurve.time)//2) * u.microhertz
		else:
			frequency = np.asarray(frequency) #Load as a numpy array
			if type(frequency) != u.quantity.Quantity: #Has no astropy units
				frequency = frequency * u.microhertz
			else:
				if frequency.unit != "uHz": #try to convert from DAYS to MICROHERTZ
					frequency *= 1./u.day
					frequency *= 1./uHz_conv

		return frequency

	def plot(self, frequency=None, scale="linear", ax=None, numax=None, **kwargs):
		if ax is None:
			fig, ax = plt.subplots()

		frequency = self.generate_freq(frequency)
		self.generate_power(frequency)

		ax.plot(frequency, self.power, **kwargs)
		ax.set_xlabel("frequency [$\mu$Hz]")
		ax.set_ylabel("power [ppm$^2$/$\mu$Hz]");
		try:
			ax.set_title('KIC {}'.format(self.lightcurve.keplerid))
		except AttributeError:
			print("No KIC was found")
		
		if numax:
			ax.fill_between([numax.value*0.8, numax.value*1.2], self.power.value.min(), self.power.value.max(), alpha=0.2, color='C3', zorder=10)
	
		if scale == "log":
			ax.set_yscale('log')
			ax.set_xscale('log')

		return ax

	def _normalize_power(self):
		bkg = self.estimate_background(self.frequency.value, self.power.value)
		df = self.frequency[1].value - self.frequency[0].value
		norm_p = self.power.value / bkg
		return bkg, df, norm_p

	def smooth_ps(self, normalized_power, df, filter=gaussian_filter):
		return filter(normalized_power, 10 / df)

	def find_nu_max(self):
		_, df, normalized_power = self._normalize_power()
		smoothed_power_spectrum = self.smooth_ps(normalized_power, df)
		peak_frequencies = self.frequency[self.find_peaks(smoothed_power_spectrum)].value
		self.nu_max = peak_frequencies[peak_frequencies > 5][0]
		return self.nu_max

	def estimate_background(self, x, y, log_width=.01):
		count = np.zeros(len(x), dtype=int)
		bkg = np.zeros_like(x)
		x0 = np.log10(x[0])
		while x0 < np.log10(x[-1]):
			m = np.abs(np.log10(x) - x0) < log_width
			bkg[m] += np.median(y[m])
			count[m] += 1
			x0 += 0.5 * log_width
		return bkg / count 

	def find_peaks(self, z):
		peak_inds = (z[1:-1] > z[:-2]) * (z[1:-1] > z[2:])
		peak_inds = np.arange(1, len(z)-1)[peak_inds]
		peak_inds = peak_inds[np.argsort(z[peak_inds])][::-1]
		return peak_inds

