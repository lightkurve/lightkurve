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
				lc=None):
		self.delta_nu = delta_nu
		self.nu_max = nu_max
		self.period = period
		self.lightcurve = lc
		self.frequency = frequency
		self.model = model
		self.power = power
		

	def from_lightcurve(lc, model=LombScargle, normalization="psd"):
		model = model((lc.time*u.day).to(u.second), lc.flux*1e6)
		return Periodogram(model=model, lc=lc)

	def get_period(self):
		#TODO: implement me
		pass

	def plot_frequency(self, frequency=None, scale="linear", ax=None, numax=None, **kwargs):
		if ax is None:
			fig, ax = plt.subplots()

		if frequency:
			self.frequency = np.asarray(frequency) #Load as a numpy array
			if type(frequency) != u.quantity.Quantity: #Has no astropy units
				self.frequency = frequency * u.microhertz
			else:
				frequency *= 1./(((1./u.day).to(u.microhertz)))
				self.frequency = frequency

		else: #we need to create frequency for them based off lightcurve
			nyquist_frequency = 0.5 * (1./((np.median(self.lightcurve.time[1:] - self.lightcurve.time[0:-1])*u.day).to(u.second))).to(u.microhertz).value
			self.frequency = np.linspace(1, nyquist_frequency, len(self.lightcurve.time)//2) * u.microhertz

		uHz_conv = 1./(((1./u.day).to(u.microhertz)))
		self.power = self.model.power(self.frequency, method="fast", normalization="psd")
		self.power *= uHz_conv / len(self.lightcurve.time)  # Convert to ppm^2/uHz

		ax.plot(self.frequency, self.power, **kwargs)
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

	def normalized_plot(self, ax=None, **kwargs):
		if ax is None:
			_, ax = plt.subplots(1)
		bkg, df, norm_power = self.normalize_power()
		smoothed_ps = self.smooth_ps(norm_power, df)

		ax.semilogy(self.frequency.value, norm_power, "k")
		ax.plot(self.frequency.value, smoothed_ps, label="smoothed", color="C1")
		ax.set_xlim(self.frequency[0].value, self.frequency[-1].value)
		
		#TODO: Create some 'reasonable' ylim for axis
		ax.set_ylim(1e-1, 3e2)

		ax.legend()
		ax.set_xlabel("frequency [$\mu$Hz]")
		ax.set_ylabel("normalized power")

		return ax

	def normalize_power(self):
		bkg = self.estimate_background(self.frequency.value, self.power)
		df = self.frequency[1].value - self.frequency[0].value
		normalized_power = self.power / bkg
		return bkg, df, normalized_power

	def smooth_ps(self, normalized_power, df, filter=gaussian_filter):
		return filter(normalized_power, 10 / df)

	def find_nu_max(self):
		_, df, normalized_power = self.normalize_power()
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

