"""Tools to interact more easily with Full Frame Image files."""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

from .utils import plot_image



__all__ = ['FullFrameImage']

_CHANNEL_LIST = np.arange(1, 85)
_MODULE_LIST = np.asarray(np.array_split(_CHANNEL_LIST, 21)).reshape((21, 4))


class FullFrameImage(object):

    def __init__(self, path_or_url, **kwargs):
        self.path = path_or_url
        if isinstance(path_or_url, fits.HDUList):
            self.hdulist = path_or_url
        else:
            self.hdulist = fits.open(self.path, **kwargs)

    def skycoord_to_pixel(self, skycoord):
        """Converts an (ra, dec) sky coordinate to a (channel, column, row) tuple."""
        for channel in range(1, 85):
            hdr = self.hdulist[channel].header
            if 'CTYPE1' not in hdr:
                continue  # dead module
            wcs = WCS(hdr)
            (col, row) = wcs.all_world2pix([skycoord.ra.deg], [skycoord.dec.deg], 0)
            col, row = col[0], row[0]
            # Dide the coordinate gall within the channel footprint?
            if (col > 0) and (col < hdr['NAXIS1']) and (row > 0) and (row < hdr['NAXIS2']):
                return channel, col, row
        return None, None, None

    def plot_cutout(self, skycoord, size=10, **kwargs):
        """Plots a small part of an FFI around a sky coordinate."""
        channel, col, row = self.skycoord_to_pixel(skycoord)
        extent = (int(col-size/2), int(col+size/2),
                  int(row-size/2), int(row+size/2))
        img = self.hdulist[channel].data[extent[2]: extent[3],
                                         extent[0]: extent[1]]
        plot_image(img, extent=extent, **kwargs)


    def plot_module(self, skycoord, size=10, **kwargs):
        """Plots a Module"""
        vmin = kwargs.pop('vmin', 1)
        vmax = kwargs.pop('vmax', 1e5)

        channel, col, row = self.skycoord_to_pixel(skycoord)
        group = _MODULE_LIST[np.asarray([channel in m for m in _MODULE_LIST])][0]
        fig, axs = plt.subplots(1, 4, figsize=(16, 3.))
        for idx, channel in enumerate(group):
            extent = (int(col-size/2), int(col+size/2),
                      int(row-size/2), int(row+size/2))
            img = self.hdulist[channel].data[extent[2]: extent[3],
                                             extent[0]: extent[1]]
            if idx < 3:
                ax = plot_image(img, extent=extent, vmax=vmax, vmin=vmin, **kwargs,
                        ax=axs[idx], show_colorbar=False)
            else:
                ax = plot_image(img, extent=extent, vmax=vmax, vmin=vmin, **kwargs,
                        ax=axs[idx], show_colorbar=True)
            if idx > 0:
                ax.set_ylabel('')
            ax.set_title('Channel : {}'.format(channel))
        return fig


    def plot(self):
        """Plot all channels on a grid."""
        pass
