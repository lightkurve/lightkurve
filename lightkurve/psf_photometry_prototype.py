from oktopus import GaussianPrior


class Star(object):
    """Holds the information on a star being fitted during PSF photometry."""
    def __init__(self, targetid, col, err_col, row, err_row, flux, err_flux):
        self.targetid = targetid
        self.col = col
        self.err_col = err_col
        self.row = self.row
        self.err_row = self.err_row
        self.flux = flux
        self.err_flux = err_flux
        self.col_prior = GaussianPrior(mean=self.col, var=self.err_col**2)
        self.row_prior = GaussianPrior(mean=self.row, var=self.err_row**2)
        self.flux_prior = GaussianPrior(mean=self.flux, var=self.err_flux**2)

    def evaluate(self, col, row, flux):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col."""
        logp = (self.col_prior.evaluate(col) +
                self.row_prior.evaluate(row) +
                self.flux_prior.evaluate(flux))
        return logp

    def plot(self):
        raise NotImplementedError('Geert was lazy.')


class Background():
    """
    mean : 2D image or float
    """
    def __init__(self, mean, sigma):
        pass


class Focus():
    pass


class Motion():
    pass
    # delta_col, delta_row = UniformPDF(), UniformPDF()


class SceneModelParameters():
    """Parameters that define a single cadence of a TPF image.

    Attributes
    ----------
    stars : list of `Star` objects
        Stars in the scene.
    """
    def __init__(self, stars, background=None, focus=None, motion=None, meta=None):
        self.stars = stars
        self.meta = meta  # intended to hold scipy fitting diagnostics (aka garbage)


class SceneModel():
    """A model which describes a single Kepler image.

    Attributes
    ----------
    stars : list of `Star` objects
    """
    def __init__(self, stars, background=None, focus=None, motion=None):
        self.params = SceneModelParameters(stars, background, focus, motion)

    def predict(self, params):
        """Produces a synthetic Kepler 2D image given a set of scene parameters."""
        # put a star at position col + delta_col
        return synthetic_image

    def fit(self, observed_data):
        self.fitted_params = self.params
        score = self.predict(self.fitted_params) - observed_data
        whatevz = optimize(score)
        self.fitted_params.meta['scipy'] = whatevz
        return self.fitted_params


class KeplerTargetPixelFile():

    def to_lightcurve(method='aperture'):
        ...

    def aperture_photometry(self):
        ...

    def get_scenemodel(self):
        return ...

    def psf_photometry(self, sourcelist, return_diagnostics=False):
        star_priors = tpf.get_starpriors(sourcelist)
        BackgroudPrior(self.data.median(), self.data.std())
        scenemodel = self.get_scenemodel(star_priors)
        for cadence in tpf.data:
            result = scenemodel.fit(tpf.data)
            results.append(result)
        for idx in range(len(r.stars)):
            LightCurve(flux=[r.stars[0].flux for r in results])
        return LightCurveCollection([lc1, lc2, lc3])


if __name__ == '__main__':
    tpf = KeplerTargetPixelFile()
    tpf.psf_photometry(sourcelist='kic')
