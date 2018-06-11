class PDF():
    """Generic distribution class"""
    def __init__():

    def mean():

    def std():

    def plot():


class Star():
    id
    col = DeltaPDF()
    row = DeltaPDF()
    flux = SampledPDF()
    
    def evaluate(ra, dec, flux=None):
        """Evaluate the prior prob."""
        return prob


class Background():
    """
    mean : 2D image or float
    """
    def __init__(self, mean, sigma):
        pass


class Focus():
    sigma


class Motion():
    delta_col, delta_row = UniformPDF(), UniformPDF()


class SceneModelParameters():
    stars = list of Stars  # mean and sigma of star properties
    background  # Background object
    diagnostics = list of scipyjunkobjects


class SceneModel():
    """A model which describes a single Kepler image.

    stars : list of Star objects
    """
    def __init__(self, stars=[], background_prior=None, focus_prior=None):
        self.star_priors = star_priors
        self.background_prior = background_prior

    def get_prior_means(self):
        """Returns the mean priors guess based on priors."""
        # Call the mean functions of all the Prior objects
        return ...

    def predict(self, params=self.get_initial_guesses()):
        """Produces a synthetic Kepler 2D image given a set of scene parameters."""
        put a star at position col + delta_col
        return synthetic_image

    def fit(self, observed_data):
        param = self.get_initial_guesses()
        score = self.predict(param) - observed_data
        whatevz = optimize(score)
        return SceneModelParameters(whatevz, garbage)


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
