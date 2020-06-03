"""DEPRECATED: `LightCurveFile` classes were removed in Lightkurve v2.0 in
favor of only having `LightCurve` classes.  To minimize breaking code, we
retain the `LightCurveFile` classes here as wrappers around the new
`LightCurve` objects, but will remove these wrappers in a future version..
"""
from astropy.utils import deprecated

from .utils import LightkurveDeprecationWarning
from . import LightCurve, KeplerLightCurve, TessLightCurve


LightCurveFile = LightCurve
KeplerLightCurveFile = KeplerLightCurve.read
TessLightCurveFile = TessLightCurve


"""
@deprecated("2.0", alternative="LightCurve", warning_type=LightkurveDeprecationWarning)
class LightCurveFile(LightCurve):
    def __init__(self, path, **kwargs):
        if isinstance(path, str):
            super().__init__(LightCurve.read(path, **kwargs))
        else:
            super().__init__(path, **kwargs)


@deprecated("2.0", alternative="KeplerLightCurve", warning_type=LightkurveDeprecationWarning)
class KeplerLightCurveFile(KeplerLightCurve):
    def __init__(self, path, **kwargs):
        self.__class__ == KeplerLightCurve
        if isinstance(path, str):
            super().__init__(KeplerLightCurve.read(path, **kwargs))
        else:
            super().__init__(path, **kwargs)


@deprecated("2.0", alternative="TessLightCurve", warning_type=LightkurveDeprecationWarning)
class TessLightCurveFile(TessLightCurve):
    def __init__(self, path, **kwargs):
        if isinstance(path, str):
            super().__init__(TessLightCurve.read(path, **kwargs))
        else:
            super().__init__(path, **kwargs)
"""