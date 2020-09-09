"""DEPRECATED: `LightCurveFile` classes were removed in Lightkurve v2.0 in
favor of only having `LightCurve` classes.  To minimize breaking code, we
retain the `LightCurveFile` classes here as wrappers around the new
`LightCurve` objects, but will remove these wrappers in a future version..
"""
from . import LightCurve, KeplerLightCurve, TessLightCurve

LightCurveFile = LightCurve
KeplerLightCurveFile = KeplerLightCurve.read
TessLightCurveFile = TessLightCurve.read
