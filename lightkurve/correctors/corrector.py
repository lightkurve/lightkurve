"""Implements the abstract `Corrector` base class to document the generic
methods the subclasses should aim to provide.

`Corrector` classes must adopt the following design:
- `__init__()` takes the required data (e.g. LightCurve, TargetPixelFile);
- `correct()` takes optional parameters and returns a LightCurve;
- `diagnose()` creates plots to elucidate the user's most recent call to `correct()`.
"""
class Corrector(object):
    """Abstract base class."""
    def correct(self):
        """Returns a corrected LightCurve."""
        raise NotImplementedError("This is an abstract base class.")

    def diagnose(self):
        """Returns plots which elucidate the most recent call to `correct()`."""
        raise NotImplementedError("This is an abstract base class.")
