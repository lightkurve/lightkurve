"""Implements the abstract `Corrector` base class to demonstrate which
standard methods the subclasses should aim to provide.

The design rules of a `Corrector` class are:
- the constructor takes all the required data;
- `correct()` takes optional correction parameters and returns a LightCurve;
- `diagnose()` takes the same arguments as `correct()` but returns plot(s).
"""
class Corrector(object):
    """Abstract base class."""
    def correct(self):
        raise NotImplementedError("This is an abstract base class.")

    def diagnose(self):
        raise NotImplementedError("This is an abstract base class.")

    def interact(self):
        raise NotImplementedError("This is an abstract base class.")
