"""Implements the abstract `Corrector` base class to demonstrate which
standard methods the subclasses should aim to provide.
"""
class Corrector(object):
    """Abstract base class."""
    def correct(self):
        raise NotImplementedError("This is an abstract base class.")

    def diagnose(self):
        raise NotImplementedError("This is an abstract base class.")

    def interact(self):
        raise NotImplementedError("This is an abstract base class.")
