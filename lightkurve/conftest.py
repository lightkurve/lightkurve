def pytest_runtest_setup(item):
    """Our tests will often run in headless virtual environments. For this
    reason, we enforce the use of matplotlib's robust Agg backend, because it
    does not require a graphical display.
    
    This avoids errors such as:
        c:\hostedtoolcache\windows\python\3.7.5\x64\lib\tkinter\__init__.py:2023: TclError
        This probably means that tk wasn't installed properly.
    """
    import matplotlib
    matplotlib.use('Agg')
