import os
import warnings

import astropy.config as astropyconfig


ROOTNAME = 'lightkurve'


class ConfigNamespace(astropyconfig.ConfigNamespace):
    rootname = ROOTNAME


class ConfigItem(astropyconfig.ConfigItem):
    rootname = ROOTNAME


def get_config_dir():
    """
    Determines the package configuration directory name and creates the
    directory if it doesn't exist.

    This directory is typically ``$HOME/.lightkurve/config``, but if the
    XDG_CONFIG_HOME environment variable is set and the
    ``$XDG_CONFIG_HOME/lightkurve`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Returns
    -------
    configdir : str
        The absolute path to the configuration directory.

    """
    return astropyconfig.get_config_dir(ROOTNAME)


def get_cache_dir():
    """
    Determines the default Lightkurve cache directory name and creates the
    directory if it doesn't exist. If the directory cannot be access or created,
    then it returns the current directory (``"."``).

    This directory is typically ``$HOME/.lightkurve/cache``, but if the
    XDG_CACHE_HOME environment variable is set and the
    ``$XDG_CACHE_HOME/lightkurve`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    The value can be also configured via ``cache_dir`` configuration parameter.

    Returns
    -------
    cachedir : str
        The absolute path to the cache directory.

    Examples
    --------
    To configure "/my_research/data" as the `cache_dir`, users can set it:

    1. in the user's ``lightkurve.cfg`` file::

        [config]
        cache_dir = /my_research/data

    2. at run time::

        import lightkurve as lk
        lk.conf.cache_dir = '/my_research/data'

    See :ref:`configuration <api.config>` for more information.
    """
    from .. import conf

    cache_dir = conf.cache_dir
    if cache_dir is None or cache_dir == "":
        cache_dir = astropyconfig.get_cache_dir(ROOTNAME)
    cache_dir = _ensure_cache_dir_exists(cache_dir)
    cache_dir = os.path.abspath(cache_dir)

    return cache_dir


def _ensure_cache_dir_exists(cache_dir):
    if os.path.isdir(cache_dir):
        return cache_dir
    else:
        # if it doesn't exist, make a new cache directory
        try:
            os.mkdir(cache_dir)
        # user current dir if OS error occurs
        except OSError:
            warnings.warn(
                "Warning: unable to create {} as cache dir "
                " (for downloading MAST files, etc.). Use the current "
                "working directory instead.".format(cache_dir)
            )
            cache_dir = "."
        return cache_dir


def warn_if_default_cache_dir_migration_needed():
    from .. import conf

    if not conf.warn_legacy_cache_dir:
        return

    cache_dir = conf.cache_dir
    if not(cache_dir is None or cache_dir == ""):
        # If an user has specified a custom cache dir, the check won't be performed.
        # Not only is the check somewhat irrelevant, the behavior is also required
        # to support the case that the user configures the legacy `~/.lightkurve-cache`
        # as the cache dir (e.g., to support running other apps/packages that require
        # older lightkurve, especially lightkurve v1.x.)
        return

    # migration check done only if default is used
    old_cache_dir = os.path.join(os.path.expanduser("~"), ".lightkurve-cache")
    new_cache_dir = os.path.join(os.path.expanduser("~"), ".lightkurve", "cache")
    if os.path.isdir(old_cache_dir):
        warnings.warn(
            f"The default Lightkurve cache directory, used by download(), etc., has been moved to {new_cache_dir}. "
            f"Please move all the files in the legacy directory {old_cache_dir} to the new location "
            f"and remove the legacy directory. "
            f"Refer to https://docs.lightkurve.org/reference/config.html#default-cache-directory-migration "
            f"for more information."
        )
