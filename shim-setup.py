#!/usr/bin/env python

# This is a shim setup.py file which only serves the purpose of allowing us
# to use setuptools to create an editable install during development,
# i.e. it allows us to run `python shim-setup.py develop`.
# For all other installation and building purposes, Lightkurve v2.x switched
# to using the `poetry` build tool.
# cf. https://snarky.ca/what-the-heck-is-pyproject-toml/

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="lightkurve",
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        packages=setuptools.find_packages(where="."),
        package_dir={"": "."},
        include_package_data=True,
    )
