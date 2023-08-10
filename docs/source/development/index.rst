.. _development:

Developing for Lightkurve
=========================

Lightkurve is an open-source, community driven package. We strongly encourage users to contribute and develop new features for Lightkurve. These pages first discuss the vision of Lightkurve, and then how to contribute Pull Requests on github for new features. These pages also discuss how to compile our documentation (including this page!) and how we release a new version of Lightkurve. Use the menu bar on the left to scroll through the docs or click below.

Developer documentation
-----------------------

.. toctree::
    :maxdepth: 1

    vision.rst
    contributing.rst
    testing.rst
    documentation.rst
    release-procedure.rst

Setting up a Development Environment
====================================

To set up a development environment for Lightkurve you can follow the steps below.

1. Fork Lightkurve's GitHub repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to create a copy of Lightkurve's GitHub repository by logging into GitHub, browsing to
`https://github.com/lightkurve/lightkurve <https://github.com/lightkurve/lightkurve>`_,
and clicking the ``Fork`` button in the top right corner.

2. Clone the fork to your computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Head to a local directory of your choice and download your fork:

.. code-block:: bash

    $ git clone https://github.com/YOUR-GITHUB-USERNAME/lightkurve.git


.. _install-dev-env:

3. Install the development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightkurve uses the `poetry <https://python-poetry.org/>`_ package to create an isolated development
environment which you can use to modify and test changes to the source code without interfering with
your existing Python environment.

You can set up the environment as follows:

.. code-block:: bash

    $ cd lightkurve
    $ python -m pip install poetry
    $ poetry install

A key advantage of the development environment is that any changes you make to the Lightkurve source
code will be reflected right away, i.e., there is no need to re-install Lightkurve or the environment
after every change.

To run code in the development environment, you will need to prefix every Python command with
`poetry run`. For example:

.. code-block:: bash

    $ poetry run python YOUR-SCRIPT.py
    $ poetry run jupyter notebook
    $ poetry run pytest  # runs all the unit tests

You can find more details on the `poetry website <https://python-poetry.org/>`_,
and you can find additional examples of tasks developers commonly execute in the development
environment in Lightkurve's `Makefile <https://github.com/lightkurve/lightkurve/blob/main/Makefile>`_.

.. note::

    The use of `poetry` is not required to prepare and propose modifications to Lightkurve.
    If you wish to do so, you can install the development version in your current
    Python environment as follows:

    .. code-block:: bash

        $ cd lightkurve
        $ python -m pip install .

    In this scenario, you will have to re-run `pip install .` every time you make changes
    to the source code.

    To avoid this extra step, you have the option of creating a symbolic
    link from your environment's `site-packages` directory to the lightkurve source code tree
    as follows:

    .. code-block:: bash

        $ python -m pip uninstall lightkurve
        $ python -m pip install --editable .  # creates the symbolic link

    This "editable install" method requires `pip` version `21.3` or higher.
    While this method of installing Lightkurve is not usually recommended, it can be useful
    when you wish to modify and test multiple different packages in a single environment.


4. Add a link to the main repository to your git environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To be able to pull in any recent changes, we need to tell your copy of lightkurve
where the upstream repository is located:

.. code-block:: bash

    $ git remote add upstream https://github.com/lightkurve/lightkurve.git

To verify that everything is setup correctly, execute:

.. code-block:: bash

    $ git remote -v

You should see something like this:

.. code-block:: bash

    origin	https://github.com/YOUR-GITHUB-USERNAME/lightkurve.git (fetch)
    origin	https://github.com/YOUR-GITHUB-USERNAME/lightkurve.git (push)
    upstream	https://github.com/lightkurve/lightkurve.git (fetch)
    upstream	https://github.com/lightkurve/lightkurve.git (push)



