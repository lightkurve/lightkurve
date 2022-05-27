.. _contributing:

======================================
Reporting issues and proposing changes
======================================

Lightkurve is actively developed on its `GitHub repository <https://github.com/lightkurve/lightkurve>`_.

If you encounter a problem with Lightkurve, we encourage you to
`open an issue on the GitHub repository <https://github.com/lightkurve/lightkurve/issues>`_.

If you would like to propose a change or bug fix to Lightkurve, please go ahead and open a pull request
using the steps explained below.


Proposing changes to Lightkurve using GitHub Pull Requests
----------------------------------------------------------

We welcome suggestions for enhancements or new features to Lightkurve via GitHub.

If you want to make a significant change such as adding a new feature,
we recommend opening a GitHub issue to discuss the changes first.
Once you are ready to propose the changes, please go ahead and open a pull request.

If in doubt on how to open a pull request, we recommend Astropy's
"`How to make a code contribution <http://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_" tutorial.
In brief, the steps are as follows:


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


5. Create a new branch
~~~~~~~~~~~~~~~~~~~~~~

You are now ready to start contributing changes.
Before making new changes, always make sure to retrieve the latest version
of the source code as follows:

.. code-block:: bash

    $ git checkout main
    $ git pull upstream main

You are now ready to create your own branch with a name of your choice:

.. code-block:: bash

    $ git branch name-of-your-branch
    $ git checkout name-of-your-branch


6. Make changes and add them to the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can now go ahead and modify source files.
When you are happy about a change, you can commit it
to your local version of the repository as follows:


.. code-block:: bash

    $ git add FILE-YOU-ADDED-OR-MODIFIED
    $ git commit -m "description of changes"


7. Push your changes to GitHub and open a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, send the changes to the fork of Lightkurve that resides in your GitHub account:

.. code-block:: bash

    $ git push origin name-of-your-branch

Head to https://github.com/lightkurve/lightkurve after issuing the `git push`
command above. You should automatically see a button that says "Compare and open a pull request".
Click the button and submit your pull request!
