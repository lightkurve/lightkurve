
Running the unit tests
======================

*Lightkurve* has many `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`_ which check that our basic functionality works as we expect. Before we can make any changes to *Lightkurve* we have to check that these tests all pass. If you open a `pull request <contributing>`_ to *Lightkurve*, these tests will be run automatically. If they pass, your code can be reviewed and potentially merged into the *Lightkurve* master branch.

However, running these tests online can take a long time. Running the tests locally on your machine is much faster, and will let you check that your work still maintains the expected *Lightkurve* behavior as you develop your code.


How do I run tests locally?
---------------------------

First off, you need to find the directory that your *Lightkurve* installation is in. You can check this by looking at the *Lightkurve* path::

    import lightkurve as lk
    print(lk.__path__)

In a terminal, `cd` into the *Lightkurve* directory. Yours should look something like this:

.. image:: https://user-images.githubusercontent.com/14965634/53126462-01a9c780-3515-11e9-9031-1f7cd06fcfb3.png
    :width: 500 px

Note that here we're using the master branch. If you want to run the tests on a branch you are developing, switch to that branch using `git checkout branchname`. Once you're in the `lightkurve` directory, go to the tests directory in `lightkurve/tests`.

.. image:: https://user-images.githubusercontent.com/14965634/53126884-ff943880-3515-11e9-9c1e-e4efc10b5bc2.png
    :width: 500 px


In this directory you should find several tests labeled `test_xxx.py`. You can run a test using `pytest`. For example, to test the `lightkurve.targetpixelfile` module you would execute::

    pytest test_targetpixelfile.py


If the tests are successful, you should see a green success message such as

.. image:: https://user-images.githubusercontent.com/14965634/53127264-e770e900-3516-11e9-8bfa-07284f499eef.png
    :width: 500 px


Why are some of the tests marked "s"/ Why are some tests skipped?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running some of our tests requires external data, e.g. some require data to be downloaded from MAST. These tests take a little longer, and so we skip them by default. In order to run all the tests simply use::

    pytest test_targetpixelfile.py --remote-data



My tests passed, but I got warning messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you will get warnings in your tests, causing your result to be yellow instead of green. For example, you may get an output that looks like this:

.. image:: https://user-images.githubusercontent.com/14965634/53127518-7f6ed280-3517-11e9-97d4-ba0af724308e.png
    :width: 500 px

While this is not ideal, some *Lightkurve* tests do raise warnings currently. This will become less and less likely as *Lightkurve* improves.


My tests failed
~~~~~~~~~~~~~~~

If your test fails, don't worry, this is what tests are for. Take a look at the traceback that pytest provides for you. If your test has failed then you will see an F next to the test you've run, for example:

.. image:: https://user-images.githubusercontent.com/14965634/53128031-b396c300-3518-11e9-9083-d12efef46043.png
    :width: 500 px

Underneath, you will then see the traceback of the test that failed. For example, the traceback below shows that there is an `AssertionError`.

.. image:: https://user-images.githubusercontent.com/14965634/53127788-38cda800-3518-11e9-866b-b7eee448041e.png
    :width: 500 px

In the test, we have made an assertion
`assert_array_equal(lc_add.flux, lc.flux + 2)`.

However in the traceback we can see that these two arrays are not actually equal, and so the test is breaking.

.. image:: https://user-images.githubusercontent.com/14965634/53128140-ff496c80-3518-11e9-95ca-3c2a06eddad8.png
    :width: 500 px

Use this information to correct the code you're developing until the tests pass. In rare cases (such as the case above) it is the test itself that is incorrect, not the lightkurve code. If you believe there is an error in one of the tests, point it out in your PR for everyone to comment and discuss.


When should I run tests?
------------------------

Before you open a PR to *Lightkurve*, ideally you should run these tests locally and check that they are all passing. If they aren't passing, and you are confused as to why they are not, you can open a PR and ask for help.


Can I write my own test?
------------------------

Ideally, any PR opened to *Lightkurve* with new functionality should include some tests. These tests check that the basic functionality of your PR works. That way, if in future people create new features that break your PR, we will be alerted. Read through the `pytest` documentation and take a look at our existing tests to get an idea of how to write your own.


I can't run any tests.
----------------------

We run our unit tests using `pytest`. This should have been installed when you installed *Lightkurve*. However, if your tests don't run, you may want to check all the test dependencies are installed by running (with `pip`)::

    pip install pytest pytest-cov pytest-remotedata

or equivalently if you are managing your Python environment using `conda`::

    conda install pytest pytest-cov pytest-remotedata
