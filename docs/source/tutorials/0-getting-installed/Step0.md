# Introduction to "An introduction into the tools and tutorials available for the analysis of TESS data" a.k.a. Step-0

## Motivation

Getting started with Lightkurve and Kepler/K2/TESS data and tools is easy! However, it does require a couple of steps which can be daunting if you haven't done them before.  Many astronomers who have a laptop set-up for doing active research will likely have the necessities available, however others including (but not limited to!) new students, interns, and the interested amateur astronomer, may find this brief guide useful.  

## Learning Goals

In this tutorial we will take you from a brand new laptop or computing environment to an environment that will allow you to run the [`Lightkurve` Tutorial Notebooks](https://docs.Lightkurve.org/tutorials/), and begin exploring and working with Kepler, TESS, and other Time-Domain data. 

## What This Means in Practice

Practically speaking, the [subsequent Lightkurve tutorials](https://docs.Lightkurve.org/tutorials/) are designed around the concept of using Python and associated packages to interact with the Kepler and TESS data in an interactive Jupyter Notebook environment. This means that the tutorials will make the following assumptions:

1. That you are operating in a [Jupyter Notebook](https://jupyter.org/) Environment

2. Running a 3.6+ version of Python with either a package manager, or preferably with an environment manager such as [Conda](https://docs.conda.io/en/latest/) (that is installed by default with [Anaconda](https://www.anaconda.com/products/distribution)) (see below)

3. With a number of packages installed, listed at the top of each tutorial, including (but not limited to) Lightkurve

## Why do we assume this?

[Lightkurve](https://docs.Lightkurve.org/) is a Python package that will be the basis of the subsequent tutorials as it has been designed to work with time-series data from the Kepler and TESS satellites, and is great for data interaction, exploration, and analysis as it incorporates many tools that the community might want for working with TESS data.  

To work with Lightkurve in these tutorials, we want an environment that will allow us to rapidly and interactively explore and analyze TESS data, with a visible workflow, and to share this workflow with others.  To do this, we plan to write our Python analyses/code inside of [Jupyter Notebooks](https://jupyter.org/), which describe themselves as a 'web-based interactive computing platform that combines live code, equations, narrative text, visualizations, and more'.  They are a great way to explore data and to share code, analyses, and results like we do in the subsequent tutorials, and run in your browser once set up. Their usage has also become incredibly common across both many scientific fields as well as industry usage in some fields such as data science and business analytics. 

Working with TESS and Lightkurve does not *require* the use of Jupyter Notebooks, despite our tutorials presenting them in that style.  All of tutorial Python commands can be run from the Python command line or in a Python script, although the tutorials may require minimal modifications to the plotting commands to make the graphics appear.  

Lightkurve requires a relatively recent version of Python 3.X (although, on rare occasions, not the most recent version) as well as scipy and other Python packages to be installed.  

A <u>package manager</u>, such as [pip](https://pypi.org/project/pip/) (the default one installed with Python), will keep track of Python package requirements and when you install a new package, install both missing package dependencies and upgrade/downgrade package versions to (hopefully) meet the requirements of all of your installed packages. You will however, only ever have one version of Python and one list of installed packages.

An <u>environment manager</u>, such as [Conda](https://docs.conda.io/en/latest/) (and which is installed as a part of Anaconda), will allow you to have multiple, distinct, parallel, environments that you can switch between with a single command.  Each environment can (and will) have unique, and different versions of Python and Python packages.  Environments are a useful (and almost mandatory) tool when you have multiple Python workflows that have different, and conflicting, package and/or Python requirements. Conda in particular can serve as both a package manager and an environment manager. 

## How do we get there?

For the remainder of this tutorial, we will assume that you will use [Anaconda](https://www.anaconda.com/products/distribution) to install Python, Jupyter notebooks, Lightkurve, and dependent packages.  We choose Anaconda as it is free to individual users, its package managing tool Conda is open-source, the packages in its base repository undergo a rigorous security check, is available for OsX, Windows, and Linux, and it is widely used by both the astronomy and wider community. It also includes Jupyter notebooks, matplotlib, and many commonly used packages in its default distribution. Other great options are available such as [virtualenviroment](https://pypi.org/project/virtualenv/), [pipenv](https://pypi.org/project/pipenv/), and more, and may be used to achieve similar results (although the exact steps achieve them will be left to the reader).  

## What if I can't do this?

This workflow requires only a modest amount of computing power and should be functional on most any home computer or laptop with a high-speed internet connection, and if you have access to these two items and intend to work with TESS data we strongly encourage you to follow the below workflow.  If you are unable to do this however (such as having access to only a mobile device, or strict internet data caps) one potential solution is to use [Google Colab](https://colab.research.google.com/?utm_source=scs-index).  Google Colab is a cloud-based  'Jupyter notebook' like environment, where all of your data downloads and computer processing live in the cloud on a remote machine.  The upside to this is that all you need is a web-browser and a stable internet connection to participate in TESS science.  The downside is that some aspects of the notebook usage may be slower or more cumbersome, and the [TESS Data-Processing Tutorial notebooks](https://heasarc.gsfc.nasa.gov/docs/tess/data-analysis-tools.html) will require some modifications to work.  We have Translated the first tutorial into a Colab notebook [here](https://colab.research.google.com/drive/1kNUcBBc2x_06PupFIdV7RAiGEmOoi0LL?usp=sharing), and leave any modifications necessary for future notebooks as an exercise for the reader. 

Some features are not supported by Google Colab - most notably interactive plots made with the Bokeh package (e.g. Lightkurve.interact())

### A note on best practices & "Stretch Goals" discussed below:

This guide is primarily designed to get you looking at TESS Data as soon as possible in the most simplified manner.  This means that there are a few Python 'best practices' that we're skipping over in pursuit of speed and simplicity.  We'll note what those are, label them as <u>stretch goals</u> down below, and while they are truly optional we suggest that you consider following them if you are comfortable with the necessary steps and plan on doing further Python coding beyond these tutorials.  These steps can always be performed later as well.   

### Step 1 - Download Anaconda

Install [Anaconda](https://www.anaconda.com/products/distribution) on your operating system of choice - the anaconda webpage will do a good job of guessing which installer that you need, but will also have a full list of installers available at the bottom of the page or found via the 'Get Additional Installers' link.  This is mostly useful if you are download the installer on a different computer than you plan to install Anaconda on.  

### Step 2 - Install Anaconda

Once downloaded, use the file to install Anaconda, following the [instillation instructions](https://docs.anaconda.com/anaconda/install/) from the Anaconda Documentation.  Choose your operating system (e.g. Apple OsX, Windows, etc) from the left side bar to see the appropriate instructions.

### Step 3 - Start and Enter a Jupyter Notebook

With anaconda installed, we next want to start a Jupyter notebook with a Python 3 kernel.  

The most common way to do this is to open up our computers command line terminal and navigate to the directory from which you would like to work on TESS data using the command line. 

#### For Apple OSX - Open up your terminal application under Applications/Utilities/Terminal

You should see something like this: 

![](images/openterm.gif)

Then, navigate to your target directory (it will open up in your 'home' directory, /Users/{username}), which is fine.  You can also make a new directory,  say TESS, and move into that directory using the command line,  e.g.

`cd ~\` This will move to your home directory if you are not already there

`mkdir TESS`This will make a new directory with the name "TESS"

`cd TESS` This will move your terminal session into the TESS directory

![Use the Terminal to make and navigate to a TESS directory](images/mvTESS.gif)

#### For Windows - navigate to the folder you want to run the tutorials at and open up your command shell

You can do this by pressing ALT+D, typing in cmd, then hitting Enter.

![Windows](images/windows.gif)

#### <u>Stretch goal #1: Add conda-forge to your repository list[^1]</u>

#### <u>Stretch Goal #2: Create and enter a new Conda environment[^2]</u>

### Then, following the [jupyter notebooks documentation](https://docs.jupyter.org/en/latest/running.html), you will start up the jupyter notebook server by entering the following at the command line (on both windows and OsX):

`jupyter notebook`

This should cause your terminal window to start printing debug information, and you should see your web browser pop open and a new page open with the notebook dashboard.    

#### If this worked, and you navigated to an empty folder called "TESS", you should see something like this:

![newly opened notebook server](images/jupyter-blank.png)

This is empty, because the folder is empty!  To create a new notebook from which to work on TESS data and follow along with the tutorials/quickstart, click on "New" on the top right side of the page and select "Python 3" under the notebook heading

This will create a new Python 3 notebook called "Untitled" that you can execute Python code from:

![creating a new notebook](images/newnb.gif)

### Now, the last thing that our introduction tutorial requires is for Lightkurve to be installed

#### <u>Stretch Goal #1.5: Install Lightkurve with `conda` instead of `pip`[^3]</u>

#### Otherwise, we can install Lightkurve using the built-in Python pip package manager

To do this, enter the following command in the text box:

`! Python -m pip install Lightkurve --upgrade`

This command can be executed using the Run button (or shift+return/enter).  If this is successful, a number of lines of debug should pop up in a cell below this and end in the line (you may have to scroll through the cell) "Successfully installed ..." followed by a list of packages installed, e.g.:

![install Lightkurve](images/jupyter-instlk.png)

## Congratulations!  Now, you should be able to run the other [Lightkurve tutorials](https://docs.Lightkurve.org/tutorials/),  and start exploring Kepler, K2, TESS, and other Time-Domain data!

### **To run the other Lightkurve tutorials on your local machine, from the [Lightkurve tutorials webpage](https://docs.Lightkurve.org/tutorials/):**
- Click on the tutorial that you would like to run
- Click the "<u>Download</u>" link in the top-right of the webpage, next to the 'Notebook' button with the orange jupyter symbol
	- This will download a jupyter notebook ("what-are-lightcurve-objects.ipynb" for the first tutorial) file into your default download directory

- Move this file into the "TESS" folder that you opened up your jupyter notebook server from

- There will now be a new notebook visible in your jupyter notebook browser, which you can double-click to start up the notebook tutorial
	- The notebook browser is the web browser window that pops up after you type `jupyter notebook` into your terminal, and from which you opened up your test notebook

# What if something went wrong?

- If you're having trouble installing Anaconda, see [their help page](https://docs.anaconda.com/anaconda/install/)  

- if you are having trouble opening a Jupyter notebook via the terminal, you can try the [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator) application that has a graphical user interface and comes installed with anaconda. ![loading-ag-599](images/AnacondaNavigator.png)
  
  - You can Add Channels (#1; like Conda-Forge in <u>Stretch Goal #1[^1]</u>) using the channels button. 
  
  - You can create a new environment (#2, side tab, like in <u>Stretch Goal #2[^2]</u>) and activate it (#3; in the drop-down menu)
  
  - You can launch a Jupyter Notebook (#4; this will open in your home directory, and you can use the left sidebar to navigate to other folders)

- If you are having trouble past that, you can uninstall Anaconda and try again from scratch.  
  
  - If you completed <u>Stretch Goal #2[^2]</u> you can also delete your `TESS` environment (using `conda remove --name TESS --all`) and try again by creating a new environment.  

- If something is broken in the Jupyter notebook environment, you can restart the kernel by going to the 'kernel' menu, or `control-c` the "notebook server" in your terminal and re-start it to re-initialize everything.


- Jupyter Notebooks will load in the default browser. If you want to use a different browser you have to manually change the default one in your system settings.
	- If your browser refuses to run the interactive widgets (e.g. Lightkurve.interact()), you will need to manually set up the socket for Bokeh - e.g. for OsX type "conda env config vars set `BOKEH_ALLOW_WS_ORIGIN=localhost:8888` " in the terminal prior to running `jupyter notebook`

- If Jupyter Notebooks are running slow:
	- If you have multiple notebooks open, shutting down notebooks from the "Running" tab may improve your performance
	- Any jupyter notebooks servers that you start up will keep running until they are stopped, occupying both CPU and memory. Stopping them is generally done by either using the `Quit` button in the juypter notebook browser, or using `ctrl + c` in your terminal that is running the notebook server.
	- You can see a list of the running jupyter notebook servers by typing `jupyter server list` into the console


[^1]: ### <u>Stretch goal #1: Add conda-forge to your repository list</u>

	When Conda, our environment, goes to install a new software page it gets a list of available packages (and their requirements) from a repository, which is effectively a software package warehouse. The default repository installed with Anaconda is one that is curated by the company and is designed for stable, robust packages that pass certain standards for enterprise needs. These are excellent, but miss many scientific research packages (such as Lightkurve) that are open source and community developed, but more niche or not targeted at commercial users.

	[**Conda-Forge**](https://conda-forge.org/) is a community led open-source repository that uses github and "continuous integration" software practices to allow most open source Python packages to distribute themselves via the Conda environment manager. These practices, and the lack of hand-curation, also means that many conda-forge packages are more up-to date (but possibly not more stable) than those on the Anaconda repository. **Lightkurve** is available on both conda-forge and [pip](https://pypi.org/project/pip/) (the [PyPi](https://pypi.org/) Python package manager), but not the default Anaconda repository.

	The default tutorial below suggests to install Lightkurve via pip - however this can introduce inconsistencies in an environment in the future if you were to install more packages since you're now using multiple package managers. **<u>The best practice is to only use one package/environment manage unless unavoidable.</u>** To this end we will add the conda-forge repository channel and install Lightkurve via conda to ensure consistency.

	Conda considers each repository its own 'channel', and so to add the conda-forge repository via the command line terminal:

	`conda config --add channels conda-forge`

	You can now install Python packages that are in the conda-forge repository, however there is not a strict preference in where to source a package between the two repositories. To change this, and reduce the chance of future errors, you can activate `strict` channel priority via the command line terminal:

	`conda config --set channel_priority strict`

[^2]: ### <u>Stretch Goal #2: Create and enter a new Conda environment</u>

	Different Python packages have different requirements, and when running multiple projects or pipelines it is entirely possible to need to use two packages that require different versions of the same package (including even the version of Python itself!).  Environment managers help resolve this by allowing you to create multiple silos of installed packages that you can easily switch through depending on your needs.  **<u>The best practice here is to create a new environment for every particular project or task to ensure that when you install or modify Python packages you don't break anything in any Python workflow, and to avoid modifying the base environment</u>.** 

	Below we install Lightkurve, and following the default tutorial this will be installed in your default environment, which according to best practices should be left minimally changed.  Here we will create a new environment for the TESS tutorials and activate it so that when we install Lightkurve it will get installed in our new environment.  A new environment will not see any of the packages in the base environment, so we will also have to pass a list of packages.  We will not include the Lightkurve package in this list, although you could if you performed <u>Stretch Goal #1</u>.  

	The packages we will install *explicitly* are:

	- Python - to get a recent version of Python 3.x

	- notebook - for Jupyter notebooks

	There will also be a (significant) number of dependent packages that that these packages require to be functional, which conda will install.  

	We can create an environment named `TESS` including the specific packages above using the command-line terminal:

	`conda create --name TESS Python notebook`

	and entering `y` to proceed when prompted

	This has created a new environment, however, at the command-line we are still in our default environment.  To activate our new environment we enter:

	`conda activate TESS`

	**This will need to be entered each time you start a new terminal session** to enter the TESS environment, or added to your startup profile (e.g. .bashrc, .profile, etc) to automate this.  

[^3]: ### <u>Stretch Goal #1.5: Install Lightkurve with `conda` instead of `pip`</u>

	Assuming that you performed <u>Stretch Goal #1</u> above, you can install Lightkurve with conda (instead of using pip as below):

	`! conda install Lightkurve -y`

	(if this installs a different Python you may need to restart your Jupyter notebook from the terminal)

	This is following the best practice of using only a single environment/package manager where possible.  And, if you started up Jupyter notebooks after creating and activating a new `TESS` environment, this will install Lightkurve into your `TESS` environment.  
