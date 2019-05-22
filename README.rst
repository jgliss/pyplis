|build-status| |docs|

Pyplis is a Python toolbox originally developed for the analysis of UV SO2 camera data. The software includes a comprehensive and flexible collection of algorithms for the analysis of atmospheric imaging data and is tested for all major operating systems and python 3 as well as python 2.7.

Contact: Jonas Gliß (jonasgliss@gmail.com)

News / Notifications
====================

- **Pyplis 1.4.3 is released**

  - supports now both Python 2 and 3
  - can be easily installed (including all requirements) via ``conda install -c conda-forge pyplis``

- **NOTE (Python 2.7 retires soon)**

  If you are still using Python 2.7 (or any other Python 2 version), please consider updating your installation to Python 3, `since Python 2 is reaching its end-of-life soon <https://pythonclock.org/>`_.

Code documentation and more
===========================

The code documentation of pyplis and more information is hosted on `Read the Docs <http://pyplis.readthedocs.io/>`_.

Main features
=============

- Detailed analysis of the measurement geometry including automatic retrieval of distances to the emission plume and/or to topographic features in the camera images (at pixel-level).
- Several routines for the retrieval of plume background intensities (either from plume images directly or using an additional sky reference image).
- Automatic analysis of cell calibration data.
- Correction for cross-detector variations in the SO2 sensitivity arising from wavelength shifts in the filter transmission windows.
- DOAS calibration routine including two methods to identify the field of view of a DOAS instrument within the camera images.
- Plume velocity retrieval either using an optical flow analysis or using signal cross correlation.
- Histogram based post analysis of optical flow field for gas velocity analysis in low contrast image regions, where the optical flow fails to detect motion (cf. `Gliss et al., 2018, AMT <https://www.atmos-meas-tech.net/11/781/2018/>`_).
- Routine for image based correction of the signal dilution effect based on contrast variations of dark terrain features located at different distances in the images.
- Support of standard image formats including `FITS format <https://de.wikipedia.org/wiki/Flexible_Image_Transport_System>`_.
- Easy and flexible setup for data import and camera specifications.

A detailed description of pyplis and its features (including analysis examples) can be found in `Gliss et al., 2017, MDPI Geosciences <http://www.mdpi.com/2076-3263/7/4/134>`_.

Installation instructions and Requirements
==========================================

We recommend using the `Anaconda Python distribution <https://www.anaconda.com/distribution/>`_ (or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, if you want to save disk space) and to use the *conda* package manager. Why? `See, e.g. here for some good reasons <https://www.opensourceanswers.com/blog/best-practices-with-conda.html>`_.

Below it is assumed that you made yourself familiar with the *conda* package manager and that it is installed on your system. It is recommended to have a look at the guidelines related to `conda virtual environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Comment regarding conda environments
------------------------------------
We highly recommend to work in individual conda environments for your different projects and not to install everything into your Anaconda root environment (base), which is usually activated by default. In other words: please do not install pyplis into your root environment but create a new one using::

  conda create -n my_awesome_conda_environment

`Why? <https://www.opensourceanswers.com/blog/best-practices-with-conda.html>`_

Installation using conda
------------------------
Pyplis is available via the `conda-forge channel <https://anaconda.org/conda-forge/pyplis>`_ and can be easily installed via::

  conda install -c conda-forge pyplis

This will install all requirements as well. This is the recommended (and by far easiest) way to get pyplis running on your system.

Requirements
------------

Before installing pyplis, make sure you have all requirements installed (which is done automatically if you install pyplis via conda as described in previous section).

A list of all mandatory requirements can be found in the provided conda environment file `pyplis_env.yml <https://github.com/jgliss/pyplis/blob/master/pyplis_env.yml>`_, which can also directly be used to install the requirements, as described below.

Optional dependencies (to use extra features)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Pillow (PIL fork) >= 3.3.0

  - may be used to define custom image read functions, see e.g. `this example <https://pyplis.readthedocs.io/en/latest/api.html#pyplis.custom_image_import.load_hd_new>`_
  - We recommend using ``pip install pillow`` rather than ``conda install pillow`` due to
  - well known installation issues, e.g. `here <https://github.com/python-pillow/Pillow/issues/2945>`_

- pydoas >= 1.0.0 (comes with conda installation and provided environment file)

Installation of the requirements
---------------------------------

Before installing *Pyplis*, you need to install all requirements. To do so, you may either use the provided conda environment file or install all requirements manually, as described in the following two sections. All instructions below assume that you use `Anaconda <https://www.anaconda.com/>`_ as package manager.

Installation of requirements using provided conda environment file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can install all mandatory requirements using the provided environment file *pyplis_env.yml* (or *pyplis_env_py27.yml* if you still use python 2.7). You can install the environment file into a new environment (here, named *pyplis*) using::

  conda env create -n pyplis_env_test -f pyplis_env.yml

Or you may install it into an existing environment by activating the environment and then::

  conda env update -f=pyplis_env.yml

Manual installation of requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may also install all requirements from scratch as described in the following step-by-step guide::

  conda create --name pyplis # creates new conda environment with name pyplis (optional)
  conda activate pyplis # activates new environment (optional)
  conda install -c conda-forge scipy pandas astropy basemap opencv geonum pydoas

Installation of pyplis
----------------------

Here, you have 3 options.

Via conda
^^^^^^^^^
From the command line, call::

  conda install -c conda-forge pyplis

This option installs pyplis and all requirements automatically.

Via pip
^^^^^^^^
From the command line, call::

  pip install pyplis

This option only installs pyplis, you have to install all requirements yourself (for details, see previous sections).

From Source
^^^^^^^^^^^
In order to install from source, please download or clone the `repo <https://github.com/jgliss/pyplis>`_ (or one of the `pyplis releases <https://github.com/jgliss/pyplis/releases>`_) into a local directory of your choice. Then, unzip and from the project root directory (the one that contains setup.py file) call::

  python setup.py install

This option only installs pyplis, you have to install all requirements yourself (for details, see previous sections).

Note
^^^^
Use Option 2 if you want to run the tests and / or example scripts (since these are not shipped with the PyPi installation that uses a binary wheel of Pyplis).

Installation remarks and known issues
-------------------------------------

- If you work on a Windows machine and run into problems with installation of one of the requirements (e.g. if you already had Python 2.7 installed and want to upgrade dependencies such as numpy or scipy), check out the pre-compiled binary wheels on Christoph Gohlke's `webpage <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_

- Sometimes it is helpful, to reinstall your whole Python environment (or, if you use Anaconda, `create a new one <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_) rather than trying to upgrade all dependencies to the required version

- If you find a bug or detect a specific problem with one of the requirements (e.g. due to future releases) please let us know or `raise an issue <https://github.com/jgliss/pyplis/issues>`_.

**Do not hesitate to contact us (or raise an issue), if you have problems installing pyplis.**

Getting started
===============

The Pyplis `example scripts <https://github.com/jgliss/pyplis/tree/master/scripts>`_ (see previous point) are a good starting point to get familiar with the features of Pyplis and for writing customised analysis scripts. The scripts require downloading the Etna example dataset (see following section for instructions). If you require more thorough testing, refer to this `wiki entry <https://github.com/jgliss/pyplis/wiki/Contribution-to-pyplis-and-testing>`_

Example and test data
=====================

The pyplis example data (required to run example scripts) is not part of the installation. It can be downloaded `from here <https://folk.nilu.no/~arve/pyplis/pyplis_etna_testdata.zip>`_ or automatically downloaded in a Python shell (after installation) using::

  import pyplis
  pyplis.inout.download_test_data(<desired_location>)

which downloads the data into the *my_pyplis* directory if <desired_location> is unspecified. Else, (and if <desired_location> is a valid location) it will be downloaded into <desired_location> which will then be added to the supplementary file *_paths.txt* located in the installation *data* directory. It can then be found by the test data search method::

  pyplis.inout.find_test_data()

The latter searches all paths provided in the file *_paths.txt* whenever access to the test data is required. It raises an Exception, if the data cannot be found.

Note
----

If the data is downloaded manually (e.g. using the link provided above), please make sure to unzip it into a local directory *<desired_location>* and let pyplis know about it, using::

  import pyplis
  pyplis.inout.set_test_data_path(<desired_location>)

Scientific background
=====================

The article:

*Pyplis - A Python Software Toolbox for the Analysis of SO2 Camera Images for Emission Rate Retrievals from Point Sources*, Gliß, J., Stebel, K., Kylling, A., Dinger, A. S., Sihler, H., and Sudbø, A., Geosciences, 2017

introduces *Pyplis* and implementation details. Furthermore, the article provides a comprehensive review of the technique of SO2 cameras with a focus on the required image analysis. The paper was published in December 2017 as part of a special issue on `Volcanic plumes <http://www.mdpi.com/journal/geosciences/special_issues/volcanic_processes>`_ of the Journal *Geosciences* (MDPI).
`Download paper <http://www.mdpi.com/2076-3263/7/4/134>`_.

Citation
--------
If you find *Pyplis* useful for your data analysis, we would highly appreciate if you acknowledge our work by citing the paper. Citing details can be found `here <http://www.mdpi.com/2076-3263/7/4/134>`__.

Copyright
=========

Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License a published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, `see here <http://www.gnu.org/licenses/>`_.

Note
----
The software was renamed from **piscope** to **Pyplis** on 17.02.2017

.. |build-status| image:: https://travis-ci.org/jgliss/pyplis.svg?branch=master
    :target: https://travis-ci.org/jgliss/pyplis

.. |docs| image:: https://readthedocs.org/projects/pyplis/badge/?version=latest
    :target: https://pyplis.readthedocs.io/en/latest/?badge=latest
