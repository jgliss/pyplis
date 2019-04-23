.. image:: docs/_graphics/logo_pyplis_v3_with_banner.png
   :target: pageapplet/index.html



Pyplis is a Python toolbox originally developed for the analysis of UV SO2 camera data. The software includes a comprehensive and flexible collection of algorithms for the analysis of atmospheric imaging data.

Contact: Jonas Gliss (jonasgliss@gmail.com)

This branch supports only python **python 2.7**. Please use branch **py3** if you intend to use pyplis with python 3.
The **py3** branch contains a *beta-version* of pyplis which supports both **python 2 and 3**.


Code documentation and more
============================

The code documentation of pyplis and more information is hosted on `Read the Docs <http://pyplis.readthedocs.io/en/latest/index.html>`__.

Main features
=============

- Detailed analysis of the measurement geometry including automised retrieval of distances to the emission plume and/or to topographic features in the camera images (at pixel-level).
- Several routines for the retrieval of plume background intensities (either from plume images directly or using an additional sky reference image).
- Automatic analysis of cell calibration data.
- Correction for cross-detector variations in the SO2 sensitivity arising from wavelength shifts in the filter transmission windows.
- DOAS calibration routine including two methods to identify the field of view of a DOAS instrument within the camera images.
- Plume velocity retrieval either using an optical flow analysis or using signal cross correlation.
- Histogram based post analysis of optical flow field for gas velocity analysis in low contrast image regions, where the optical flow fails to detect motion.
- Routine for image based correction of the signal dilution effect based on contrast variations of dark terrain features located at different distances in the images.
- Support of standard image formats including `FITS format <https://de.wikipedia.org/wiki/Flexible_Image_Transport_System>`__.
- Easy and flexible setup for data import and camera specifications.



Requirements
============

Pyplis requires the following packages:

- numpy >= 1.11.0
- scipy >= 0.17.0
- opencv (cv2) >= 2.4.11 (please note `this issue <https://github.com/jgliss/pyplis/issues/4>`__)
- astropy >= 1.0.3
- geonum >= 1.2.0 (refer also to `geonum <https://github.com/jgliss/geonum>`__)

  - latlon23 >= 1.0.7
  - srtm.py >= 0.3.2
  - pyproj  >= 1.9.5.1
  - basemap >= 1.0.7

- pandas >= 0.16.2
- matplotlib >= 1.4.3

**Optional dependencies (to use extra features)**

- Pillow (PIL fork) >= 3.3.0

  - may be used to define custom image read functions, see e.g. `here <https://pyplis.readthedocs.io/en/latest/api.html#pyplis.custom_image_import.load_hd_new>`__
  - We recommend using ``pip install pillow`` rather than ``conda install pillow`` due to
  - well known installation issues, e.g. `here <https://github.com/python-pillow/Pillow/issues/2945>`__

- pydoas >= 1.0.0

Details about the installation of Pyplis and all requirements can be found in the following section.

We recommend using `Anaconda <https://www.continuum.io/downloads>`_ as package manager since it includes most of the required dependencies and is updated on a regular basis. Moreover, it is probably the most comfortable way to postinstall and upgrade dependencies such as OpenCV (`see here <http://stackoverflow.com/questions/23119413/how-to-install-python-opencv-through-conda>`__) or the scipy stack (for .

Please, if you have problems installing Pyplis, contact us or better, raise an Issue.

.. _install:

Installation instructions
=========================

Python installation
-------------------

We recommend using the `Anaconda Python 2.7 distribution <https://www.anaconda.com/distribution/>`__ (or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__, if you want to save disk space) and to use the *conda* package manager.

Below it is assumed that you made yourself familiar with the *conda* package manager and that it is installed on your system. It is recommended to have a look at the guidelines related to `conda virtual environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.

Installation of all *Pyplis* requirements
-----------------------------------------

Before installing *Pyplis*, you need to install all requirements. In order to do so, you have 2 options, either using the provided conda environment file or by installing all requirements manually, as described in the following two sections. All instructions below assume that you use `Anaconda <https://www.anaconda.com/>`__ as package manager and

Installation of requirements using provided conda environment file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install *all mandatory requirements is to use the provided environment file *pyplis_env_py27.yml*. This requires that the `conda` package manager is available. You can install the environment file either into a new environment (here, named *pyplis*) using::

  conda env create -n pyplis_env_test -f pyplis_env_py27.yml

Or you may install it into an existing environment using::

  conda env update -f=pyplis_env_py27.yml

Manual installation of requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may also install all requirements from scratch as described in the following step-by-step guide:

1. (Optional): Create new empty python 2.7 environment for pyplis
  ::

    conda create --name pyplis python=2.7

  Activate the new environment
  ::

    conda activate pyplis

2. Install scipy, pandas and astropy
  ::

    conda install scipy pandas astropy

3. Install basemap and OpenCV
  ::

    conda install -c conda-forge basemap opencv

  Note: this installs opencv version 4.

4. Install geonum
  ::
  
    conda install latlon23
    pip install SRTM.py
    pip install geonum

5. Install pydoas
  ::

    pip install pydoas

Installation of *pyplis*
^^^^^^^^^^^^^^^^^^^^^^^^

Here, you have two options.

- Option 1: Installation of latest `PyPi release <https://pypi.python.org/pypi/pyplis>`__
  ::

      pip install pyplis

- Option 2: Installation of latest development version
  ::

    Clone the `repository <https://github.com/jgliss/pyplis>`__ (green button "Clone or download") into a local directory of your choice. Unzip, and call
    ::

      python setup.py install

.. note::

  Use Option 2 if you want to run the tests and / or example scripts (since these are not shipped with the PyPi installation that uses a binary wheel of Pyplis).


Installation remarks and known issues
-------------------------------------

- If you work on a Windows machine and run into problems with installation of one of the requirements (e.g. if you already had Python 2.7 installed and want to upgrade dependencies such as numpy or scipy), check out the pre-compiled binary wheels on Christoph Gohlke's `webpage <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_

- Sometimes it is helpful, to reinstall your whole Python environment (or, if you use Anaconda, `create a new one <https://conda.io/docs/user-guide/tasks/manage-environments.html>`__) rather than trying to upgrade all dependencies to the required version

- If you find a bug or detect a specific problem with one of the requirements (e.g. due to future releases) please let us know or `raise an issue <https://github.com/jgliss/pyplis/issues>`__.


Getting started
===============

The Pyplis `example scripts <https://github.com/jgliss/pyplis/tree/master/scripts>`_ (see previous point) are a good starting point to get familiar with the features of Pyplis and for writing customised analysis scripts. The scripts require downloading the Etna example dataset (see following section for instructions). If you require more thorough testing, refer to this `wiki entry <https://github.com/jgliss/pyplis/wiki/Contribution-to-pyplis-and-testing>`__

Example and test data
=====================

The pyplis example data (required to run example scripts) is not part of the installation. It can be downloaded `here <https://folk.nilu.no/~arve/pyplis/pyplis_etna_testdata.zip>`__ or automatically downloaded in a Python shell (after installation) using::

  import pyplis
  pyplis.inout.download_test_data(<desired_location>)

which downloads the data into the *my_pyplis* directory if <desired_location> is unspecified. Else, (and if <desired_location> is a valid location) it will be downloaded into <desired_location> which will then be added to the supplementary file *_paths.txt* located in the installation *data* directory. It can then be found by the test data search method::

  pyplis.inout.find_test_data()

The latter searches all paths provided in the file *_paths.txt* whenever access to the test data is required. It raises an Exception, if the data cannot be found.

.. note::

  If the data is downloaded manually (e.g. using the link provided above), please make sure to unzip it into a local directory *<desired_location>* and let pyplis know about it, using::

    import pyplis
    pyplis.inout.set_test_data_path(<desired_location>)


.. _paper:

Scientific background
=====================

The article

*Pyplis–A Python Software Toolbox for the Analysis of SO2 Camera Images for Emission Rate Retrievals from Point Sources*, Gliß, J., Stebel, K., Kylling, A., Dinger, A. S., Sihler, H., and Sudbø, A., Geosciences, 2017

introduces *Pyplis* and implementation details. Furthermore, the article provides a comprehensive review of the technique of SO2 cameras with a focus on the required image analysis. The paper was published in December 2017 as part of a special issue on `Volcanic plumes <http://www.mdpi.com/journal/geosciences/special_issues/volcanic_processes>`__ of the Journal *Geosciences* (MDPI).
The paper can be downloaded `here <http://www.mdpi.com/2076-3263/7/4/134>`__.

Citation
--------
If you find *Pyplis* useful for your data analysis, we would highly appreciate if you acknowledge our work by citing the paper. Citing details can be found `here <http://www.mdpi.com/2076-3263/7/4/134>`__.


Copyright
=========

Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License a published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see `here <http://www.gnu.org/licenses/>`__.

.. note::

  The software was renamed from **piscope** to **Pyplis** on 17.02.2017
