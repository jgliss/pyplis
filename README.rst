Pyplis is a Python toolbox for the analysis of UV SO2 camera data. The software includes a comprehensive and flexible collection of algorithms for the analysis of such data.

Contact: Jonas Gliss (jonasgliss@gmail.com)

.. note::

  The software was renamed from **piscope** to **Pyplis** on 17.02.2017

.. _news:

News
====

**Version 1.3.0 released (21/05/2018)**
---------------------------------------

.. note::

  *Pyplis* version 1.3.0 comes with **many new features** and **major improvements in user-friendliness and performance** (compared to previous release 1.0.1)

  Please update your version as soon as possible, e.g. using::

    pip install pyplis

  or::

    git clone https://github.com/jgliss/pyplis.git -b v1.3.0

  For detailed installation instructions see detailed installation instructions below.

In the following, a brief overview is provided over the most important changes associated with this release. Please see latest `changelog <file:///C:/Users/Jonas/repos/pyplis/docs/_build/html/changelog.html#release-1-0-1-1-3-0>`__ for a detailed description of all changes.

.. _release_1.3.0:

**Release notes (v1.3.0)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pyplis version 1.3.0 comes with many new features and improvements. The most important changes include:

**Measurement geometry** (``MeasGeometry``):

- more accurate plume distance retrievals (i.e. now also in dependency of vertical distance).
- redesigned API -> improved user-friendliness.

**Image analysis**: Image registration shift can now be applied to images.

- :func:`shift` in class Img.
- Comes with new *mode*  (:attr:`shift_mode`) in :class:`ImgList` objects.
- Default on / off shift for camera can be set in :class:`Camera` using attribute :attr:`reg_shift_off` (and correspondingly, in file *cam_info.txt*).

**Camera calibration**. Major improvements and API changes:

- new abstraction layer (:mod:`calib_base`) including new calibration base class. :class:`CalibData`: both :class:`DoasCalibData` and :class:`CellCalibData` are now inherited from new base class :class:`CalibData`. Advantages and new features:

  - arbitrary definition of calibration fit function.
  - fitting of calibration curve, I/O (read / write FITS) and visualisation of DOAS and cell calibration data are now unified in :class:`CalibData`.

**Further changes**

- :class:`ImgStack` more intuitive and flexible (e.g. dynamically expandable).
- Improved index handling and performance of image list objects (:mod:`imagelists`).
- :class:`PlumeBackgroundModel`: revision, clean up and performance improvements.
- Improved user-friendliness and performance of plume background retrieval in :class:`ImgList` objects.
- Correction for signal dilution (:class:`DilutionCorr`): increased flexibility and user-friendliness.
- Improved flexibility for image import using :class:`Dataset` class (e.g. on / off images can be stored in the same file).
- Reviewed and largely improved performance of general workflow (i.e. iteration over instances of :class:`ImgList` in ``calib_mode``, ``dilcorr_mode`` and ``optflow_mode``).

**Major bug fixes**

- Fixed conceptual error in cross-correlation algorithm for velocity retrieval (:func:`find_signal_correlation` in module :mod:`plumespeed`).
- Fixed: :class:`ImgList` in AA mode used current off-band image (at index ``idx_off``) both for the current and next on-band image (and not ``idx_off+1``).

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

Copyright
=========

Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License a published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see `here <http://www.gnu.org/licenses/>`__.

Code documentation and more
============================

The code documentation of Pyplis and more information is hosted on `Read the Docs <http://pyplis.readthedocs.io/en/latest/index.html>`__.

Requirements
============

Pyplis requires the following packages:

- numpy >= 1.11.0
- scipy >= 0.17.0
- opencv (cv2) >= 2.4.11 (please note `this issue <https://github.com/jgliss/pyplis/issues/4>`__)
- Pillow (PIL fork) >= 3.3.0 (installs scipy.misc.pilutil)
- astropy >= 1.0.3
- geonum >= 1.2.0

  - latlon >= 1.0.2
  - srtm.py >= 0.3.2
  - pyproj  >= 1.9.5.1
  - basemap >= 1.0.7

- pandas >= 0.16.2
- matplotlib >= 1.4.3

**Optional dependencies (to use extra features)**

- pydoas >= 1.0.0

Details about the installation of Pyplis and all requirements can be found in the following section.

We recommend using `Anaconda <https://www.continuum.io/downloads>`_ as package manager since it includes most of the required dependencies and is updated on a regular basis. Moreover, it is probably the most comfortable way to postinstall and upgrade dependencies such as OpenCV (`see here <http://stackoverflow.com/questions/23119413/how-to-install-python-opencv-through-conda>`__) or the scipy stack (for .

Please, if you have problems installing Pyplis, contact us or better, raise an Issue.

.. _install:

Installation instructions
=========================

In the following, a step-by-step guide for the installation on *Pyplis* is provided. It is assumed, that no Python 2.7 installation exists. If you already have Python 2.7 installed (with potentially some packages), make sure you install `all requirements <https://github.com/jgliss/pyplis#requirements>`__.
If you use `Anaconda <https://www.anaconda.com/>`__ as package manager, you can check your installed packages using::

  conda list

Else, you can use ``pip`` to check your package list::

  pip freeze


Install from scratch
--------------------

If you already have Anaconda2 installed on your machine you can skip point 1., else:

1. Download and install the latest version of `Anaconda2 <https://www.anaconda.com/download/#windows>`__ (Python 2.7)

2. Install basemap
  ::

    conda install -c conda-forge basemap

3. Install opencv version 2
  ::

    conda install -c menpo opencv

4. Install `Geonum <https://github.com/jgliss/geonum>`__
  ::

    pip install geonum

5. Install `Pydoas <https://github.com/jgliss/pydoas>`__
  ::

    pip install pydoas

6. Install Pyplis. Here, you have two options.

  - Option 1: Installation using `PyPi <https://pypi.python.org/pypi/pyplis>`__
    ::

      pip install Pyplis

  - Option 2: Installation from source

    Download `the latest release <https://github.com/jgliss/pyplis/releases>`__ or the latest (not released) version of the `repository <https://github.com/jgliss/pyplis>`__ (green button "Clone or download") into a local directory of your choice. Unzip, and call
    ::

      python setup.py install

.. note::

  Use Option 2 if you want to run the tests and / or example scripts (since these are not shipped with the PyPi installation that uses a binary wheel of Pyplis).

After installation, try::

  >>> import pyplis

from your Python or IPython console.

Installation remarks and known issues
-------------------------------------

  - If you work on a Windows machine and run into problems with installation of one of the requirements (e.g. if you already had Python 2.7 installed and want to upgrade dependencies such as numpy or scipy), check out the pre-compiled binary wheels on Christoph Gohlke's `webpage <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_

  - Sometimes it is helpful, to reinstall your whole Python environment (or, if you use Anaconda, `create a new one <https://conda.io/docs/user-guide/tasks/manage-environments.html>`__) rather than trying to upgrade all dependencies to the required version

  - If you find a bug or detect a specific problem with one of the requirements (e.g. due to future releases) please let us know or `raise an issue <https://github.com/jgliss/pyplis/issues>`__.

Testing your installation
=========================

.. note::

  The following steps can only be done if download and install from source (Option 2, previous point) and do not work if you install via pip.

Running tests
-------------

Pyplis contains a (currently incomplete) test suite (located `here <https://github.com/jgliss/pyplis/tree/master/pyplis/test>`__.

The tests can be run manually from the toplevel directory (where the setup.py file lies) using your *command line* (not Python console) using::

  python -m pytest

If any test fails, please `raise an issue <https://github.com/jgliss/pyplis/issues>`__.

Running the pyplis Etna example scripts
---------------------------------------

In order to run the Etna example scripts, you have to download the Etna test dataset (about 2.7 GB). You can download the testdata automatically into a specified folder <desired_location>::

  >>> import pyplis
  >>> pyplis.inout.download_test_data(<desired_location>)

If you leave <desired_location> empty, the data will be downloaded into the *my_pyplis* folder, that is automatically created on installation in your user home directory (`more details below <https://github.com/jgliss/pyplis#example-and-test-data>`__).

The scripts can be found in the *scripts* folder of the repo. They include a test mode (can be activated in `SETTINGS.py <https://github.com/jgliss/pyplis/blob/master/scripts/SETTINGS.py>`__ or on script execution via command line  using option --test 1, see below) and can be run automatically from the command line by executing the following two scripts::

  python RUN_INTRO_SCRIPTS.py --test 1

and::

  python RUN_EXAMPLE_SCRIPTS.py --test 1

Getting started
===============

The Pyplis `example scripts <https://github.com/jgliss/pyplis/tree/master/scripts>`_ (see previous point) are a good starting point to get familiar with the features of Pyplis and for writing customised analysis scripts. The scripts require downloading the Etna example dataset (see following section for instructions).

Example and test data
=====================

The pyplis example data (required to run example scripts) is not part of the installation. It can be downloaded `here <https://folk.nilu.no/~arve/pyplis/pyplis_etna_testdata.zip>`__ or automatically within a Python shell (after installation) using::

  import pyplis
  pyplis.inout.download_test_data(<desired_location>)

which downloads the data into the *my_pyplis* directory if <desired_location> is unspecified. Else, (and if <desired_location> is a valid location) it will be downloaded into <desired_location> which will then be added to the supplementary file **_paths.txt** located in the installation **data** directory. It can then be found by the test data search method::

  pyplis.inout.find_test_data()

The latter searches all paths provided in the file **_paths.txt** whenever access to the test data is required. It raises an Exception, if the data cannot be found.

.. note::

  If the data is downloaded manually (e.g. using the link provided above), please make sure to unzip it into a local directory ``LOCAL_DIR`` and let pyplis know about it, using::

    import pyplis
    pyplis.inout.set_test_data_path(<desired_location>)


Future developments / ideas
===========================

1. Re-implementation of GUI framework
2. Include DOAS analysis for camera calibration by combining `pydoas <https://pypi.python.org/pypi/pydoas/1.0.1>`__ with `flexDOAS <https://github.com/gkuhl/flexDOAS>`__.
3. Include online access to meteorological databases (e.g. to estimate wind direction, velocity)
