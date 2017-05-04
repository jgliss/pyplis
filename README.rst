Pyplis is a Python toolbox for the analysis of UV SO2 camera data. The software package includes a comprehensive collection of algorithms for the analysis of such data, for instance:

Main features
=============

- Support of many common image formats including (`FITS format <https://de.wikipedia.org/wiki/Flexible_Image_Transport_System>`__)
- Easy and flexible setup for data import and camera specifications 
- Detailed analysis of the measurement geometry including automized retrieval of distances to the emission plume and/or to topographic features in the camera images (down to pixel-level)
- Several routines for retrieval of plume background intensities (either from plume images directly or using an additional sky reference image).
- Automatic cell calibration
- Correction for cross-detector variations in SO2 sensitivity (due to wavelength shifts of filter transmission windows) using masks retrieved from cell calibration data
- DOAS calibration routine including two methods to identify the field of view of a DOAS instrument within the camera images
- Plume velocity retrieval either using an optical flow analysis or using signal cross correlation
- Histogram based post analysis of optical flow field for motion estimates in low contrast image regions (where optical flow fails to detect motion)
- Routine for image based light dilution correction
  
.. note::

  The software was renamed from **piscope** to **pyplis** on 17.02.2017 
  
Requirements
============

Requirements are listed ordered in decreasing likelyhood to run into problems when using pip for installation (on Windows machines you may use the pre-compiled binary wheels on Christoph Gohlke's `webpage <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_)

- numpy >= 1.11.0
- scipy >= 0.17.0
- opencv (cv2) >= 2.4.11
- Pillow (PIL fork) >= 3.3.0 (installs scipy.misc.pilutil)
- astropy >= 1.0.3
- geonum >= 1.0.0
    
  - latlon >= 1.0.2
  - srtm.py >= 0.3.2
  - pyproj  >= 1.9.5.1
    
- pandas == 0.16.2
- matplotlib >= 1.4.3

**Optional dependencies (to use extra features)**

- pydoas >= 1.0.0

We recommend using `Anaconda <https://www.continuum.io/downloads>`_ as package manager since it includes most of the required dependencies and is updated on a regular basis. Moreover, it is probably the most comfortable way to postinstall and upgrade dependencies such as OpenCV (`see here <http://stackoverflow.com/questions/23119413/how-to-install-python-opencv-through-conda>`__) or the scipy stack.

Installation
============

pyplis can be installed from `PyPi <https://pypi.python.org/pypi/pyplis>`_ using::

  pip install pyplis
  
or from source by downloading and extracting the latest release. After navigating to the source folder (where the setup.py file is located), call::

  python setup.py install

On Linux::
  
  sudo python setup.py install 
  
In case the installation fails make sure that all dependencies (see above) are installed correctly. pyplis is currently only supported for Python v2.7.


Code documentation
==================

The code documentation of pyplis is hosted `here <http://pyplis.readthedocs.io/en/latest/code_lib.html>`__. 

Getting started
===============

The Pyplis `example scripts <https://github.com/jgliss/pyplis/tree/master/scripts>`_ are a good starting point to get familiar with the features of Pyplis and for writing customised analysis scripts. The scripts require downloading the Etna example dataset (see following section for instructions).

Example and test data
=====================

The pyplis example data (required to run example scripts) is not part of the installation. It can be downloaded `here <https://folk.nilu.no/~gliss/pyplis_testdata/pyplis_etna_testdata.zip>`__ or automatically within a Python shell (after installation) using::

  import pyplis
  pyplis.inout.download_test_data(LOCAL_DIR)
  
which downloads the data to the installation **data** directory if ``LOCAL_DIR`` is unspecified. Else, (and if ``LOCAL_DIR`` is a valid location) it will be downloaded into ``LOCAL_DIR`` which will then be added to the supplementary file **_paths.txt** located in the installation **data** directory. It can then be found by the test data search method::

  pyplis.inout.find_test_data()
  
The latter searches all paths provided in the file **_paths.txt** whenever access to the test data is required. It raises an Exception, if the data cannot be found.

.. note::

  If the data is downloaded manually (e.g. using the link provided above), please make sure to unzip it into a local directory ``LOCAL_DIR`` and let pyplis know about it, using::
  
    import pyplis
    pyplis.inout.set_test_data_path(``LOCAL_DIR``)
    
    
TODO's
======

1. Write high level analysis class for signal cross correlation (currently only a low level method exists)
  
Future developments / ideas
===========================

1. Re-implementation of GUI framework
#. Include DOAS analysis for camera calibration by combining `pydoas <https://pypi.python.org/pypi/pydoas/1.0.1>`__ with `flexDOAS <https://github.com/gkuhl/flexDOAS>`__. 
#. Include online access to meteorological databases (e.g. to estimate wind direction, velocity)
