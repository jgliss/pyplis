piscope is a Python toolbox for the analysis of UV SO2 camera data. It includes a large collection of routines for the analysis of such data, for instance:

  1. Several routines for plume background estimation
  #. Automatic cell calibration 
  #. DOAS calibration routine including two methods to identify the field of view of a DOAS instrument within the camera images
  #. Plume velocity retrieval either using an optical flow analysis or using signal cross correlation
  #. Detailed analysis of the measurement geometry including automized retrieval of distances to the emission plume and/or to topographic features in the camera images (on a pixel basis)
  #. Routine for image based light dilution correction

Requirements
------------

Requirements are listed in the order of likelyhood to run into problems when using pip for installing them (on Windows machines you may use the pre-compiled binary wheels on Christoph Gohlke's `webpage <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_)

  - numpy >= 1.11.0
  - scipy >= 0.17.0
  - opencv (cv2) >= 2.4.11
  - Pillow (PIL fork) >= 3.3.0 (installs scipy.misc.pilutil)
  - astropy >= 1.0.3
  - geonum >= 1.0.0
    - latlon >= 1.0.2
    - srtm.py >= 0.3.2
    - pyproj  >= 1.9.5.1
  - pandas >= 0.16.2
  - matplotlib >= 1.4.3

**Optional dependencies (to use extra features)**

  - pydoas >= 1.0.0
  - scikit-image (skimage) >= 0.11.3 (for blob detection in optical flow analysis)
  

We recommend using `Anaconda <https://www.continuum.io/downloads>`_ as package manager since it includes most of the required dependencies and is updated on a regular basis. Moreover, it is probably the most comfortable way to postinstall and upgrade dependencies such as OpenCV (`see here <http://stackoverflow.com/questions/23119413/how-to-install-python-opencv-through-conda>`_) or the scipy stack.

Installation
------------

piscope can be installed from source by downloading and extracting the latest release. After navigating to the source folder (where the setup.py file is located), call::

  python setup.py install

On Linux::
  
  sudo python setup.py install 
  
In case the installation fails make sure that all dependencies (see above) are installed correctly. piscope is currently only supported for Python v2.7.

Instructions and code documentation
-----------------------------------

Coming soon ...

Getting started
---------------

After installation try running and understanding the example scripts (can be found in the scripts folder). The scripts require downloading the example data (see following section).

Example and test data
---------------------

The piscope example data (required to run example scripts) is not part of the installation. It can be downloaded `here <https://folk.nilu.no/~gliss/piscope_testdata/piscope_etna_testdata.zip>`_

or automatically after installation using::

  import piscope
  piscope.inout.download_test_data(<local_path>)
  
which downloads the data to the installation *data* directory if *<local_path>* is unspecified. If <local_path> is a valid location it will be downloaded to the specified folder and <local_path> will be added to the supplementary file "./data/_paths.txt", i.e. it will be added as default search path to the test data search method::

  piscope.inout.find_test_data()
  
which searches all valid test data folders and raises Exception, if the data cannot be found.

