This folder contains supplementary data for piscope
---------------------------------------------------

This includes

  1. Icons for GUI features
  #. file cam_info.txt (use that to create new default camera types)
  #. file my_sources.txt (use that to create new default sources)
  #. Dummy image (no_images_dummy.png)
  #. _paths.txt (file which is used to store local paths, e.g. to test data location)

Testdata information
--------------------

The piscope test data set (required to run example scripts) is not part of the piscope installation. It can be downloaded `here <https://folk.nilu.no/~gliss/piscope_testdata/>`_

The data can be downloaded automatically using::

  import piscope.inout
  piscope.download_test_data(<path>)
  
if <path> is unspecified, the data is downloaded at the installation location "./data/piscope_etna_testdata/", if <path> is a valid location it will be downloaded to the specified folder and the <path> will be added to the supplementary file "./data/_paths.txt", i.e. will be included to the test data search routine::

  piscope.inout.find_test_data()
  
which searches all valid test data folders and raises Exception, if the data cannot be found.
