**********************
Pyplis example scripts
**********************

Folder containing pyplis example scripts. Most of these scripts use the pyplis test dataset from Mt. Etna, which can be installed as described below.

Example and test data
=====================

The pyplis example data (required to run example scripts) is not part of the installation. It can be downloaded `here <https://folk.nilu.no/~gliss/pyplis_testdata/pyplis_etna_testdata.zip>`__ or automatically within a Python shell (after installation) using::

  import pyplis
  pyplis.inout.download_test_data(LOCAL_DIR)
  
which downloads the data to the installation **data** directory if ``LOCAL_DIR`` is unspecified. Else, (and if ``LOCAL_DIR`` is a valid location) it will be downloaded into ``LOCAL_DIR`` which will then be added to the supplementary file **_paths.txt** located in the installation **data** directory. It can then be found by the test data search method::

  pyplis.inout.find_test_data()
  
The latter searches all paths provided in the file **_paths.txt** whenever access to the test data is required. It raises an Exception, if the data cannot be found.

.. note::

  If you download the data manually (e.g. using the link provided above), please unzip it into a suitable directory ``LOCAL_DIR`` and let pyplis know about it using::
  
    import pyplis
    pyplis.inout.set_test_data_path(<LOCAL_DIR>)
