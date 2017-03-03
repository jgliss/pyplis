**************************************************
Data import - Specifying custom camera information
**************************************************

.. note::

  In development, more information follows soon
  
In order to use all features of pyplis, certain specifications related to camera and image acquisition need to be defined. Basic information about the camera (e.g. detector specifics) and the corresponding file convention (image type, which data can be extracted from file names) are specified within :class:`pyplis.setupclasses.Camera` objects. This tutorial introduces the camera class and how to set up your custom camera type based on your data format, including definitions of your file naming convention.

How is the data imported?
=========================

At the very beginning of the analysis, the images need to be imported and separated by image type (e.g. on-band plume, off-band plume, dark, offset, on / off-band cell calibration). In order to use the automated separation for a given dataset (e.g. a single folder ``IMG_DIR``` containing images of all types) it is required that the image type can be identified from the file names.

The relevant information for identifying different image types (e.g. plume on-band, dark, offset) can be specified using either of the following two classes:

  1. :class:`Filter`: specifies file access information for all image types that are NOT dark or offset images (e.g. on / off images plume / background)
  #. :class:`DarkOffsetInfo`: specifies different types of dark images and offset images.
  
Such a collection of :class:`Filter` and :class:`DarkOffsetInfo` objects is then stored within a :class:`Camera` object. 

These information is used to separate the individual image types when creating a :class:`Dataset` object. The latter searches all valid image files in a given folder ``IMG_DIR`` and creates :class:`ImgList` objects for each :class:`Filter` and :class:`DarkImgList` objects for each :class:`DarkOffsetInfo` object defined in the :class:`Camera`. Each of these lists is then filled with the file paths of the corresponding image type located in ``IMG_DIR``. The :class:`Camera` object furthermore includes relevant specs of the camera (e.g. pixel geometry, lens).

The following list provides an overview of relevant parameters for filename access information using examplary filenames of the ECII camera type as well as the HD-Custom camera type.

Table: Example file naming conventions
======================================

.. note::

  Here follows an Excel table showing exemplary file naming conventions, currently it can not be built on RTD due to bug in module `sphinxcontrib-exceltable <https://bitbucket.org/birkenfeld/sphinx-contrib/issues/178/excel-table-not-working-with-sphinx-151>`_
  
Caption: Example file naming conventions for pyplis default cameras
  
In the following, all relevant :class:`Camera` parameters are introduced using the example of the ECII camera type.

Example 1: The ECII camera standard
===================================

Here, an exemplary :class:`Camera` is setup based on the ECII-camera type and file naming convention (cf. :ref:`ex02`).

