************************************
Specifying custom camera information
************************************

.. note::

  In development, more information follows soon
  
In order to use all features of pyplis, certain specifications related to camera and image acquisition need to be defined. Basic information about the camera (e.g. detector specifics) and the corresponding file convention (image type, which data can be extracted from file names) are specified within :class:`pyplis.setupclasses.Camera` objects. 

What information is needed?
===========================

Here follows a list of parameters needed in order to perform emission rate analysis using cams. This also includes data management information, i.e. how are images stored (file naming convention, image type) and how this information can be specified in order to be understood by pyplis.

How does pyplis import the data?
================================

Make the following points clear:

  1. What is specified in the Camera class
  2. Image type separation (on, off, dark, calibration data) happens via file naming convention, i.e. it should be possible to identify the type via the file name
  3. Image loading functions can be customised (e.g. to add import of meta information or to perform image preparation such as rotating)
  4. How is the data organised after import


The first step of each analysis is a well defined setup of the analysis data. When dealing with SO2 camera data, this means, specifying camera characteristics (e.g. pixel pitch, detector dimension, focal length) and file naming conventions. The latter are important in order to separate images of different types (e.g. on-band, off-band, dark) and potentially belonging to different time intervals (e.g. plume image data, cell calibration data).  

Typical data situation
======================
In the following, the relevant parameters are explained using an exemplary (fictional) file naming convention.
Consider a dataset of SO2 camera images stored within one directory ``IMG_DIR``. The folder contains all images (on, off, dark) recorded on one day (say 1/1/2017) of a field campaign. Let's assume the goal is to analyse a volcanic plume dataset recorded between 15:00 - 15:30 UTC of that day. Let's further say, cell calibration was performed right after that between 15:30 - 15:35 UTC. Your camera saves images in the following format::

  20170101151025345_onband_plume.tiff 
  20170101151025346_offband_plume.tiff
  20170101151025345_onband_calib.tiff 
  20170101151025346_offband_calib.tiff
  
to be continued ...


