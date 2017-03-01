Changelog
=========

This file keeps track of major changes applied to the code after the first 
release (version 0.9.2)

28/02/2017 - 01/03/2017
-----------------------

  1. Allowing for defining custom image import method (file custom_image_import.py)
  2.  Fixed bug regarding assignment of dark / offset lists for HD-Custom camera: if multiple type dark (and / or offset) image lists exist (based on camera file naming convention after file separation in Dataset), then the list with the shortest difference in the exposure time is set (using the first image of the respective image lists for intercomparison)
  3. Expanded handling of start / stop time specifications in Setup classes and Dataset classes (for initialisation of working environment) -> now, the user can also provide time stamps (datetime.time objects) or dates (datetime.date objects) as input and they will converted automatically to datetime. Also, if the default date is set (i.e. 1/1/1900) in a Setup, it will be disregarded and only the time stamps HHMMSS are considered to identify image files belonging to specified start /stop interval.
  4. Minor convenience changes in Dataeset enabling to set attributes and options of MeasSetup class directly from Dataset class using @property decorators (e.g. for start, stop, base_dir, etc.)
  5. Updated specs for HD-Custom camera such that cell calibration data can be imported
  6. Expanded functionality for dark and offset list assignments in Dataset and ImgList objects. 
  
    1. Master dark / offset images are now searched for each dark / offset image list individually
    2. Customised assignment of dark / offset lists in image lists for cameras where meas type is specified in own filename substring (e.g. HD cam). 
  