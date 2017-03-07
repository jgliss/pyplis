This file keeps track of major changes applied to the code after the first 
release (version 0.9.2)

28/02/2017 - 01/03/2017
=======================

  1. Allowing for defining custom image import method (file custom_image_import.py)
  2.  Fixed bug regarding assignment of dark / offset lists for HD-Custom camera: if multiple type dark (and / or offset) image lists exist (based on camera file naming convention after file separation in Dataset), then the list with the shortest difference in the exposure time is set (using the first image of the respective image lists for intercomparison)
  3. Expanded handling of start / stop time specifications in Setup classes and Dataset classes (for initialisation of working environment) -> now, the user can also provide time stamps (datetime.time objects) or dates (datetime.date objects) as input and they will converted automatically to datetime. Also, if the default date is set (i.e. 1/1/1900) in a Setup, it will be disregarded and only the time stamps HHMMSS are considered to identify image files belonging to specified start /stop interval.
  4. Minor convenience changes in Dataeset enabling to set attributes and options of MeasSetup class directly from Dataset class using @property decorators (e.g. for start, stop, base_dir, etc.)
  5. Updated specs for HD-Custom camera such that cell calibration data can be imported
  6. Expanded functionality for dark and offset list assignments in Dataset and ImgList objects. 
  
    1. Master dark / offset images are now searched for each dark / offset image list individually
    2. Customised assignment of dark / offset lists in image lists for cameras where meas type is specified in own filename substring (e.g. HD cam). 
    
02/03/2017
==========

  1. Included new default camera type "hd_new" (2. camera from group in Heidelberg, Germany). Currently missing detector and optics specs
  #. Expanded flexibility for meta information access via filename for acquisition time, meas_type and filter_id in Camera class: now, the conversion / identification strings can also include the actual delimiter (e.g. delim="_", time_info_pos = 0, time_info_str="%Y%m%d_%H%M%S_%f" or filter_id_pos=3 and filter.acronym="A_dark"). This is for instance required for file naming convention of new default SO2 camera type "hd_new".
  #. Improved functionality for dark and offset image access in ImgList classes
  #. Improved data import speed in Dataset -> search of master_dark image is only applied to lists that actually include image data
  
03/03/2017
==========

  1. Included image check for negative numbers or zeros after dark image correction and before tau / AA image calculation: correction is directly applied to images (no warning), i.e. pixels <= 0 are set to smallest positive float of system.
  2. Removed bugs regarding image time stamps in MeasSetup and image match search in Dataset (when specifying start / stop time stamps are provided as time object and not as datetime object). These two bugs resulted from changes applied in 0.9.3.dev1 (1/3/2017) and are irrelevant for previous versions.
  
05/03/2017
==========

.. note::

  Detected and fixed bug related to signal cross correlation based plume velocity retrievals after pandas updgrade from 0.16.2 -> 0.19.2.
  
06/03/2017
==========

  1. Removed bug in :class:`ImgStack` method ``merge_with_time_series``: generalised catch of first exception (applies if ``doas_series`` is pandas ``Series`` object and not pydoas ``DoasResults``).
  
07/03/2017
==========

  1. Improved performance in long for loops (e.g. :func:`make_stack`, :func:`get_mean_value` in :class:`BaseImgList` or file searching methods in :class:`Dataset`  by removing ``self.`` operations within the loops)
  2. :class:`EmissionRateResults` can now be saved as text file and has some new methods, the most important ones are: 
  
    - :func:`__str__`: informative string representation
    - :func:`to_dict`: converts results to dictionary
    - :func:`to_pandas_dataframe`: converts object into pandas :class:`DataFrame` class
    - :func:`from_pandas_dataframe`: imports data from pandas :class:`DataFrame` class
    - :func:`save_txt`: save results as text file
    
  2. Updated options for xlabel formatting when plotting time series
  
  
  
  