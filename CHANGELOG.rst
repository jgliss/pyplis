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

  1. Improved performance in long "for" loops (e.g. :func:`make_stack`, :func:`get_mean_value` in :class:`BaseImgList` or file searching methods in :class:`Dataset`  by removing ``self.`` operations within the loops)
  #. :class:`EmissionRateResults` can now be saved as text file and has some new methods, the most important ones are: 
  
    - :func:`__str__`: informative string representation
    - :func:`to_dict`: converts results to dictionary
    - :func:`to_pandas_dataframe`: converts object into pandas :class:`DataFrame` class
    - :func:`from_pandas_dataframe`: imports data from pandas :class:`DataFrame` class
    - :func:`save_txt`: save results as text file
    
  #. Updated options for xlabel formatting when plotting time series
  #. Improved optical flow histogram analysis
    
    - Renamed settings param ``sigma_tol_mean_dir`` to ``hist_dir_sigma``
    - New: choose from two options for retrieval of average displacement length from length histogram (in :func:`get_main_flow_field_params` of :class:`OpticalFlowFarneback`):
    
      - "argmax": uses bin with largest count as mean displacement estimate (new)
      - "multigauss": tries to perform :class:`MultiGaussFit` to data and if this fails, uses method "argmax"
  
    - new global settings parameters for maximum number of fitted gaussians in both orientation and length histogram, can now be set via :class:`OpticalFlowFarnebackSettings`

08/03/2017
==========

  1. New functions in ``Img`` class:
    
    - ``to_binary`` and corresponding entry ``is_bin`` in ``edit_log``` dict. 
    - ``dilate``: apply morphological transform *dilation* to image using method ``cv2.dilate``
    - ``invert``: inverts an image object (added entry in edit_log)
  
  #. New method ``get_mean_img`` in image list classes: determines average image based on start / stop index (or for all images in list)
  #. Removed bug in ``Img`` method ``to_pyrlevel`` for going up in pyramid
  
09/03/2017
==========

  1. Class ``Dataset`` objects can now be initiated with differnt ``ImgList`` types (e.g. ``CellCalibEngine`` is now initiated with ``CellImgList`` objects)
  #. implementation of method to apply dilution correction to an plume image ``correct_img`` moved as global method in ``dilutioncorr.py`` and is now a wrapper in ``DilutionCorr`` class.
  #. New methods in ``DilutionCorr`` class:
  
    - ``get_ext_coeffs_imglist``: retrieve extinction coefficients for all images in an :class:`ImgList` object.

13/03/2017
==========

  1. New functions in ``ImgList``:
  
      - :func:`get_thresh_mask`: get mask based on intensity threshold (e.g. tau thresh)
      - :func:`prepare_bg_fit_mask`: (BETA) for background modelling mode 0 (PolySurfaceFit). Determines a mask specifying background pixels based on intensities in reference rectangles and optical flow analysis (slow).
      - :func:`correct_dilution`: correct current image for signal dilution
      - :func:`set_bg_img_from_polyfit`: determines an *initial* background image in list using ``PolySurfaceFit`. The result is set as new ``bg_img`` attribute, i.e. is used for backrgound modelling in modes 1-6. This can be done if no measured sky radiance image is available.
      - :func:`correct_dilution`: applies Dilution correction to current image if all requirements are fulfilled for that
      - start / stop indices can now be set in :func:`make_stack`
      
  2. Removed automatic load of previous image in ``ImgList`` objects
  3. Included AA image calculation for CORR_MODE == 0 in ``PlumeBackgroundModel``.
  4. Removed dark corr check between plume and BG image in ``PlumeBackgroundModel`` when modelling tau images.