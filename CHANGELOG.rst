.. include:: ../CHANGELOG_v0_9_2_v0_11_2.rst
.. include:: ../CHANGELOG_v0_11_2_v1_0_0.rst

Release 1.0.0 -> 1.0.1
=======================================

This release includes only minor changes compared to the last one. These are mainly related to the access and handling / modelling of the plume background intensities in :class:`ImgList` objects.

25/11/2017 - 12/01/2018 (v1.0.0 -> v1.0.1)
------------------------------------------

1. Fixed some bugs 

#. Added more tests.

#. Improved access of plume background images in :class:`ImgList` objects

Release 1.0.1 -> 1.3.0
================================

.. note::

  This release includes major API changes, performance improvements and bug fixes compared to version 1.0.1. Please update your installation as soon as possible. 
  
1.0.1 -> 1.1.0
--------------

  1. :class:`Img` object 
  
    - Included read / write of meta info dictionary for FITS load / save
    - New method :func:`is_darkcorr`

  2. DOAS calibration (:mod:`doascalib`)
  
    - More flexible retrieval of DOAS calibration curves

  3. :class:`ImgStack`
    
    - new method :func:`ImgStack.sum`
    - Can now be dynamically extended (i.e. dynamic update of 3D array size). Corresponding API changes:
      
      - REMOVED: method :func:`append_img` 
      - NEW methods: :func:`insert_img`, :func:`add_img`, :func:`init_stack_array`
    
  4. Measurement geometry (:class:`MeasGeometry`)
  
    - More accurate plume distance retrieval (now also in vertical direction, cf. `Fig. 2 from example script 2 <https://github.com/jgliss/pyplis/blob/master/scripts/scripts_out/ex02_out_2.png>`__)
    
  5. Other changes
  
    - Moved Etna test data to new URL
    - Fixed bugs
    
1.1.0 -> 1.2.1
--------------

  1. :class:`Img` object
  
    - new method :func:`Img.sum` 
    
  2. Image list classes (:mod:`imagelists`, MAJOR API CHANGES)
  
    - Improved flexibility and clarity in index management
    - New attribute :attr:`skip_files` (i.e. load only every nth image from the filelist)
    - New method :func:`iter_indices`
    - Renamed method :func:`update_index_linked_lists` to :func:`change_index_linked_lists`
    - Removed method :func:`change_index`
    
  3. Plume background retrieval (:mod:`plumebackground`)
  
    - Getter / setter for attr. :attr:`surface_fit_mask` (ensure it is type :class:`Img`)
  
  4. Changes related to I/O
  
    - Moved option `LINK_OFF_TO_ON` from :class:`Dataset` to :class:`BaseSetup` (no API changes in :class:`Dataset`) 
    - New I/O option ``ON_OFF_SAME_FILE`` in :class:`BaseSetup` that can be set if on and off images are stored in one (e.g. FITS) file (like for the new USGS CVO camera type)
    - I/O options for data import can now be specified in file *cam_info.txt* for each camera individually using keyword ``io_opts`` 
    - Included I/O info for camera of USGS CVO (uses previous point)
    - Source info can now be saved automatically to file *my_sources.txt*
    
  5. Other changes
  
    - New method :func:`matlab_datenum_to_datetime` in :mod:`helpers`
    - Fixed bugs
    
1.2.1 -> 1.3.0
--------------

.. note::

  This version includes major refactoring and changes in API, aiming for more transparency and intuitive design. For instance, both the :class:`DoasCalibData` and :class:`CellCalibData` now inherit from a new base class :class:`CalibData` (in new module :mod:`calib_base`). 
  
.. note::

  Changes related to camera calibration API (e.g. renaming, refactoring or removing of methods) may not be resolved in full detail below (following point 1.). 
  
  1. Camera calibration
  
    - NEW MODULE :mod:`calib_base` containing new calibration base class :class:`CalibData` (both :class:`DoasCalibData` and :class:`CellCalibData` inherit from this base object)
    - MAIN CHANGES associated with with refactoring into general base class :class:`CalibData`
      
      - NEW FEATURE: Fit function for calibration data (both cell and DOAS) can now be defined arbitrarily (before, only polynomials were possible). See also module :mod:`model_functions`, in particular new class :class:`CalibFuns`
      - I/O (.e.g to / from FITS, or csv) are now unified for cell and DOAS calibration
      - Visualisation (e.g. plot of calibration curve and data) now unified for cell and DOAS calibration
      - New default fit function based on Kern et al. 2015
      - UNCERTAINTY treatment: Error in calibrated CDs is now computed based on the standard deviation of fit residual (if more than 10 datapoints are available for retrieval of calibration curve).
      
  2. Image list classes (:mod:`imagelists`)
  
    - New attribute :attr:`update_cam_geodata` (default is `False`). If True, the measurement geometry (i.e. plume distance) is automatically updated if image files contain camera geodata (e.g. lat, lon, viewing direction).
    - Method :func:`pop` now raises `NotImplementedError`
    - Introduced getter / setter method for new attribute :attr:`skip_files` (introduced in v1.2.1, see above) to ensure index update and reload (on change)
    
    
  3. Measurement geometry (:class:`MeasGeometry`, MAJOR API CHANGES)
  
    - Improved user-friendliness and performance: getter / setter methods for all attributes
    - Attribute dictionaries now private (e.g. `.cam` -> `._cam`, `.source` -> `._source`). Intended of attributes is via new getter / setter methods (e.g. `geom.cam["lon"]` -> geom.cam_lon)
    - Method :func:`update_geosetup` is called whenever a relevant attribute is updated via the corresponding setter method. This ensures, that derived values such as plume distance are always up-to-date with the current attributes.
    - New method :func:`get_topo_distance_pix` (determines distance to local topography in viewing direction of individual image pixel)
    
  4. :class:`Img` object
  
    - Renamed attribute "alt_offset" -> "altitude_offset"
    - New methods / properties / attributes:
    
      :attr:`is_cropped`
      :attr:`is_resized`
    
  X. Other changes
  
    - In :mod:`fluxcalc` (and all included classes): renamed attr :attr:`cd_err_rel` to :attr:`cd_err` (note changes in uncertainty treatment of calibration data!)
    - Retrieval of extinction coefficients for dilution correction based on dark terrain features can now also be performed for individual pixel coordinates in the images, in addition to the distance retrieval based on lines in the images (see `example script 11 <http://pyplis.readthedocs.io/en/latest/examples.html#example-11-image-based-signal-dilution-correction>`__)
    - Improved performance and user-friendliness of :class:`MeasGeometry` (e.g. getter / setter methods for all parameters, better handling of recomputation of geometry in case, parameters are updated)
    - Moved class :class:`LineOnImage` into module :mod:`utils`
    - Moved method :func:`model_dark_image` from :mod:`processing` to :mod:`image` as well as class :class:`ProfileTimeSeriesImg`
    - Changed input parameter of :func:`model_dark_image` in :mod:`processing`
    - Changed default colormap for optical density (and calibrated) images from `bwr` to `viridis` (perceptually uniform)
    - Fixed bugs
    - Included new tests (test suite still very low coverage...)
    
    