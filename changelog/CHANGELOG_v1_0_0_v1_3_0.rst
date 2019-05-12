Release 1.0.0 -> 1.0.1
=======================================

This release includes only minor changes compared to the last one. These are mainly related to the access and handling / modelling of the plume background intensities in :class:`ImgList` objects.

25/11/2017 - 12/01/2018 (v1.0.0 -> v1.0.1)
------------------------------------------

1. Fixed some bugs

2. Added more tests.

3. Improved access of plume background images in :class:`ImgList` objects

Release 1.0.1 -> 1.3.0
================================

.. note::

  This release includes major API changes, performance improvements and bug fixes compared to version 1.0.1. Please update your installation as soon as possible.

Summary
-------

**Measurement geometry** (:class:`MeasGeometry`):

- more accurate plume distance retrievals (i.e. now also in dependency of vertical distance).
- redesigned API -> improved user-friendliness.

**Image analysis**: Image registration shift can now be applied to images.

- :func:`shift` in class Img.
- Comes with new *mode*  (:attr:`shift_mode`) in :class:`ImgList` objects.
- Default on / off shift for camera can be set in :class:`Camera` using attribute :attr:`reg_shift_off` (and correspondingly, in file *cam_info.txt*).

**Camera calibration**. Major improvements and API changes:

- new abstraction layer (:mod:`calib_base`) including new calibration base class. :class:`CalibData`: both :class:`DoasCalibData` and :class:`CellCalibData` are now inherited from new base class :class:`CalibData`. Advantages and new features:

  - arbitrary definition of calibration fit function.
  - fitting of calibration curve, I/O (read / write FITS) and visualisation of DOAS and cell calibration data are now unified in :class:`CalibData`.

**Further changes**

- :class:`ImgStack` more intuitive and flexible (e.g. dynamically expandable).
- Improved index handling and performance of image list objects (:mod:`imagelists`).
- :class:`PlumeBackgroundModel`: revision, clean up and performance improvements.
- Improved user-friendliness and performance of plume background retrieval in :class:`ImgList` objects.
- Correction for signal dilution (:class:`DilutionCorr`): increased flexibility and user-friendliness.
- Improved flexibility for image import using :class:`Dataset` class (e.g. on / off images can be stored in the same file).
- Reviewed and largely improved performance of general workflow (i.e. iteration over instances of :class:`ImgList` in ``calib_mode``, ``dilcorr_mode`` and ``optflow_mode``).

**Major bug fixes**

- Fixed conceptual error in cross-correlation algorithm for velocity retrieval (:func:`find_signal_correlation` in module :mod:`plumespeed`).
- Fixed: :class:`ImgList` in AA mode used current off-band image (at index ``idx_off``) both for the current and next on-band image (and not ``idx_off+1``).

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

    - Moved option ```LINK_OFF_TO_ON`` from :class:`Dataset` to :class:`BaseSetup` (no API changes in :class:`Dataset`)
    - New I/O option ``ON_OFF_SAME_FILE`` in :class:`BaseSetup` that can be set if on and off images are stored in one (e.g. FITS) file (like for the new USGS CVO camera type)
    - I/O options for data import can now be specified in file *cam_info.txt* for each camera individually using keyword ``io_opts`` and is stored as dict in :class:`CameraBaseInfo` (base class of :class:`Camera`)
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

  Changes related to camera calibration API (e.g. renaming, refactoring or removing of methods) are not resolved in full detail below (following point 1.).

1. Camera calibration

  - NEW MODULE :mod:`calib_base` containing new calibration base class :class:`CalibData` (both :class:`DoasCalibData` and :class:`CellCalibData` inherit from this base object)
  - MAIN CHANGES associated with with refactoring into general base class :class:`CalibData`

    - NEW FEATURE: Fit function for calibration data (both cell and DOAS) can now be defined arbitrarily (before, only polynomials were possible). See also module :mod:`model_functions`, in particular new class :class:`CalibFuns`
    - I/O (.e.g to / from FITS, or csv) are now unified for cell and DOAS calibration
    - Visualisation (e.g. plot of calibration curve and data) now unified for cell and DOAS calibration
    - New default fit function based on Kern et al. 2015
    - UNCERTAINTY treatment: Error in calibrated CDs is now computed based on the standard deviation of fit residual (if more than 10 datapoints are available for retrieval of calibration curve).

2. :class:`Img` object

  - Renamed attribute "alt_offset" -> "altitude_offset"
  - Moved custom import for ECII camera into new custom method :func:`load_ecII_fits` in module :mod:`custom_image_import`

  - New attributes:

    - :attr:`is_cropped`
    - :attr:`is_resized`
    - :attr:`is_shifted`

  - New methods:

    - :func:`shift` (applies x/y pixel shift of image)
    - :func:`convolve_with_mask`, for instance, when applied to an AA image, the input mask may be, e.g. a parameterised DOAS FOV (e.g. fitted 2D super-Gauss). The function then returns the weighted average AA within the FOV.
    - :func:`get_thresh_mask`

3. Image list classes (:mod:`imagelists`)

  - **New list mode** :attr:`shift_mode` (only for offband lists, i.e. lists with attribute ``type="off"``): activate / deactivate shift (dx, dy) of images on image load (cf. other list modes, such as :attr:`tau_mode`, :attr:`calib_mode`, :attr:`optflow_mode`). If activated, the default shift :attr:`reg_shift_off` of the assigned :class:`Camera` instance is used (is set (0, 0) if not explicitly defined (either in file *cam_info.txt* for a camera type (cf. cam "usgs" therein) or in instance of :class:`Camera` directly).

  - **Reviewed and optimised:**

    - :func:`correct_dilution` reviewed, largely rewritten and optimised

  - **New attributes:**

    - :attr:`update_cam_geodata` (default is ``False``). If True, the measurement geometry (i.e. plume distance) is automatically updated if image files contain camera geodata (e.g. lat, lon, viewing direction).

  - **New methods:**

    - :func:`calc_plumepix_mask` (for dilution correction)
    - :func:`timestamp_to_index` (returns list index corresponding to a datetime object)
    - :func:`_iter_num` (number of iterations to loop through the whole list, resulting from the total number of files :attr:`nof` and :attr:`skip_files`)

  - :func:`pop` now raises `NotImplementedError`
  - Introduced @property methods (getter / setter) for the attributes :attr:`skip_files` (newly introduced in v1.2.1, see above) and :attr:`edit_active` to ensure index update and reload (on change)

  - Further changes, deprecated, renamed

    - Introduced new input parameter `reload_here` in :func:`goto_img` (if True, :func:`load` is called even if the new index is the same as the current index, defaults to ``False``)
    - **Deprecated**:

      - Removed attribute :attr:`which_bg` (now handled automatically by @property attribute :attr:`bg_img`)

    - **Renamed**

      - :attr:`aa_corr_mask` -> :attr:`senscorr_mask`
      - :attr:`DARK_CORR_OPT` -> :attr:`darkcorr_opt`

  - Bug fixes:

    - Fixed: on-band list AA mode used current off-band image (at index ``idx_off``) both for the current and next on-band image (and not ``idx_off+1``).

4. Measurement geometry (:class:`MeasGeometry`, MAJOR API CHANGES)

  - Improved user-friendliness and performance: getter / setter methods for all attributes

    - Intended access / modification of attributes is via new getter / setter methods (e.g. ```geom.cam["lon"]`` -> ``geom.cam_lon``)
    - Comes with better handling of recomputation requirements of geometry in case individual parameters (e.g. camera viewing direction, position, wind direction) are updated (in this context, note new  attribute :attr:`update_cam_geodata` in :class:`ImgList` objects). Specifically:
    - Method :func:`update_geosetup` is called whenever a relevant attribute is updated via the corresponding setter method. This ensures, that derived values such as plume distance are always up-to-date with the current attributes.
    - Attribute dictionaries now private (e.g. ``.cam`` -> ``._cam``, ``.source`` -> ``._source``).

  - New methods:

    - :func:`get_topo_distance_pix` (determines distance to local topography in viewing direction of individual image pixel)

5. :class:`PlumeBackgroundModel` (Review and clean-up)

  - New attribute :attr:`last_tau_image`
  - New method :func:`_init_bgsurf_mask`: initiate mask for 2D background polynomial surface fit (only relevant for correction mode ``mode=0``)

  - **Removed**

    - dictionary :attr:`_current_imgs`: kept copies of input images (private dictionary)
    - Methods: :func:`get_current`, :func:`pyrlevel`, :func:`current_plume_background`, :func:`subtract_tau_offset`, :func:`_prep_img_type`, :func:`set_current_images`, :func:`plot_tau_result_old`

6. :class:`DilutionCorr`

  - Retrieval of extinction coefficients for dilution correction based on dark terrain features can now also be performed for individual pixel coordinates in the images, in addition to the distance retrieval based on lines in the images (see `example script 11 <http://pyplis.readthedocs.io/en/latest/examples.html#example-11-image-based-signal-dilution-correction>`__)
  - New methods:

    - :func:`add_retrieval_point`
    - :func:`add_retrieval_line`

7. Module :mod:`model_functions`:

  - New calibration fit function(s) based on `Kern et al., 2015 <https://www.sciencedirect.com/science/article/pii/S0377027314003783?via%3Dihub>`__
  - New class :class:`CalibFuns` for access of calibration fit functions

8. Plume velocity retrievals (:mod:`plumespeed.py`)

  - Cross correlation method (:func:`find_signal_correlation`)

    - Improved retrieval robustness: introduced percentage max shift that describes the maximum shift in percent of the second relative to the first time-series based on the total length of both series.
    - Fixed systematic retrieval error: Before, the second signal was rolled over the first, meaning, that the "end" of the 2. signal was attached to it's beginning and thus, correlated with the beginning of the first signal. That behaviour has been resolved.

  - Optical flow (:class:`OptflowFarneback` and :class:`FarnebackSettings`)

    - :attr:`i_min` (lower end of contrast range for optical flow calculation) can now also be smaller than 0.

9. I/O and setup classes (modules :mod:`inout` and :mod:`setupclasses`)

  - **my_pyplis** folder is now created on installation (in user home directory)

    - includes copies of *cam_info.txt* file and *my_sources.txt*

  - New method :func:`save_default_source` in :mod:`inout` (is saved in file *my_sources.txt*)
  - New method :func:`save_to_database` in :class:`Source` (wrapper method for :func:`save_default_source`)
  - New I/O option ``REG_SHIFT_OFF`` in classes :class:`BaseSetup` and :class:`MeasSetup`: if True (and if image lists are created using :class:`Dataset` and corresponding :class:`MeasSetup` object), then, the off-band images (in off-band :class:`ImgList`) are automatically shifted to on-band images (in on-band :class:`ImgList`) using the registration shift that is specified in :attr:`Camera.reg_shift_off` (can be set in file *cam_info.txt*)

10. Other changes

  - New method :func:`integrate_profile` in class :class:`LineOnImage`
  - New method :func:`make_circular_mask` in module :mod:`helpers.py`
  - In :mod:`fluxcalc` (and all included classes): renamed attr :attr:`cd_err_rel` to :attr:`cd_err` (note changes in uncertainty treatment of calibration data!)
  - :class:`EmissionRateSettings`: new option / attribute :attr:`min_cd_flow` (in addition to already existing :attr:`min_cd`) that may be used to explicitly define the minimum column-density of an image pixel for it to be considered valid with respect to `optical flow histogram analysis <https://www.atmos-meas-tech.net/11/781/2018/>`__ (before, the threshold :attr:`min_cd` was used). Is set equal :attr:`min_cd` if not explicitly specified
  - Moved class :class:`LineOnImage` into module :mod:`utils`
  - Moved method :func:`model_dark_image` from :mod:`processing` to :mod:`image` as well as class :class:`ProfileTimeSeriesImg`
  - Changed input parameter of :func:`model_dark_image` in :mod:`processing`
  - Changed default colormap for optical density (and calibrated) images from `bwr` to `viridis` (perceptually uniform)
  - **Major performance improvements**: reviewed typical workflow chain and removed irrelevant duplications of image arrays in certain objects ()
  - Fixed bugs
  - Included new tests (test suite still very low coverage...)
