Release 0.11.2 -> 1.0.0
=======================

30-31/03/2017
-------------

1. Renamed :attr:`roi_rad` in Farneback classes to ``roi_rad_abs`` (makes it clearer in which coordinates it is supposed to be defined). The old name also still works but a warning is given if used.

2. Renamed :attr:`hist_dir_sigma` in Farneback classes to ``hist_sigma_tol`` since it is applied both to the fit result of the main peak of the orientation histogram but also to determine the uncertainty in the length histogram from the moments analysis. The old name also still works but a warning is given if used.

3. More features in :class:`LineOnImage`

  - Global velocity estimates (and uncertainties) can now be assigned (e.g. for emission rate analysis)
  - :class:`LocalPlumeProperties` can now be assigned (e.g. for emission rate analysis and if a time series of local displacement vectors for velocity retrieval was calculated beforehand and the results are accessible in a text file)).

01-07/04/2017
-------------

1. Added features in :class:`LocalPlumeProperties`

  - Interpolation can now also be performed onto index array of other time series (e.g. image list time stamps)
  - New method :func:`apply_significance_thresh` in :class:`LocalPlumeProperties`: can be used to remove data points with significance lower than provided thresh (and combined e.g. with interpolation).
  - New method :func:`to_pyrlevel`: converts displacement lengths (and errors) to a given Gauss pyramid level.

2. Expanded functionality in classes :class:`EmissionRateAnalysis` and :class:`EmissionRateSettings`:

  - global velocities can now be assigned for each PCS line individually (need to be assigned in the :class:`LineOnImage` objects directly).

  - :class:`LocalPlumeProperties` assigned to retrieval lines (:class:`LineOnImage` objects) in :class:`EmissionRateAnalysis` are now considered for analysis when using ``velo_mode_farneback_histo`` (e.g. they might be calculated and post processed beforehand).

  - New velocity method ``farneback_hybrid`` for velocity retrievals: uses optical flow output along line modified such that vectors that are not in expectation range (retrieved from histo analysis) are replaced by the average flow vector from the histo analysis.

3. New attribute ``residual`` in :class:`DoasCalibData`

4. Fixed some bugs related to scale space conversion in :class:`ImgList` objects (e.g. related to activation of ``tau_mode``, dilution correction)

5. Corrected bug related to SO2-uncertainty based on slope error of calibration curve from covariance matrix of poly fit. Previously: used value of slope error as measure of uncertainty (wrong), now: use relative error, e.g. calibration curve zero y-axis offset and with slope, slope err: ``m=1e19, m_err=1e17`` then the mapped SO2 error (for a given tau value ``tau0``) is determined as :``so2 = tau0 * m`` and ``so2_err = so2 * m_err / m``

6. Added mathematical operators to ``EmissionRateResults`` class

  - __add__: use "+" operator to add results (e.g. retrieved at two different lines from two crater emissions)
  - __sub__: use "-" operator to subtract results (e.g. retrieved at two different positions downwind of the crater emissions)


10/04/2017
----------

1. Added option in :func:`make_stack` in :class:`ImgList` objects: the method includes now the option to specify a reference ROI in the image (e.g. sky reference area) and a corresponding min / max range for the expectation value in that range: if the input is specified, then only images are added to the stack that are within the specified range within the ROI.

2. New features in :class:`EmissionRateAnalysis` and :class:`EmissionRateSettings`

  - Added same feature (as described in 1.) to emission rate retrieval classes, relevant attributes in ``EmissionRateSettings`` class are:

    - ``ref_check_mode``: activate / deactivate the new mode
    - ``bg_roi_abs`` (ROI used for check)
    - ``ref_check_lower_lim``: lower intensity limit
    - ``ref_check_upper_lim``: upper intensity limit

  - Moved attr. ``bg_roi`` from analysis class to settings class and renamed to ``bg_roi_abs``.


11/04/2017
----------

1. Added check of date information in :func:`get_img_meta_all_filenames` of :class:`ImgList` which is, for instance, used for accessing datetime information of acq. times of all images in the list: a problem may occur if the file names only include information of acq. times of the images but not dates.  Then, the retrieved timestamps (numpy array of datetime objects) will only include acq. times of all images and the default date: 1/1/1900. If this is the case, then the method replaces these default dates in the array using the date stored in the meta header of the currently loaded image in the the list. This is, for instance relevant for the HD default camera which includes date information in the tiff header (will be loaded and stored in meta header of ``Img`` class on load, but not in the file names).

12/04 - 04/05/2017
------------------

1. Minor changes in plot style for standard outputs

#. Worked on docs

04/05 - 21/05/2017 (v0.11.4 -> v0.12.0)
---------------------------------------

.. note::

  Not downwards compatible change in :mod:`fluxcalc.py`: changed name of velocity retrieval modes and functions related to optical flow from e.g. ``farneback_hybrid`` to ``flow_hybrid``.

1. Minor improvements in documentation of example scripts

#. Changes in docs

#. Minor changes in plot style for standard outputs

#. DOAS calibration polynomial is now fitted only using mantissa of the CDs (to avoid large number warning in polyfit)

#. Changes in optimisation strategy for optical flow histogram analysis and correction (modules: :mod:`plumespeed.py`, :mod:`fluxcalc.py`)

  1. Minimum required length (per line and image is set at lower end of 1sigma of expectation interval of histo analysis

  #. More sophisticated uncertainty analysis for effective velocities

#. Changed all names in :mod:`fluxcalc.py` related to optical flow based velocity retrievals which included ``farneback`` to ``flow`` (not downward compatible)

#. New class ``EmissionRateRatio`` in :mod:`fluxcalc`

22/05 - 29/08/2017 (v0.11.4 -> v0.12.0)
---------------------------------------

1. Minor bug fixes

#. Added functionality to :class:`Img` objects

#. DOAS calibration data can now be fitted using weighted regression based on DOAS fit errors. Note, that new default is weighted fitting, if applicable (i.e. if uncertainties are available).

#. New class :class:`VeloCrossCorrEngine` in :mod:`plumespeed.py` for high level computing of cross correlation based velocity retrievals. Note that this includes changes in example script 8, which now uses the new class. Thus, running the current version of example script 8 will not work with older versions of pyplis.

#. Started with implementation of test suite using pytest

30/08 - 05/10/2017 (v0.12.0 -> v0.13.4)
---------------------------------------

1. Minor bug fixes

#. Improved convenience functionality of classes in :mod:`doascalib` by adding some @property decorators.

#. New high-level default method :func:`run_fov_fine_search` in :class:`DoasFOVEngine`

#. Renamed key vor wind velocity (and error) in :class:`MeasGeometry` from "vel" to "velo"

#. New method :func:`find_movement` in :mod:`plumespeed`. The method performs an iterative computation of the optical flow between two images under variation of the considered input brightness ranges.

#. Improved functionality for automated retrieval of sky-background pixels in an plume image (now uses new method :func:`find_movement` to identify and exclude pixels showing motion.

5/10/2017 - 25/11/2017 (v0.13.4 -> v1.0.0)
------------------------------------------

1. Fixed some bugs

#. Started with setting up a test-suite (available in the GitHub repo but not yet included in standard installation of the code)

#. Added test-dataset of size reduced images from the Etna testdat (mainly for tests. This dataset is not yet included in the standard installation

# Automatic SRTM access can now be deactivated in :class:`MeasGeometry` objects

#. Made MultiGaussFit optional for histogram post analysis of optical flow

#. Removed requirement for :mod:`progressbar`

#. Changed colour and plot styles in some of the standard plotting methods (e.g. cross-correlation velocity)

#. Improvements and new methods in :class:`CellCalibData` objects (e.g. fitting of calibration curve, access to covariance matrix, slope error, calculation of uncertainties).

#. Renamed some methods

#. Improvements in efficiency and new methods in :class:`MeasGeometry` objects.

# New methods in :mod:`helpers.py`

#. Minor changes to example scripts

#. Major changes to :class:`ImgList` objects

  1. New list mode: ``dilution_corr``: images are loaded as dilution corrected images using the method from Campion et al., 2015. Can be activated and deactivated like all other modes (e.g. ``tau_mode``).

  #. Updated all list methods related to signal dilution correction.

  #. @property decorators (and setters) for plume distance and integration step length, i.e. :attr:`plume_dists` and :attr:`integration_step_length`

  #. Renamed :func:`next_img` and :func:`prev_img` to :func:`goto_next` and :func:`goto_prev` respectively (old names still work as well)

#. Changes to :class:`DoasFOV`: :attr:`fov_mask` is now called :attr:`fov_mask_rel`. Renamed :func:`transform_fov_mask_abs_coords` to :func:`fov_mask_abs`.

#. :class:`EmissionRateAnalysis` can now also be run with setting ``dilcorr`` using the new ``dilcorr_mode`` of :class:`ImgList` objects (see above and example script 12).

#. Some new features in class :class:`Img` (e.g. :func:`avg_in_roi`, or :func:`erode`).
