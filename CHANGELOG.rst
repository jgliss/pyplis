.. include:: ../CHANGELOG_v0_9_2_v0_11_2.rst

After release 0.11.2 (not yet released)
=======================================

30-31/03/2017
-------------

1. Renamed :attr:`roi_rad` in Farneback classes to ``roi_rad_abs`` (makes it clearer in which coordinates it is supposed to be defined). The old name also still works but a warning is given if used.

2. Renamed :attr:`hist_dir_sigma` in Farneback classes to ``hist_sigma_tol`` since it is applied both to the fit result of the main peak of the orientation histogram but also to determine the uncertainty in the length histogram from the moments analysis. The old name also still works but a warning is given if used.

3. More features in :class:`LineOnImage`

  - Global velocity estimates (and uncertainties) can now be assigned (e.g. for emission rate analysis)
  - :class:`LocalPlumeProperties` can now be assigned (e.g. for emission rate analsysis and if a time series of local displacement vectors for velocity retrieval was calculated beforehand and the results are accessible in a text file)).
  
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
  
3. New attribute ``residual`` :class:`DoasCalibData`

4. Fixed some bugs related to scale space conversion in :class:`ImgList` objects (e.g. related to activation of ``tau_mode``, dilution correction)

5. Corrected bug related to SO2-uncertainty based on slope error of calibration curve from covariance matrix of poly fit. Previously: used value of slope error as measure of uncertainty (wrong), now: use relative error, e.g. calibration curve zero y-axis offset and with slope, slope err: ``m=1e19, m_err=1e17`` then the mapped SO2 error (for a given tau value ``tau0``) is determined as :``so2 = tau0 * m`` and ``so2_err = so2 * m_err / m``

6. Added mathematical operators to ``EmissionRateResults`` class

  - __add__: use "+" operator to add results (e.g. retrieved at two different lines from two crater emissions) 
  - __sub__: use "-" operator to subtract results (e.g. retrieved at two different positions downwind of the crater emissions)
  
  
10/04/2017
==========

1. Added option in :func:`make_stack` in :class:`ImgList` objects: the method includes now the option to specify a reference ROI in the image (e.g. sky reference area) and a corresponding min / max range for the expectation value in that range: if the input is specified, then only images are added to the stack that are within the specified range within the ROI.

2. New features in :class:`EmissionRateAnalysis` and :class:`EmissionRateSettings`

  - Added same feature (as described in 1.) to emission rate retrieval classes, relevant attributes in ``EmissionRateSettings`` class are: 
  
    - ``ref_check_mode``: activate / deactivate the new mode 
    - ``bg_roi_abs`` (ROI used for check)
    - ``ref_check_lower_lim``: lower intensity limit
    - ``ref_check_upper_lim``: upper intensity limit
    
  - Moved attr. ``bg_roi`` from analysis class to settings class and renamed to ``bg_roi_abs``.


11/04/2017
==========

1. Added check of date information in :func:`get_img_meta_all_filenames` of :class:`ImgList` which is, for instance, used for accessing datetime inforamtion of acq. times of all images in the list: a problem may occur if the file names only include information of acq. times of the images but not dates.  Then, the retrieved timestamps (numpy array of datetime objects) will only include acq. times of all images and the default date: 1/1/1900. If this is the case, then the method replaces these default dates in the array using the date stored in the meta header of the currently loaded image in the the list. This is, for instance relevant for the HD default camera which includes date information in the tiff header (will be loaded and stored in meta header of ``Img`` class on load, but not in the file names).
  