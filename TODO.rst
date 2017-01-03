TODO's
------

Main todo's for v1.0.0 release

.. todo::
        
    1. Optical flow
        a. Auto preanalysis (main flow field parameters)
        #. significancy handling
    #. Define class CalibPolyImage: this is basically a stack of two images, the first defining y axis offset of poly, the second defining slope of calib poly. Consider following cases:
      a. Only cell calibration available => potentially retrieve calib poly for each pixel individually. Determine calib poly in all image pixels on the 4th or 5th level of gaussian pyramide (i.e. longest edge < 100 pix) and then use cv2.pyrUp to retrieve calib poly image in full resolution (this can be done because the change of the poly towards the image edges is smooth and slow). Advantage of calib poly image is that then the so2 images can be calibrated simply by ``so2Im = pIm[:,:,0] + pIm[:,:,1]*tauIm`` where so2Im is the so2 image (shape: [h,w,1]), pIm is the calib poly image stack (shape: [h,w,2])and tauIm is the tau image which is supposed to be calibrated. 
      b. Only DOAS calib available: then just create calib poly image with the same poly in each pixel
      c. Cell and DOAS calib available: perform step a and determine DOAS perform correction using DOAS calib poly
    #. Calibration with spectral Data
        a. Check availability of spectra / DOASIS fit resultfiles in
        specified time window
        #. Interpolate spectral calibration to whole image area using cell
        calibration
    
    #. Finish AutoEvalution scheme
    #. Pydoas: reading routines for spectral results
        
    #. Specify supported input (more for docs..)
        a. Image file types
        
    #. Review setup convenience 
        a. Especially for unspecified data (i.e. no default
        FileNameConvention available, etc)
        #. CamSpecs and SourceDatabase into text files included in package data
            i. Users can easily create new defaults which will be found
            by the software
            #. Functionality to create a new camera type (which will then
              be written in the local file)
            #. Functionality to add a volcano to database file
            
    #. Include routine for light dilution correction (à la
    `Campion et al., 2015 <https://www.researchgate.net/publication/272239426_Image-based_correction_of_the_light_dilution_effect_for_SO2_camera_measurements>`_)
        a.  Include "distance to topo calc" into MeasGeometry
        #.  Class for fitting
            i.  Should be very low level (e.g. two input arrays: distances and intensities)
        #.  Maybe one parent class for time series analysis of dilution
            i.  Input ImgList (MeasGeometry, Forms)
            #.  Definition of pixels (forms) used for distance retrieval and intensity measure
            
    #.  Save current image with all forms in it
    #.  Uncertainty treatment
    #.  Review Background evolution polynomials in rects, consider "dip detector to automatically mask out bad sub time windows
    #.  Review optical flow analysis (does not work properly currently)
    #.  Review FOV search
        a.  does not seem to work currently
        #.  check for exposure time changes (i.e. if they are considered or dealt with somehow)
    #.  Write test routines
    #.  Scripts
        a.  live access test data (stored on git server) 
    #.  GUI
        1.  Background image edit and modelling features
        #.  Interactive viewing direction correction using object in FOV
        #.  BGPoly determination interacitvely
        #.  Save images
        #.  Plot menu (list of all plots possibilities)
        #.  ImDispWidget, right click (open menu, e.g. for plotting means)
        #.  Pop out option for embedded GraphWidgets (with toolbar, e.g. for saving)
        
        
Further stuff (to be sorted out)

.. todo::

    1.  Introduce logging in whole library, get rid of nasty prints
    #.  Spectral calibration less dependend on pydoas library
        i. e.g. introduce function self.set_doas_results(doasID="avantes",data) where data can be e.g. pandas.Series object..