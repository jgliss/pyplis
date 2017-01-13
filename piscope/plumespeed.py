# -*- coding: utf-8 -*-
"""
Classes for plume speed retrievals
----------------------------------
"""
from time import time
from numpy import mgrid,vstack,int32,sqrt,arctan2,rad2deg,ogrid,\
    asarray, linspace,logical_and, histogram, exp, nan, average

from copy import deepcopy
#from scipy.misc import bytescale
from os.path import exists
from os import mkdir
from traceback import format_exc

from collections import OrderedDict as od
from matplotlib.pyplot import subplot, subplots, figure,\
    close, Figure, colorbar

from matplotlib.patches import Rectangle, Circle
from scipy.ndimage.filters import median_filter
try:
    from skimage.feature import blob
    BLOBAVAILABLE = 1
except:
    BLOBAVAILABLE = 0
    
from pandas import Series

from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN,\
    cvtColor,COLOR_GRAY2BGR,line,circle,VideoCapture,COLOR_BGR2GRAY,\
    waitKey, imshow

from .helpers import bytescale
from .processing import LineOnImage, ImgStack
from .optimisation import MultiGaussFit
from .image import Img
    
class OpticalFlowFarnebackSettings(object):
    """Settings for optical flow Farneback calculations and visualisation"""
    def __init__(self):
        """Initiation of settings object
        """
        self.contrast = od([("i_min"  ,   -9999.0),
                            ("i_max"  ,   9999.0)])
        
        self.flowAlgo = od([("pyr_scale"  ,   0.5), 
                            ("levels"     ,   4),
                            ("winsize"    ,   16), 
                            ("iterations" ,   6), 
                            ("poly_n"     ,   5), 
                            ("poly_sigma" ,   1.1)])
                            
        self.analysis = od([("roi"  ,   [0, 0, 9999, 9999])])
        
        self.display = od([("skip"               ,   10),
                           ("length_thresh"       ,   3)])
    
    def __str__(self):
        s="Input contrast:\n"
        for key, val in self.preEdit.iteritems():
            s += "%s: %s\n" %(key, val)
        s += "\nOptical flow input (see OpenCV docs):\n"
        for key, val in self.flowAlgo.iteritems():
            s += "%s: %s\n" %(key, val)
        s += "\nPlot settings:\n"
        for key, val in self.display.iteritems():
            s += "%s: %s\n" %(key, val)
        return s
    
    """
    Magic methods (overloading)
    """
    def __call__(self, item):
        for key, val in self.__dict__.iteritems():
            try:
                if val.has_key(item):
                    return val[item]
            except:
                pass
    
    def __setitem__(self, key, value):
        for k, v in self.__dict__.iteritems():
            try:
                if v.has_key(key):
                    v[key]=value
            except:
                pass
            
    def __getitem__(self, name):
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        for k,v in self.__dict__.iteritems():
            try:
                if v.has_key(name):
                    return v[name]
            except:
                pass

def determine_ica_cross_correlation(self, icaValsPCS1, icaValsPCS2, timeStamps):
    """Determines ICA cross correlation from two ICA time series
    :param ndarray icaValsPCS1: time series values of first ICA
    :param ndarray icaValsPCS1: time series values of second ICA
    :param ndarray timeStamps: array with image acquisition time stamps 
        (datetime objects)
    """
    
class OpticalFlowFarneback(object):
    """Implementation of Optical flow Farneback algorithm of OpenCV library. 
    Advanced post processing analysis of flow field in order to automatically
    identify and distinguish reliable output from unreliable (the latter is 
    mainly in low contrast regions).      
    
    .. note::
    
        Image handling withhin this object is kept on low level base, i.e. on 
        numpy arrays. Input :class:`piscope.Image.Img` also works on input but
        this class does not provide any funcitonality based on the functionality
        of :class:`piscope.Image.Img` objects (i.e. access of meta data, etc).
        As a result, this engine cannot perform any wind speed estimates (it
        would need to know about image acquisition times for that and further
        about the measurement geometry) but only provides functionality to 
        calculate and analyse optical flow fields in detector pixel 
        coordinates.
    """
    def __init__(self, name = "", **settings):        
        """Initialise the Optical flow environment"""
        self.name = name
    
        #settings for determination of flow field
        self.settings = OpticalFlowFarnebackSettings()

        
        #images used for optical flow
        self.images_prep = {"this" : None,
                            "next" : None}
        
        self._img_prep_modes = {"update_contrast"   :   False}
        #the actual flow arrays (result from cv2 algo)
        self.flow = None
        
        #flow lines calculated from self.flow
        self._flow_lines_disp = None
        
        #if you want, you can connect a TwoDragLinesHor object (e.g. inserted
        #in a histogram) to change the preEditSettings "i_min" and "i_max"
        #This will be done in both directions
        self._contrast_control_object = None
        
        for key, val in settings.iteritems():
            self.settings[key] = val
    
    
    def set_mode_auto_update_contrast_range(self, value = True):
        """Activate auto update of image contrast range
        
        If this mode is active (the actual parameter is stored in 
        ``self._img_prep_modes["update_contrast"]``), then, whenever the 
        optical flow is calculated, the input contrast range is updated based
        on minimum / maxium intensity of the first input image within the 
        current ROI.
        
        :param bool value (True): new mode
        """
        self._img_prep_modes["update_contrast"] = value
        
    def set_roi(self, roi = [0, 0, 9999, 9999]):
        """Set current ROI (Wrapper of :func:`change_roi`)
        
        :param list roi: list containing ROI coordinates ``[x0,y0,x1,y1]``
        
        """
        self.settings.analysis["roi"] = roi
    
    @property
    def roi(self):
        """Get current ROI (property)"""
        return self.settings.analysis["roi"]
        
    def current_contrast_range(self):
        """Get min / max intensity values for image preparation
        """
        i_min = float(self.settings.contrast["i_min"])
        i_max = float(self.settings.contrast["i_max"])
        return i_min, i_max
    
    def update_contrast_range(self, i_min, i_max):
        """Updates the actual contrast range for opt flow input images"""
        self.settings.contrast["i_min"] = i_min
        self.settings.contrast["i_max"] = i_max
        
    def check_contrast_range(self, img_data):
        """Check input contrast settings for optical flow calculation"""
        i_min, i_max = self.current_contrast_range()
        if i_min < img_data.min() and i_max < img_data.min() or\
                    i_min > img_data.max() and i_max > img_data.max():
            self.update_contrast_range(i_min, i_max)
    
    def get_contrast_range(self):
        """Get current min / max intensity for preparation of flow images"""
        return self.settings.contrast["i_min"], self.settings.contrast["i_max"]
        
    def set_images(self, this_img, next_img):
        """Update the current image objects 
        
        :param ndarray this_img: the current image
        :param ndarray next_img: the next image
        """
        self.flow = None
        self._flow_lines_disp = None
        i_min, i_max = self.get_contrast_range() 
        if any([abs(int(x)) == 9999 for x in [i_min, i_max]]) or\
                            self._img_prep_modes["update_contrast"]:
            roi = self.roi
            sub = this_img[roi[1]:roi[3], roi[0]:roi[2]]
            i_min, i_max = sub.min(), sub.max()
            self.update_contrast_range(i_min, i_max)

        self.images_prep["this"] = bytescale(this_img, cmin = i_min, cmax = i_max)
        self.images_prep["next"] = bytescale(next_img, cmin = i_min, cmax = i_max)
        
    
    def calc_flow(self, this_img = None, next_img = None):
        """Calculate the optical flow field
        
        Uses :func:`cv2.calcOpticalFlowFarneback` to calculate optical
        flow field between two images using the input settings specified in
        ``self.settings``.
        
        :param ndarray this_img (None): the first of two successive images (if 
            unspecified, the current images in ``self.images_prep`` are used, 
            else, they are updated)
        :param ndarray next_img (None): the second of two successive images (if 
            unspecified, the current images in ``self.images_prep`` are used, 
            else, they are updated)
            
        """
        try:
            self.set_images(this_img, next_img)
        except:
            pass
        settings = self.settings.flowAlgo
        print "Calculating optical flow"
        t0 = time()
        self.flow = calcOpticalFlowFarneback(self.images_prep["this"],\
                self.images_prep["next"], flags = OPTFLOW_FARNEBACK_GAUSSIAN,\
                **settings)
        print "Elapsed time: " + str(time() - t0)
        return self.flow 
        
    def get_flow_in_roi(self):
        """Get the flow field in the current ROI"""
        x0, y0, x1, y1 = self.roi
        return self.flow[y0 : y1, x0 : x1,:]
    
    def _prep_flow_for_analysis(self):
        """Flatten the flow fields for analysis"""
        fl = self.get_flow_in_roi()
        fx, fy = fl.T
        return fx.flatten(), fy.flatten()
    
    def get_flow_orientation_image(self):
        """Returns image corresponding to flow orientation values in each pixel"""
        fl = self.get_flow_in_roi()
        fx, fy = fl[:,:,0], fl[:,:,1]
        return rad2deg(arctan2(fx, -fy))
        
    def get_flow_vector_length_image(self):        
        """Returns image corresponding to displacement length in each pixel"""
        fl = self.get_flow_in_roi()
        fx, fy = fl[:,:,0], fl[:,:,1]
        return sqrt(fx ** 2 + fy ** 2)
        
    def _ang_dist_histo(self, bin_res_degrees = 6, multi_gauss_fit = 1, len_thresh = 0):
        """Get histogram of angular distribution of current flow field
        
        :param int bin_res_degrees (6): bin width of histogram (is rounded to
            nearest integer if not devisor of 360)
        :param bool multi_gauss_fit (True): apply multi gauss fit to histo
        :param int len_thresh (1): flow vectors shorter than len_thresh are not
            included into histogram determination
        """
        fx, fy = self._prep_flow_for_analysis()
        angles = rad2deg(arctan2(fx,-fy))
        lens = sqrt(fx**2 + fy**2)
        cond = lens > len_thresh
        angles_cond = angles[cond]
        num_bins = int(round(360 / bin_res_degrees))
        print "Number of bins histo: " + str(num_bins)
        count, bins = histogram(angles_cond, num_bins)
        
        if multi_gauss_fit:
            x = asarray([0.5 * (bins[i] + bins[i + 1]) for\
                                                i in xrange(len(bins) - 1)])
            fit = MultiGaussFit(count, x)
        return count, bins, angles, fit
    
    def _len_dist_histo(self, fx = None, fy = None, gaussFit = 1):
        if None in [fx,fy]:
            fx,fy=self._prep_flow_for_analysis()
        lens=sqrt(fx**2+fy**2)
        n, bins=histogram(lens,60)
#==============================================================================
#         
#         m=self.multiGaussHisto.lens
#         m.init_results()
#         #estimate the histogram offset by taking the mean
#         if gaussFit:
#             x = asarray([0.5 * (bins[i] + bins[i+1]) for i in xrange(len(bins)-1)])
#             m.set_data(x,n)
#             if not m.fit_multiple_gaussian():
#                 #and if this does not work, try to fit a single gaussian (based
#                 #on position of maximum count)
#                 m.init_results()
#                 m.init_data()
#                 m.fit_single_gaussian()
#             
#             if m.got_results() and m.get_peak_to_peak_residual()>2*m.noiseAmpl:
#                 print ("Non optimum residual after first iteration of (Multi)"
#                     "gauss fit: Running optimisation")
#                 
#                 m.run_optimisation()
#==============================================================================

        return n, bins, lens#, m
    
    def get_main_flow_field_params(self, lenThresh=4, angRangeSigmaFrac=1):
        """Apply
        Try to fit a gaussian to the angular distribution and determine mean
        flow direction (+/- sigma) and the average displacement length from
        statistical using masking operations
        """
        #n1, bins1, lens1=self._len_dist_histo()
#==============================================================================
#         if not m1.got_results():
#             print ("Could not retrieve main flow field parameters, fit of "
#                 "length distribution histogram failed")
#             return 0
#==============================================================================
        #mu1, sigma1,_,_=m1.get_main_gauss_info()
        #lenThresh=mu1+3*sigma1
        print "\n\nGETTING ANGULAR DISTRIBUTION HISTOGRAM\N------------------------------------------------\n\n"
        #print "Applying length thresh to histo: " + str(lenThresh)
        n, bins, angles, m = self._ang_dist_histo(lenThresh=lenThresh)
        if not m.got_results():
            print ("Could not retrieve main flow field parameters..probably "
            "due to failure of multi gaussian fit to angular distribution "
            "histogram")
            return 0
        mu,sigma,_,_=m.get_main_gauss_info()
        angMin, angMax = mu - sigma * angRangeSigmaFrac,\
                                    mu + sigma * angRangeSigmaFrac
        angles = self.get_flow_angle_image().flatten()
        lens = self.get_flow_vector_length_image().flatten()
        cond1 = logical_and(angles > angMin, angles < angMax)
        cond2 = lens > lenThresh
        cond = cond1 * cond2
        
        goodLens = lens[cond]
        goodAngles = angles[cond]
        #badLens=self.get_flow_vector_length_image().flatten()[invert(cond)]
        #lMax=badLens.max()
        v, vErr, vmax = goodLens.mean(), goodLens.std(), goodLens.max()
#==============================================================================
#         steps=50
#         step=float(lMax/steps)
#==============================================================================
        #totNum=float(len(badLens))        
        #print "Maximum length in selected angular range: " + str(lMax)
        print "AngMin: " + str(angMin)
        print "AngMax: " + str(angMax)
#==============================================================================
#         for k in range(50):
#             thresh=k*step
#             if len(badLens[badLens<thresh])/totNum>0.997:
#==============================================================================
        return lenThresh, mu, sigma, v, vErr, goodLens, goodAngles, vmax
        
    def gauss(self, x, *p):
        O, A, mu, sigma = p
        return O+A*exp(-(x-mu)**2/(2.*sigma**2))
    
    def _map_blob_coordinates(self, blobs, pyrDownSteps):
        mapped = []
        op = 2 ** pyrDownSteps
        for bl in blobs:
            b=[]
            for num in bl:
                b.append(num * op)
            mapped.append(b)
        return asarray(mapped)
    
    def create_circular_masks_from_blobs(self, blobs):
        """
        Creates circular masks for optical flow image evaluation in areas
        of increased flow lengths
        
        :param blobs: output from :func:`find_blobs`        
        """           
        masks=[]
        for bl in blobs:
            print "Current blob: " + str(bl)
            masks.append(self.make_circular_mask(bl[1], bl[0],bl[2]))
        return masks
            
    def make_circular_mask(self,cx, cy, radius):
        """Create a mask for flow field evaluation
        """
        h,w = self.flow.shape[:2]
        y,x = ogrid[:h,:w]
        m=(x-cx)**2+(y-cy)**2 < radius**2
        return m
    
    def find_blobs(self, minSize=None, maxSize=None, pyrDownSteps=2, draw=0):
        """
        Find regions of increased velocity based on the flow length image
        
        :returns list blobs: list of coordinates of regions (y,x,radius)
        
        .. todo::
        
            separate blob search for anlge image and length image
            
        """
        res=self.settings("winsize")/3
        if minSize is None:
            minSize=res
        if maxSize is None:
            maxSize=res*6
        print "Searching blobs"
        print "Minimum considered size: " + str(res)
        print "Maximum considered size: " + str(maxSize)
        prep=ImagePreparation()
        prep["pyrlevel"]=pyrDownSteps
        minSize=minSize/2**pyrDownSteps
        maxSize=maxSize/2**pyrDownSteps
        arr=self.get_flow_vector_length_image()
        lenImObj=Img(arr)
        lenImObj=prep.apply_img_shape_settings(lenImObj)
        blobs=blob.blob_doh(lenImObj.img,min_sigma=minSize,max_sigma=maxSize)
        blobs=self._map_blob_coordinates(blobs,pyrDownSteps)
        if draw:
            fig, ax=subplots(1,2)
            self.draw_flow(showInROI=1,ax=ax[0])
            self.draw_blobs(blobs,ax=ax[0])
            ax[0].set_title("Flow vector image")
            disp=ax[1].imshow(arr, cmap="gray")
            colorbar(disp, ax=ax[1])
            self.draw_blobs(blobs,ax[1])
            ax[1].set_title("Flow length image")
        return blobs
    
    def draw_blobs(self, blobs=None,ax=None):
        """
        Draw blobs into axes
        """
        if blobs is None:
            blobs=self.find_blobs()
        if ax is None:
            fig,ax=subplots(1,1)
        for bl in blobs:
             y, x, r = bl
             c = Circle((x, y), r, color="lime", linewidth=2, fill=False)
             ax.add_patch(c)
    
    def estimate_mean_displacement_from_blobs(self, mu, sigma, lenThresh, nSigma=1):
        """
        Get all blobs which are within the angular range of the mean flow and
        get flow length displacement estimate from those
        """
        blobs=self.find_blobs()
        if len(blobs) is 0:
            print ("Could not estimate flow displacement from blobs, no blobs"
                " were detected")
            return 0
        angIm = self.get_flow_angle_image()
        lenIm = self.get_flow_vector_length_image()
        masks = self.create_circular_masks_from_blobs(blobs)
        goodMasks=[]
        meanLens=[]
        stdLens=[]
        weights=[]
        sig=nSigma*sigma
        for m in masks:
            angles=angIm[m]
            lens=lenIm[m]
            cond=lens>lenThresh
            angles=angles[cond]
            lens=lens[cond]
            if mu-sig < angles.mean() < mu+sig and angles.std() < sig:
                goodMasks.append(m)
                meanLens.append(lens.mean())
                stdLens.append(lens.std())
                weights.append(len(lens))
        if len(goodMasks) is 0:
            print ("Could not estimate flow displacement from blobs")
            return 0
        v=average(meanLens,weights=weights)
        vErr=average(stdLens,weights=weights)
        return v, vErr, goodMasks,meanLens,stdLens
        
    def median_filter(self, shape=(3,3)):
        """Apply a median filter to flow field, i.e. to both flow images (dx, dy
        stored in self.flow) individually
        
        :param tuple shape (3,3): size of the filter
        """
        
        self.flow[:,:,0]=median_filter(self.flow[:,:,0],shape)
        self.flow[:,:,1]=median_filter(self.flow[:,:,1],shape)
        
    """
    Plotting / visualisation etc...
    """
    def plot_gauss(self, x, gaussParams, ax=None):
        if ax is None:
            ax=subplot(1,1,1)
        xFit=linspace(x.min(),x.max(),100)
        yFit=self.gauss(xFit,*gaussParams)
        ax.plot(xFit,yFit,"--r", label="Gauss fit")
        
    def plot_flow_histograms(self,lenThresh = 4, drawGauss = 1, forApp = 0):
        """
        Plot the histograms of the current flow field (which was determined in 
        rectangle self.settings.imgShapePrep.settings.roi).
        
        :param list rect: get the statistics only in this sub rectangle (in 
            absolute image coordinates -> needless to say, that rect 
            needs to be within ROI of flow)

        :param bool applyLenThresh: if true, then the flow vector angle histo
            only includes vectors with lengths exceeding the length thres which
            is determined from the upper end (3*sigma) of a gauss fit applied 
            to the length histogram (based on the assumption that most lengths
            retrieved in the optical flow calculation window are unreliable and
            thus short, i.e. most areas of the image have too little contrast 
            in order for the algorithm to work). 
            
        """
        #set up figure and axes
        if not forApp:
            fig=figure(figsize=(16,6))
        else:
            fig=Figure(figsize=(16,6))
        #three strangely named axes for top row 
        ax1=fig.add_subplot(2,3,1)
        ax4=fig.add_subplot(2,3,2)
        ax2=fig.add_subplot(2,3,3)
        
        ax11=fig.add_subplot(2,3,4)        
        ax5=fig.add_subplot(2,3,5)
        ax3=fig.add_subplot(2,3,6)
        
        
#==============================================================================
#         gs=GridSpec(2,3)
#         ax1=subplot(gs[0,0])
#         ax11=subplot(gs[1,0])
#         ax2=subplot(gs[0,2])
#         ax3=subplot(gs[1,2])
#         ax4=subplot(gs[0,1])
#         ax5=subplot(gs[1,1])
#==============================================================================
        #draw the optical flow image
        self.draw_flow(showInROI=0,ax=ax1)
        self.draw_flow(showInROI=1,ax=ax11)
        #load and draw the length and angle image
        #fx,fy=self.flow[:,:,0],self.flow[:,:,1]
        angleIm=self.get_flow_angle_image()#rad2deg(arctan2(fx,-fy))
        lenIm=self.get_flow_vector_length_image()#sqrt(fx**2+fy**2)
        aIm=ax4.imshow(angleIm,interpolation='nearest')
        ax4.set_title("Angles",fontsize=11)        
        fig.colorbar(aIm,ax=ax4)
        lIm=ax5.imshow(lenIm,interpolation='nearest')
        fig.colorbar(lIm,ax=ax5)
        ax5.set_title("Lengths", fontsize=11)        
#==============================================================================
#         if rect is not None:
#             x0,y0,h,w=self.settings.imgShapePrep._rect_coordinates(rect)
#             ax1.add_patch(Rectangle((x0,y0),w,h,fc="none",ec="g"))
#==============================================================================
        #fx,fy=self._prep_flow_for_analysis(rect)
        
        #now prepare the histograms
        fx,fy=self._prep_flow_for_analysis()
        n1, bins1, lens=self._len_dist_histo(fx,fy)
        tit="Flow angle histo"
        substr=""
#==============================================================================
#         if m1.got_results():
#             mu1, sigma1,_,_= m1.get_main_gauss_info()
#==============================================================================
        if lenThresh>0:
            #lenThresh=mu1+3*sigma1
            ax3.axvline(lenThresh,linestyle="--", color="r")
            ax3.text(lenThresh*1.01,n1.max()*0.95,"Length thresh", color="r")
            substr="\nOnly vectors longer than " + "{:.1f}".format(lenThresh)
        print "Getting angular histogram, length threshold: " +  str(lenThresh)
        n, bins, angles, m = self._ang_dist_histo(lenThresh=lenThresh)
        #n, bins,angles,m=self._ang_dist_histo(fx,fy,gaussFit=1)
        if m.got_results():
            mu,sigma,_,_=m.get_main_gauss_info()
        
            muSigmaStr=(" mu (sigma) =" + "{:.1f}".format(mu) + " (+/- " 
                                               + "{:.1f}".format(sigma) + ")")
            tit=tit + muSigmaStr + substr
        w=bins[1]-bins[0]
        ax2.bar(bins[:-1],n,width=w, label="Histo")
        ax2.set_title(tit, fontsize=11)
        if m.got_results() and drawGauss:
            m.plot_multi_gaussian(ax=ax2, label="Multi-gauss fit")
        ax2.set_xlim([-180,180])    
        ax2.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)                
        #now the lenght histogram
        tit="Flow length histo"
#==============================================================================
#         if m1.got_results():
#             muSigmaStr=(" mu (sigma) =" + "{:.1f}".format(mu1) + " (+/- " 
#                                             + "{:.1f}".format(sigma1) + ")")
#             tit=tit + muSigmaStr
#==============================================================================
        w=bins1[1]-bins1[0]
        ax3.bar(bins1[:-1],n1,width=w)
#==============================================================================
#         if m1.got_results() and drawGauss:
#             m1.plot_multi_gaussian(ax=ax3, label="Gauss fit")
#==============================================================================
        ax3.set_ylim([0,n1.max()*1.1])
        ax3.set_xlim([0,bins1[-1]])
        ax3.set_title(tit, fontsize=11)
        #suptitle(self.rawImages.this._meta("startUTC").strftime('%d/%m/%Y %H:%M:%S'))
        
        return fig
     
    def calc_flow_lines(self):
        """
        Determine line objects for visualisation of the latest flow field
        
        :param step: Resolution of line grid 
        
        """
        settings = self.settings
        step = settings("skip")
        #get the shape of the rectangle in which the flow was determined
        flow = self.get_flow_in_roi()
        h, w = flow.shape[:2]
        #create and flatten a meshgrid 
        y,x = mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
        fx,fy = flow[y,x].T
        
        thresh = self.settings("length_thresh")
        if thresh > 0:
            #use only those flow vectors longer than the defined threshold
            cond = sqrt(fx**2 + fy**2) > thresh
            x, y, fx, fy = x[cond], y[cond], fx[cond], fy[cond]
        # create line endpoints
        lines = int32(vstack([x, y,x + fx, y + fy]).T.reshape(-1,2,2))
        return lines
#==============================================================================
#         dx,dy=self.analysisSettings.get_subimg_offset_xy()
#         #self.flowLines=lines
#         self.flowLinesInt=[int32(lines), dx, dy]
#         return self.flowLinesInt
#==============================================================================
    
    def show_flow_field(self, rect=None):
        """Plot the actual flow field (dx,dy images) 
        
         :param list rect: sub-img rectangle specifying ROI for flow field \
            analysis (in absolute image coordinates)
        """
        raise NotImplementedError
#==============================================================================
#         if rect is None:
#             x0,y0=0,0
#             h,w = self.flow.shape[:2]
#         else:
#             x0,y0=self.settings.imgShapePrep.map_coordinates(rect[0][0],rect[0][1])
#             h,w=self.settings.imgShapePrep.get_subimg_shape()
#==============================================================================
            
    
    def draw_flow(self, showInROI = False, ax = None):
        """Draw the current optical flow field
        
        :param bool showInROI: if True, the flowfield is plotted in the 
            cropped image (using current ROI), else, the whole image is 
            drawn and the flow field is plotted within the ROI which is 
            indicated with a rectangle
        :param ax (None): matplotlib axes object
        """
        if ax is None:
            fig, ax=subplots(1,1)
        dx,dy=0,0
        img=self.rawImages.this
        if showInROI:
            img = img.get_sub_img(self.analysisSettings["roi"])
        i_min,i_max = self.get_current_contrast_range()
        disp = cvtColor(bytescale(img.img,cmin=i_min,cmax=i_max),COLOR_GRAY2BGR)    

        if not showInROI:
#==============================================================================
#             dx,dy=self.settings.imgShapePrep.get_subimg_offset_xy()
#             h,w=self.settings.imgShapePrep.get_subimg_shape()
#==============================================================================
            dx,dy = self.analysisSettings.get_subimg_offset_xy()
            h,w = self.analysisSettings.get_subimg_shape()
            ax.add_patch(Rectangle((dx,dy),w,h,fc="none",ec="c"))
            
        lines=self.calc_flow_lines()[0]
        if lines is None:
            print "Could not draw flow, no flow available"
            return
        
        for (x1,y1),(x2,y2) in lines:
            line(disp,(x1+dx,y1+dy),(x2+dx,y2+dy),(0,255,255),1)
            circle(disp,(x2+dx,y2+dy),1,(255,0,0), -1)
        self.flowImage=disp
        ax.imshow(disp)
        tit=("delT: " + str(self.get_current_delt()) + "s, Thresh: " + 
            str(self.settings.drawSettings["flowLinesThresh"]) + "pix")
        ax.set_title(tit,fontsize=10)
        return ax,disp
        
    def live_example(self, roi = None):
        """Show live example using webcam"""
        cap = VideoCapture(0)
        ret,im = cap.read()
        gray = cvtColor(im,COLOR_BGR2GRAY)
        self.images_prep["this"] = gray
        while True:
            # get grayscale image
            ret, im = cap.read()
            self.images_prep["next"] = cvtColor(im,COLOR_BGR2GRAY)
            
            # compute flow
            #flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
            flow = self.calc_flow()
            self.images_prep["this"] = self.images_prep["next"]
        
            # plot the flow vectors
            vis = cvtColor(self.images_prep["this"], COLOR_GRAY2BGR)
            lines = self.calc_flow_lines()
            for (x1,y1),(x2,y2) in lines:
                line(vis,(x1,y1),(x2,y2),(0,255,255),1)
                circle(vis,(x2,y2),1,(255,0,0), -1)
            imshow("Optical flow live view", vis)
            if waitKey(10) == 27:
                self.flow = flow
                break
            
    """
    Connections etc.
    """
    def connect_histo(self,canvasWidget):
        self.contrastControlObject = canvasWidget
        
    
    """
    Magic methods (overloading)
    """
    def __call__(self, item=None):
        if item is None:
            print "Returning current optical flow field, settings: "
            print self.settings
            return self.flow
        for key, val in self.__dict__.iteritems():
            try:
                if val.has_key(item):
                    return val[item]
            except:
                pass 
            
class OpticalFlowAnalysis(object):
    """A class for analysis of optical flow characteristics considering
    all images in an :class: ImgList` object. The analysis of the flow field 
    is based on one or more ROIs within the original images. These ROIs have to 
    be added manually using:: 
    
        self.add_roi(rect=None,id)
    
    """
    def __init__(self, lst = None, line = None, settings = None, imgPrepDict = None):
        self.line = None
        self.imgList = None
        
        self.optFlowEdit = None
        
        #:Main ROIs, i.e. ROIs for which the optical flow field will be determined
        #:(size needs to be large than 80x80 pixels)
        self.mainROI = None
        #: Sub ROIs: ROIs within main ROIs
        self.subRois=Bunch()
        
        self.meanFlowFieldInfo=None
        if lst:
            self.set_imglist(lst,imgPrepDict)        
        if line is not None:
            self.set_line(line)
        if settings is not None:
            self.optFlowEdit.add_settings(settings)
            self.optFlowEdit.change_current_settings_object(settings.id)
    
    def set_imglist(self, lst, imgPrepDict = None):
        """Set a deepcopy of the input list and if applicable, change image 
        preparation settings
        
        :param ImgListStatic lst: the image list
        :param dict imgPrepDict (None): img preparation settings
        
        """
        try:
            if lst.numberOfFiles>0:
                self.imgList=deepcopy(lst)
                self.optFlowEdit=self.imgList.optFlowEdit=OpticalFlowFarneback(id=self.imgList.id)
                if isinstance(imgPrepDict, dict):
                    self.imgList.update_img_prep_settings(imgPrepDict)
        except:
            raise TypeError(format_exc())

    def img_prep_settings(self):
        """Return info about the image preparation setting
        """
        return self.imgList.current_edit()
        
    def add_settings(self, optFlowSettings):
        self.optFlowEdit.add_settings(optFlowSettings)
    
    def change_current_settings_object(self, key):
        self.optFlowEdit.change_current_settings_object(key)
        self.imgList.update_img_prep_settings(self.optFlowEdit.settings.imgPrepSettings)
        
    def set_save_path(self, p):
        if exists(p):
            self.savePath=p
            
    def set_line(self, line):
        if not isinstance(line, LineOnImage):
            raise TypeError("Input is not a piscope.Processing.LineOnImage object")
        self.line=line
        
    def set_main_roi(self,rect):
        """Add one ROI for optical flow analysis
        """
        self.mainROI=self.rect_2_roi(rect)
    
    def _in_roi(self, subRoi, roi):
        """Test if subRoi lies in roi
        """
        s,r=subRoi,roi
        if not(s[0]>=r[0] and s[1]<=r[1] and s[2]>=r[2] and s[3]<=r[3]):
            raise ValueError("Sub-ROI exceeds borders of parent ROI")
        return 1
        
        
    def add_sub_roi(self, rect, id):
        """Add sub ROI to existing main ROI
        
        :param list rect: rectangle defining ROI
        :param str id: id of sub-roi (e.g. "zoom")
        
        If main ROI exists and sub-ROI is within main ROI then add subROI        
        """
        subRoi=self.rect_2_roi(rect)
        mainRoi=self.mainROI
        if self._in_roi(subRoi, mainRoi):
            self.subRois[id]=subRoi
        
    def rect_2_roi(self,rect,inverse=0):
        return self.optFlowEdit.settings.imgShapePrep.rect_2_roi(rect, inverse)
        
    def set_main_roi_from_line(self,**kwargs):
        if self.line is None:
            raise KeyError("No line found")
        self.line.set_roi(self.imgList.loadedImages.this.img,**kwargs)
        self.mainROI=self.line.roi()
        self.update_flow_roi(self.line.id)
    
    def update_flow_roi(self, roiId):
        self.optFlowEdit.set_roi(self.rect_2_roi(self.mainROI,1))
    
    def estimate_mean_flow(self, plotLengthProfile=0):
        """Estimates the mean values of the flow field along the line
        """
        p=self.optFlowEdit.get_main_flow_field_params()
        if p == 0:
            return 0
        lenThresh, mu, sigma, v, vErr, goodLens, goodAngles,vmax=p
        fx,fy=self.optFlowEdit.flow[:,:,0],self.optFlowEdit.flow[:,:,1]
        lx=self.line.get_line_profile(fx,key="roi")
        ly=self.line.get_line_profile(fy,key="roi")
        #totNum=len(lx)
        lens=sqrt(lx**2+ly**2)
        if plotLengthProfile:
            fig,ax=subplots(1,1)
            ax.plot(lens, ' x')
            ax.set_title("Flow length vector distribution along line")
        angles=rad2deg(arctan2(lx,-ly))
        cond1=lens>lenThresh
        #cond2=logical_and(mu-3*sigma<angles,mu+3*sigma>angles)
        cond2=logical_and(mu-sigma<angles,mu+sigma>angles)
        cond=cond1*cond2
        #get all lens on line pointing in the right direction and having
        #an acceptable length
        mask=lens[cond]
        if len(mask)>0:
#==============================================================================
#         if not len(mask) > 0.1*totNum or len(mask) < 20:
#             print ("Mean flow could not be estimated too little amount of"
#                 " datapoints on the line")
#             return 0
#==============================================================================
            fit = MultiGaussFit(datY=lens, id="Lengths on PCS")
            fit.set_noise_amplitude(lenThresh)
            if not fit.fit_multiple_gaussian():
                #and if this does not work, try to fit a single gaussian (based
                #on position of maximum count)
                fit.init_results()
                fit.init_data()
                fit.fit_single_gaussian()
            if fit.got_results():   
                fit.run_optimisation()
                mu1, sigma1,_,_=fit.get_main_gauss_info()
                mask=lens[logical_and(mu1-sigma1<fit.x,mu1+sigma1>fit.x)]
                print fit.gauss_info()
                #v,vErr=mask.mean(),mask.std()
                v,vErr=mask.max(),mask.std()
                print "\nSuccesfully estimated mean flow"
                print v, vErr, lenThresh
            else:
                print ("Mean flow velocity along PCS could not be determined")
                v, vErr=nan,nan
        else:
            v, vErr=nan,nan
            
        return lenThresh, mu, sigma, v, vErr, fit

    def flow_field_mean_analysis(self, lenThresh=4):
        """Histogram based analysis of optical flow in image time series of 
        `self.imgList`.
        """
        self.optFlowEdit.active=1
        self.imgList.goto_im(0)
        num=self.imgList.numberOfFiles
        #bad=FlowFieldAnalysisResults(self.mainROI)
        good=FlowFieldAnalysisResults(self.mainROI)
        h,w=self.optFlowEdit.flow.shape[:2]
        good.init_stacks(h,w,num-1)
        times=self.imgList.get_img_times()
        lastParams=self.optFlowEdit.get_main_flow_field_params()
        if lastParams == 0:
            print("Mean analysis of flow field failed, significant flow direction"
                " could not be initialised, gaussian fit insignificant, check"
                " optical flow input and contrast input settings for ROI: " +
                str(self.mainROI))
            return 0
        lenThresh, mu, sigma, v, vErr, goodLens, goodAngles, vmax=lastParams
#==============================================================================
#         blInfo=self.optFlowEdit.estimate_mean_displacement_from_blobs(mu=mu,\
#                                             sigma=sigma, lenThresh=lenThresh)
#         if blInfo is not 0:
#             v1, v1Err=blInfo[0], blInfo[1]
#         else:
#             v1,v1Err=nan,nan
#==============================================================================
        good.append_result(times[0],v,vErr,lenThresh, mu, sigma,vmax)
        good.stacks.angleImgs.set_img(self.optFlowEdit.get_flow_angle_image(),0)
        good.stacks.lenImgs.set_img(self.optFlowEdit.get_flow_vector_length_image(),0)
        for k in range(1,num-1):
            self.imgList.next_im()
            good.stacks.angleImgs.set_img(self.optFlowEdit.get_flow_angle_image(),k)
            good.stacks.lenImgs.set_img(self.optFlowEdit.get_flow_vector_length_image(),k)
            lastParams=self.optFlowEdit.get_main_flow_field_params()
            if lastParams !=0:
                lenThresh, mu, sigma, v, vErr, goodLens, goodAngles, vmax=lastParams
#==============================================================================
#                 blInfo=self.optFlowEdit.estimate_mean_displacement_from_blobs(\
#                                         mu=mu, sigma=sigma, lenThresh=lenThresh)
#                 if blInfo is not 0:
#                     v1, v1Err=blInfo[0], blInfo[1]
#                 else:
#                     v1,v1Err=nan,nan
#==============================================================================
            else:
                print "Failed to estimate mean flow at image num: " + str(k)
                lenThresh, mu, sigma, v, vErr,vmax =[nan,nan,nan,nan,nan,nan]
            good.append_result(times[k],v,vErr,lenThresh, mu, sigma,vmax)
        good.make_pandas_series()
        self.meanFlowFieldInfo=good
        return good
        
    def draw_current_flow(self, includeBlobs=1, disp=1):
        if disp:
            fig=figure(figsize=(18,8))
        else:
            fig=Figure(figsize=(18,8))
        axes=[]
        axes.append(fig.add_subplot(1,2,1))
        axes.append(fig.add_subplot(1,2,2))
        axes[0], img=self.optFlowEdit.draw_flow(showInROI=0, ax=axes[0])
        axes[1], roiImg=self.optFlowEdit.draw_flow(showInROI=1, ax=axes[1])
        if includeBlobs:
            self.optFlowEdit.draw_blobs(ax=axes[1])
        if isinstance(self.line, LineOnImage):
            l=self.line
            axes[0].plot([l.start[0],l.stop[0]], [l.start[1],l.stop[1]],'co-')
            roi=l.roi()
            if roi is not None:
                dx,dy=roi[1]-roi[0], roi[3]-roi[2]
                axes[0].add_patch(Rectangle((roi[0],roi[2]),dx,dy,fc="none",ec="c"))
        
            x0,y0=self.optFlowEdit.settings.imgShapePrep.map_coordinates(l.start[0], l.start[1])
            x1,y1=self.optFlowEdit.settings.imgShapePrep.map_coordinates(l.stop[0], l.stop[1])
            axes[1].plot([x0,x1], [y0,y1],'co-')
            
        axes[0].set_xlim([0,img.shape[1]])
        axes[0].set_ylim([img.shape[0],0])
        axes[1].set_xlim([0,roiImg.shape[1]])
        axes[1].set_ylim([roiImg.shape[0],0])
        #axis('image')
        s=self.imgList.current_time().strftime("%Y.%m.%d %H:%M:%S")
#==============================================================================
#         try:
#==============================================================================
        lenThresh, mu, sigma, v, vErr, _,_,vmax=self.\
                    optFlowEdit.get_main_flow_field_params()
        s=(s+"\nMean displacement: " + "{:.1f}".format(v) + " (+/- " + "{:.1f}".format(vErr) + "), Max: " + "{:.1f}".format(vmax) + "pix\n"
        "Mean direction: " + "{:.1f}".format(mu) + " (+/- " + "{:.1f}".format(sigma) + ") deg")
#==============================================================================
#         except:
#             raise ValueError()
#==============================================================================
        axes[0].set_title(s,fontsize=12)
        return fig, axes
    
    def determine_and_save_all_flow_images(self, folderName, startNum=None, stopNum=None):
        if startNum is None:
            startNum=0
        if stopNum is None:
            stopNum=self.imgList.numberOfFiles-1
        if not self.savePath:
            print "Error"
            
            return
        p=self.savePath + folderName + "/"
        if exists(p):
            print "Path already exists, choose another name"
            return
        mkdir(p)
        self.optFlowEdit.active=1
        self.imgList.goto_im(startNum)
        for k in range(startNum+1,stopNum):
            fig, ax=self.draw_current_flow(disp=1)
            fig.savefig(p+str(k)+".png")
            close(fig) 
            del fig, ax
            self.imgList.next_im()
        
        

class FlowFieldAnalysisResults(object):
    """This object stores results (mean direction +/-, mean length, thresholds
    etc...) for an image time series (:class:`ImgList`) in a certain ROI
    of the image stack
    """
    def __init__(self, roi):
        self.roi=roi
        self.times=[]
        self.stacks=Bunch({"lenImgs" : ImgStack("lenImgs"),
                           "angleImgs": ImgStack("angleImgs")})
                           
        self.results=Bunch({"lenThreshs"    :   [],
                            "meanDirs"      :   [],
                            "meanDirErrs"   :   [],
                            "meanVelos"     :   [],
                            "meanVeloErrs"  :   [],
                            "maxVelos"      :   []})
        
        self.pandasSeries=Bunch()
        
        
    def init_stacks(self,h,w,d):
        for stack in self.stacks.values():
            stack.init_stack(h,w,d)
    
    def plot_overview(self):
        fig, axes=subplots(2,1)
        errs=self.pandasSeries.meanDirErrs
        self.pandasSeries.meanDirs.plot(ax=axes[0],yerr=errs.values)        
        axes[0].set_title("Mean flow direction [deg]")
        errs=self.pandasSeries.meanVeloErrs
        self.pandasSeries.meanVelos.plot(ax=axes[1], label="Mean (histogram analysis)",yerr=errs.values)
        blobErrs=self.pandasSeries.blobVeloErrs.values
        self.pandasSeries.blobVelos.plot(ax=axes[1], label="Mean (Blob analysis)",yerr=blobErrs)
        axes[1].set_title("Flow diplacement length [pix]")
        self.pandasSeries.maxVelos.plot(ax=axes[1], label="Max")
        axes[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
        
    def append_result(self,time,v,verr,lenThresh, mu, sigma, vmax):
        self.times.append(time)
        self.results.lenThreshs.append(lenThresh)
        self.results.meanVelos.append(v)
        self.results.meanVeloErrs.append(verr)
        self.results.maxVelos.append(vmax)
        self.results.meanDirs.append(mu)
        self.results.meanDirErrs.append(sigma)

    
    def make_pandas_series(self):
        """Make pandas time series objects of all results
        """
        for key,val in self.results.iteritems():
            self.pandasSeries[key]=Series(val,self.times)
    def prepare_pandas_series(self,):
        """
        """
        
#==============================================================================
# class WindRetrievalCollection(object):
#     """The base class for storing any information about the wind field and
#     standard algorithms to retrieve wind information using different methods. 
#     
#     In the current version, this is mainly:
#     
#         1. Manually setting average displacement and average orientation angle
#         #. Optical flow farneback analysis
#             1. Use output as is 
#             #. Do meanFlowField analysis (link..)
#     
#     """
#     def __init__(self, imgList=None, measGeometry=None):
#         self.imgList = imgList
#         self.measGeometry = measGeometry
#         
#         self.optFlowAnalysis = OpticalFlowAnalysis()
#         #this dictionary sets the image preparation info for optical flow
#         #calculations (see also :func:`self.set_imglist`)
#         self.optFlowImgPrep = Bunch([("DarkCorr"    ,   1),
#                                      ("Blurring"    ,   1)])
#         
#         
#         #:In this dictionary, global displacements (however measured, e.g. in
#         #: GUI) can be added, please use :func:`self.add_glob_displacement` to
#         #: do so. 
#         self._glob_displacements = Bunch()
#         self.warnings=["No warning"]
#     
#     def get_opt_flow_settings(self):
#         """Get current optical flow settings
#         """
#         try:
#             return self.optFlowAnalysis.optFlowEdit.settings
#         except:
#             msg=("Could not retrieve optical flow settings, OpticalFlowAnalysis "
#                 "tool might not be set")
#             self._warning(msg)
#             return 0
#             
#     def set_opt_flow_settings(self, settings):
#         """Set settings object for optical flow calculations
#         """
#         try:
#             if self.optFlowAnalysis.optFlowEdit.moreSettings.has_key(settings.id):
#                 msg=("Opt flow settings with id: " + str(settings.id) + " were "
#                     "overwritten")
#                 self._warning(msg)
#             self.optFlowAnalysis.add_settings(settings)
#             self.optFlowAnalysis.change_current_settings_object(settings.id)
#         except:
#             msg=("Could not retrieve optical flow settings, OpticalFlowAnalysis "
#                 "tool might not be set")
#             self._warning(msg)
#             return 0
#         
#     def add_glob_displacement(self,timeStamp, delt, len, lenErr, orientation,\
#                                                                 orientationErr):
#         """Set global values for displacement vector
#         
#         :param datetime timeStamp: datetime to which the displacment corresponds
#         :param float delt: time difference between the two images used to 
#             measure displacement
#         :param float len: length of displacement in pix
#         :param float len: uncertainty of length of displacement in pix
#         :param float orientation: orientation angle in deg 
#             0: upwards (-y direction)
#             180 and -180: downwards (+ y direction)
#             90: to the right (+ x direction)
#             -90: to the left (- x direction)
#         
#         Writes the data into `self._glob_displacements` in the following format::
#         
#             self._glob_displacements[timeStamp]=[delt, len, lenErr, orientation,\
#                 orientationErr]
#         
#         .. note::
#         
#             if you use this, make sure, to use the right datatypes, no control
#             of input performed here
#             
#         """
#         self._glob_displacements[timeStamp]=[delt, len, lenErr, orientation,\
#                                                                 orientationErr]
#     
#     def get_wind_info_glob(self, timeStamp=None):            
#         """Get global information about wind velocity and vetor orientation from
#         information in `self._glob_displacements`
#         :param datetime timeStamp: the time at which the data is supposed to be
#             retrieved (searches closest available info in 
#             `self._glob_displacements` dictionary)
#         """
#         if timeStamp is None:
#             try:
#                 timeStamp=self.imgList.current_time()
#             except:
#                 self.warning()
#         t0=min(self._glob_displacements.keys(), key=lambda x: abs(x - timeStamp))
#         dx=self.measGeometry.calcs["pixLengthX"] #the pixel lenght in m
#         info=self._glob_displacements[t0]
#         v, vErr=float(info[1])*dx/info[0],float(info[2])*dx/info[0]
#         return WindVector2D(v, vErr, info[3], info[4])
#     
#     def update_imprep_optflow(self, dictLike):
#         """Update valid entries in image preparation dict for optical flow 
#         calculations
#         :param dict dictLike: dictionary with variables
#         
#         """
#         for key, val in dictLike.iteritems():
#             if self.optFlowImgPrep.has_key(key):
#                 self.optFlowImgPrep[key]=val
#                 
#     def set_imglist(self,imgList, imgPrepDict=None):
#         """Try setting imgList object used for wind field analysis
#         """
#         try: 
#             if isinstance(imgPrepDict, dict):
#                 self.update_imprep_optflow(imgPrepDict)
#             if imgList.numberOfFiles>0:
#                 self.imgList=imgList
#                 self.optFlowAnalysis.set_imglist(imgList,self.optFlowImgPrep)
#             return 1
#         except:
#             msg=("Could not set imgList, probably wrong input type or empty list")
#             self._warning(msg)
#             return 0
# 
#     def set_and_init_pcs_line_optflow(self, line, **kwargs):
#         """Set the line of interest for optical flow calculations
#         
#         :param LineOnImage line: line object
#         :param **kwargs: accepted keys (addTop, addBottom, addLeft, addRight)
#         
#         .. note::
#         
#             a ROI is determined automatically such that it includes the 
#             rectangle spanned by the line on the image using 
#             :func:`LineOnImage.set_roi`, **kwargs (expand line borders) are 
#             passed to :func:`LineOnImage.set_roi`
#         
#         """
#         self.optFlowAnalysis.set_line(line)
#         self.optFlowAnalysis.set_main_roi_from_line(**kwargs)
#         self.optFlowAnalysis.update_flow_roi(line.id)
#         
#         
#     def set_meas_geometry(self, measGeometry):
#         """Try setting :class:`piscope.Utils.MeasGeometry` object used 
#         for wind field analysis
#         """
#         try: 
#             if not measGeometry.basicDataAvailable:
#                 msg=("Could not set measGeometry, basic data not available")
#                 self._warning(msg)
#                 return 0
#                 
#             self.measGeometry=measGeometry
#             return 1
#         except:
#             msg=("Could not set measGeometry, probably wrong input type")
#             self._warning(msg)
#             return 0
#            
#     def _warning(self,msg):
#         self.warnings.append(msg)
#         print msg
#         
# class WindVector2D(object):
#     """Object representing a 2D wind vector"""
#     def __init__(self,v,vErr,orientation, orientationErr,unit="m/s"):
#         self.v=v
#         self.vErr=vErr
#         self.orientation=orientation
#         self.orientationErr=orientationErr
#         self.unit=unit
#     
#     def __call__(self):
#         """On call, return velocity information
#         """
#         return (self.v,self.vErr)
#         
#     def __str__(self):
#         s = "Wind vector info\n------------------\n"
#         s = s + "v (+/-): " + str(self.v) + "+/-" + str(self.vErr) + " " + self.unit + "\n"
#         s = s + "Angle [deg] (+/-) " + str(self.orientation) + "+/-" + str(self.orientationErr) + "\n"
#         return s
#==============================================================================