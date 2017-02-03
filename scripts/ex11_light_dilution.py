# -*- coding: utf-8 -*-
"""
piscope example script no. 10 - Image based light dilution correction
"""
import piscope as piscope
from geonum.base import GeoPoint
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from os.path import join, exists

from piscope.dilutioncorr import DilutionCorr
from piscope.doascalib import DoasCalibData

### IMPORT GLOBAL SETTINGS
from SETTINGS import IMG_DIR, SAVEFIGS, SAVE_DIR, FORMAT, DPI

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex10_bg_image_lists import get_bg_image_lists

### SCRIPT OPTONS  
# lower boundary for I0 value in dilution fit
I0_MIN = 0.0 

# Retrieval lines for dilution correction (along these lines, topographic
# distances and image radiances are determined for fitting the atmospheric
# extinction coefficients)
PCS1 = piscope.processing.LineOnImage(1100, 650, 1000, 900, line_id="far")
PCS2 = piscope.processing.LineOnImage(1000, 990, 1100, 990, line_id= "close")

# all lines in this array are used for the analysis
USE_LINES = [PCS1, PCS2]

# specify pixel resolution of topographic distance retrieval (every nth pixel 
# is used)
SKIP_PIX_LINES = 10

# Specify region of interest used to extract the ambient intensity (required
# for dilution correction)
AMBIENT_ROI = [1240, 10, 1300, 70]

# Specify plume velocity (for emission rate estimate)
PLUME_VELO = 4.14 #m/s (result from ex8)
SO2_MMOL = piscope.fluxcalc.MOL_MASS_SO2

### RELEVANT DIRECTORIES AND PATHS

CALIB_FILE = join(SAVE_DIR, "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts")

### SCRIPT FUNCTION DEFINITIONS        
def create_dataset_dilution():
    """Create a :class:`piscope.dataset.Dataset` object for dilution analysis
    
    The test dataset includes one on and one offband image which are recorded
    around 6:45 UTC at lower camera elevation angle than the time series shown
    in the other examples (7:06 - 7:22 UTC). Since these two images contain
    more topographic features they are used to illustrate the image based 
    signal dilution correction.
    
    This function sets up the measurement (geometry, camera, time stamps) for
    these two images and creates a Dataset object.
    """
    
    start = datetime(2015, 9, 16, 6, 43, 00)
    stop = datetime(2015, 9, 16, 6, 47, 00)
    #the camera filter setup
    cam_id = "ecII"
    filters= [piscope.utils.Filter(type = "on", acronym = "F01"),
              piscope.utils.Filter(type = "off", acronym = "F02")]
    
    geom_cam = {"lon"           :   15.1129,
                "lat"           :   37.73122,
                "elev"          :   15.0, #from field notes, will be corrected
                "elev_err"      :   5.0,
                "azim"          :   274.0, #from field notes, will be corrected 
                "azim_err"      :   10.0,
                "focal_length"  :   25e-3,
                "alt_offset"    :   7} #meters above topography

    #create camera setup
    cam = piscope.setupclasses.Camera(cam_id=cam_id, filter_list=filters,
                                      **geom_cam)
    
    ### Load default information for Etna
    source = piscope.setupclasses.Source("etna") 
    
    #### Provide wind direction
    wind_info= {"dir"      : 0.0,
                "dir_err"  : 15.0}


    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = piscope.setupclasses.MeasSetup(IMG_DIR, start, stop, camera=cam,
                                         source = source,
                                         wind_info = wind_info)
    return piscope.dataset.Dataset(stp)                  

def correct_view_dir(geom):
    """Performs a correction of the viewing direction using crater in img
    
    :param MeasGeometry geom: measurement geometry
    :param str which_crater: use either "ne" (northeast) or "se" (south east)
    :return: - MeasGeometry, corrected geometry
    """
    # Use position of NE crater in image
    posx, posy = 1051, 605 #pixel position of NE crate in image
    # Geo location of NE crater (info from Google Earth)
    ne_crater = GeoPoint(37.754788,  14.996673, 3287, name = "NE crater")
    
    geom.correct_viewing_direction(pix_x=posx, pix_y=posy, pix_pos_err=100,
                                   geo_point=ne_crater, draw_result=True)
    return geom

def prepare_lists(dataset):
    """Prepare on and off lists for dilution analysis
    
    Steps:
        
        1. get on and offband list
        #. load background image list on and off (from ex10)
        #. set image preparation and assign background images to on / off list
        #. configure plume background model settings
    
    :param Dataset dataset: the dilution dataset (see 
        :func:`create_dataset_dilution`)
    :return:
        - ImgList, onlist
        - ImgList, offlist
        
    """
    onlist = dataset.get_list("on")
    offlist = dataset.get_list("off")
    bg_onlist, bg_offlist = get_bg_image_lists() #dark_corr_mode already active
    
    #prepare img pre-edit
    onlist.darkcorr_mode = True
    onlist.add_gaussian_blurring(2)
    offlist.darkcorr_mode = True
    offlist.add_gaussian_blurring(2)
    
    #prepare background images in lists
    onlist.bg_img = bg_onlist.current_img()
    offlist.bg_img = bg_offlist.current_img()
    onlist.bg_img.add_gaussian_blurring(2)
    offlist.bg_img.add_gaussian_blurring(2)
    
    #prepare plume background modelling setup in both lists
    onlist.bg_model.CORR_MODE = 6
    onlist.bg_model.guess_missing_settings(onlist.current_img())
    onlist.bg_model.xgrad_line_startcol = 10
    offlist.bg_model.update(**onlist.bg_model.settings_dict())
    
    return onlist, offlist

def prepare_images(onlist, offlist):
    """Prepare all relevant images for dilution correction
    
    :param ImgList onlist: on band image list (prepared, see 
                                                    :func:`prepare_lists`)
    :param ImgList offlist: off band image list (prepared, see 
                                                    :func:`prepare_lists`)     
    :return:
        - Img, vignetting corrected on band image
        - Img, vignetting corrected off band image
        - Img, plume background image on band
        - Img, plume background image off band
        - Img, plume pixel mask
        - Img, tau on band image
        - Img, tau off band image
        
    """
    # Determine and store a tau image for on band and off band. This is used
    # to retrieve the plume background map
    onlist.tau_mode = True
    offlist.tau_mode = True
    
    tau_on = onlist.current_img().duplicate()
    tau_off = offlist.current_img().duplicate()
    
    # plot the tau images
    onlist.bg_model.plot_tau_result(onlist.current_img()).suptitle(r"$\tau_{on}$")
    offlist.bg_model.plot_tau_result(offlist.current_img()).suptitle(r"$\tau_{off}$")
    
    # now activate AA mode to determine a pixel mask for the dilution correction
    onlist.aa_mode = True
    tau_mask = piscope.Img(onlist.current_img().img > 0.03)
    tau_mask.img[840:,:] = 0 #remove tree in lower part of the image
    tau_mask.show()
    # deactivate AA mode
    onlist.aa_mode = False
    
    # activate vignetting correction mode in lists and load the two vignetting
    # corrected plume images
    onlist.vigncorr_mode = True
    offlist.vigncorr_mode = True
    
    on_vigncorr= onlist.current_img()
    off_vigncorr = offlist.current_img()
    
    # retrieve plume background intensity map from the two vignetting corrected
    # images (and from the two tau images determined above)
    bg_on = onlist.current_img() * np.exp(tau_on.img)
    bg_off = offlist.current_img() * np.exp(tau_off.img)
    
    return on_vigncorr, off_vigncorr, bg_on, bg_off, tau_mask, tau_on, tau_off

def plot_lines_into_image(img):
    ax = img.show()
    ax.set_title("Retrieval lines")
    for line in USE_LINES:
        line.plot_line_on_grid(ax=ax, marker="", color="lime")
    
    ax.legend(loc="best", framealpha=0.5, fancybox= True, fontsize = 10)    
    return ax

### SCRIPT MAIN FUNCTION       
if __name__ == "__main__":
    if not exists(CALIB_FILE):
        raise IOError("Calibration file could not be found at specified "
            "location:\n %s\nYou might need to run example 6 first")

    plt.close("all")
    
    calib = DoasCalibData()    
    calib.load_from_fits(CALIB_FILE)
    
    # create dataset and correct viewing direction
    ds = create_dataset_dilution()
    geom = correct_view_dir(ds.meas_geometry)
    
    #get plume distance image    
    pix_dists, _, plume_dist_img = geom.get_all_pix_to_pix_dists()  
    
    #prepare on and offband list
    onlist, offlist = prepare_lists(ds)
    
    #prepare all relevant images for dilution correction
    on_vigncorr, off_vigncorr, bg_on, bg_off, tau_mask, tau_on, tau_off =\
                                                prepare_images(onlist, offlist)
    
    # Plot the retrieval lines into on band image
    ax0 = plot_lines_into_image(on_vigncorr)
    
    # Create dilution correction class
    dil = DilutionCorr(USE_LINES, geom, skip_pix=SKIP_PIX_LINES)
    
    # Determine distances to the two lines defined above (every 6th pixel)
    for line_id in dil.line_ids:
        dil.det_topo_dists_line(line_id)
    
    # Plot the results in a 3D map
    basemap = dil.plot_distances_3d(alt_offset_m = 10, axis_off = False)
    
    # exemplary plume cross section line for emission rate retrieval
    pcs_line = piscope.processing.LineOnImage(x0=530,y0=586,x1=910,y1=200,
                                              line_id="pcs")                                                          
    
    # retrieve pixel distances for pixels on the line 
    # (for emission rate estimate)
    pix_dists_line = pcs_line.get_line_profile(pix_dists)
    
    #get pixel coordinates of PCS center position ...
    col, row = pcs_line.center_pix
    
    # ... and get uncertainty in plume distance estimate for the column
    pix_dist_err = geom.pix_dist_err(col)
    
    fig, ax = plt.subplots(2, 2, figsize = (12,8))
    
    ia_on = on_vigncorr.crop(AMBIENT_ROI, True).mean()
    ia_off = off_vigncorr.crop(AMBIENT_ROI, True).mean()
    
    ext_on, i0_on, _, _ = dil.apply_dilution_fit(img=on_vigncorr,
                                                 rad_ambient=ia_on, 
                                                 i0_min=I0_MIN,
                                                 plot=True,
                                                 ax=ax[0, 0])
                                                 
    ax[0, 0].set_title(r"On: $I_A$ = %.1f DN" %(ia_on))        
    
    ext_off, i0_off, _, _ = dil.apply_dilution_fit(img=off_vigncorr,
                                                   rad_ambient=ia_off,
                                                   i0_min=I0_MIN,
                                                   plot=True,
                                                   ax=ax[0, 1])
                                                   
    ax[0, 1].set_title(r"Off: $I_A$ = %.1f DN" %(ia_off), fontsize = 12)        
    
    
    #determine uncorrected so2-CD image by calibrating the AA image 
    so2_img_uncorr = calib(tau_on - tau_off)
    
    so2_cds_uncorr = pcs_line.get_line_profile(so2_img_uncorr)
    
    # Calculate flux and uncertainty
    phi_uncorr, phi_uncorr_err =\
        piscope.fluxcalc.det_emission_rate(cds=so2_cds_uncorr,
                                           velo=PLUME_VELO,
                                           pix_dists=pix_dists_line,
                                           cds_err=calib.slope_err,
                                           pix_dists_err=pix_dist_err)
                                           
    on_corr = dil.correct_img(on_vigncorr, ext_on, bg_on,
                              plume_dist_img, tau_mask)
                              
    tau_on_corr = piscope.Img(np.log(bg_on.img / on_corr.img))
    
    off_corr = dil.correct_img(off_vigncorr, ext_off, bg_off,
                              plume_dist_img, tau_mask)
                              
    tau_off_corr = piscope.Img(np.log(bg_off.img / off_corr.img))
    
    so2_img_corr = calib(tau_on_corr - tau_off_corr)
    so2_img_corr.edit_log["is_tau"] = True #for plotting
    so2_cds_corr = pcs_line.get_line_profile(so2_img_corr)
    
    phi_corr, phi_corr_err =\
        piscope.fluxcalc.det_emission_rate(cds=so2_cds_corr,
                                           velo=PLUME_VELO,
                                           pix_dists=pix_dists_line,
                                           cds_err=calib.slope_err,
                                           pix_dists_err=pix_dist_err)
                                           
    so2_img_corr.show(ax = ax[1, 0])
    
    ax[1, 0].set_title("Dilution corrected AA image", fontsize = 12)
    pcs_line.plot_line_on_grid(ax = ax[1, 0], ls="-", color = "g")
    x0, y0, w, h = piscope.helpers.roi2rect(AMBIENT_ROI)
    ax[1, 0].add_patch(plt.Rectangle((x0, y0), w, h, fc = "none", ec = "c"))
    
    # Calculate flux and uncertainty                                    
    ax[1,1].plot(so2_cds_uncorr, "--b", label=r"Uncorr: $\Phi_{SO2}=$"
        "%.2f (+/- %.2f) kg/s" %(phi_uncorr/1000.0, phi_uncorr_err/1000.0))
    ax[1,1].plot(so2_cds_corr, "-g", label=r"Corr: $\Phi_{SO2}=$"
        "%.2f (+/- %.2f) kg/s" %(phi_corr/1000.0, phi_corr_err/1000.0))
    
    ax[1,1].set_title("Cross section profile", fontsize = 12)
    ax[1,1].legend(loc="best", framealpha=0.5, fancybox= True, fontsize = 10)
    ax[1,1].set_xlim([0, len(pix_dists_line)])
    ax[1,1].set_ylim([0, 5e18])
    
    if SAVEFIGS:    
        ax0.figure.savefig(join(SAVE_DIR, "ex11_out_1.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        basemap.ax.set_axis_off()
        basemap.ax.figure.savefig(join(SAVE_DIR, "ex11_out_2.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        fig.savefig(join(SAVE_DIR, "ex11_out_3.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)

        


