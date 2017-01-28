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

from ex1_measurement_setup_plume_data import img_dir, save_path
from ex10_bg_image_lists import get_bg_image_lists

calib_file = join(save_path, "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts")

if not exists(calib_file):
    raise IOError("Calibration file could not be found at specified location:\n"
        "%s\nYou might need to run example 6 first")

plt.close("all")

skip_pix_line = 10

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
    cam = piscope.setup.Camera(cam_id = cam_id, filter_list = filters,\
                                                                **geom_cam)
    
    ### Load default information for Etna
    source = piscope.setup.Source("etna") 
    
    #### Provide wind direction
    wind_info= {"dir"      : 0.0,
                "dir_err"  : 15.0}


    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = piscope.setup.MeasSetup(img_dir, start, stop, camera=cam,\
                        source = source, wind_info = wind_info)
    return piscope.dataset.Dataset(stp)                  

def correct_view_dir(geom, which_crater = "ne"):
    """Performs a correction of the viewing direction using crater in img
    
    :param MeasGeometry geom: measurement geometry
    :param str which_crater: use either "ne" (northeast) or "se" (south east)
    :return: - MeasGeometry, corrected geometry
    """
    se_crater_img_pos = [735, 575] #x,y
    se_crater = GeoPoint(37.747757, 15.002643, 3267, name = "SE crater")
    geom.geo_setup.add_geo_point(se_crater)
    
    ne_crater_img_pos = [1051, 605] #x,y
    ne_crater = GeoPoint(37.754788,  14.996673, 3287, name = "NE crater")
    geom.geo_setup.add_geo_point(ne_crater)
    
    if which_crater == "se":
        obj_id = "SE crater"
        c = se_crater_img_pos
    else:
        obj_id = "NE crater"
        c = ne_crater_img_pos        
    elev_new, az_new, _, map = geom.correct_viewing_direction(\
                        c[0], c[1], obj_id = obj_id, draw_result =  True)
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


if __name__ == "__main__":
    from scipy.constants import Avogadro
    SO2_MMOL = 64 #g/mol
    NA = Avogadro #n/mol

    #Script options
    # lower boundary for I0 value in dilution fit
    I0_MIN = 0.0 
    #specify the lines to be used for distance retrieval (defined in the next
    #step)
    USE_LINES = ["far", "close"] 
    AMBIENT_ROIS = [[1240, 10, 1300, 70],
                    [1240, 260, 1300, 320],
                    [1240, 460, 1300, 520],]
    PLUME_VEL = 4.14 #m/s (result from ex8)
    
    calib = DoasCalibData()    
    calib.load_from_fits(calib_file)
    
    #create lines specifying topographic features used to perform dilution
    #correction                                                 
    lines = [piscope.processing.LineOnImage(1196, 650, 1196, 850,\
                                                            line_id= "far")]
    
    lines.append(piscope.processing.LineOnImage(1090, 990, 1100, 990,\
                                                        line_id= "close"))
    
    # create dataset and correct viewing direction
    ds = create_dataset_dilution()
    geom = correct_view_dir(ds.meas_geometry, which_crater = "ne")
    
    #get plume distance image    
    pix_dists, _, plume_dist_img = geom.get_all_pix_to_pix_dists()  
    
    #prepare on and offband list
    onlist, offlist = prepare_lists(ds)
    
    #prepare all relevant images for dilution correction
    on_vigncorr, off_vigncorr, bg_on, bg_off, tau_mask, tau_on, tau_off =\
                                                prepare_images(onlist, offlist)
    
    ax = on_vigncorr.show()
    ax.set_title("Vignetting corrected onband image")
    lines[0].plot_line_on_grid(ax = ax, marker="", color="r")
    lines[1].plot_line_on_grid(ax = ax, marker="")
    ax.legend(loc="best", framealpha=0.5, fancybox= True, fontsize = 10)
        
    #Create dilution correction class
    dil = DilutionCorr(lines, geom, skip_pix=6)
    #Retrieved distances to the two lines defined above (every 6th pixel)
    dil.det_topo_dists_line("close")
    dil.det_topo_dists_line("far")
    
    #Plot the results in a 3D map
    basemap = dil.plot_distances_3d(alt_offset_m = 10, axis_off = False)
    
    #exemplary plume cross section line
    pcs_line = piscope.processing.LineOnImage(*[530, 586,910,200],
                                                      line_id = "pcs")                                                          
    
    ax.figure.savefig(join(save_path, "ex11_out_1.png"))
    basemap.ax.figure.savefig(join(save_path, "ex11_out_2.png"))
    k=3
    for AMBIENT_ROI in AMBIENT_ROIS:
        aa_uncorr = tau_on - tau_off
        
        aa_noise = aa_uncorr.crop(AMBIENT_ROI, True)
        aa_noise_amp = aa_noise.max() - aa_noise.min()
        
        aa_profile_uncorr = pcs_line.get_line_profile(aa_uncorr)
        cond = aa_profile_uncorr > aa_noise_amp
        so2_cds = calib(aa_profile_uncorr[cond])*100**2 #per sqm
        pix_dists_line = pcs_line.get_line_profile(pix_dists)[cond]
        ica = sum(so2_cds * pix_dists_line)
        
        flux_uncorr = ica * PLUME_VEL * SO2_MMOL / NA / 1000.0 #kg/s
        
        fig, ax = plt.subplots(2, 2, figsize = (12,8))
        
        ia_on = on_vigncorr.crop(AMBIENT_ROI, True).mean()
        ia_off = off_vigncorr.crop(AMBIENT_ROI, True).mean()
        
        ext_on, i0_on, _, _ = dil.apply_dilution_fit(on_vigncorr, ia_on, 
                                                     i0_min=I0_MIN, 
                                                     line_ids=USE_LINES,
                                                     ax=ax[0, 0])
                                                     
        ax[0, 0].set_title(r"On: $I_A$ = %.1f DN" %(ia_on))        
        
        ext_off, i0_off, _, _ = dil.apply_dilution_fit(off_vigncorr, ia_off,
                                                       i0_min=I0_MIN, 
                                                       line_ids=USE_LINES,
                                                       ax=ax[0, 1])
        ax[0, 1].set_title(r"Off: $I_A$ = %.1f DN" %(ia_off), fontsize = 12)        
        
        on_corr = dil.correct_img(on_vigncorr, ext_on, bg_on,
                                  plume_dist_img, tau_mask)
                                  
        tau_on_corr = piscope.Img(np.log(bg_on.img / on_corr.img))
        
        off_corr = dil.correct_img(off_vigncorr, ext_off, bg_off,
                                  plume_dist_img, tau_mask)
                                  
        tau_off_corr = piscope.Img(np.log(bg_off.img / off_corr.img))
        
        aa_corr = tau_on_corr - tau_off_corr
        aa_corr.edit_log["is_tau"] = True #for plotting
        aa_profile_corr = pcs_line.get_line_profile(aa_corr)
        aa_corr.show(ax = ax[1, 0])
        ax[1, 0].set_title("Dilution corrected AA image", fontsize = 12)
        pcs_line.plot_line_on_grid(ax = ax[1, 0], ls="-", color = "g")
        x0, y0, w, h = piscope.helpers.roi2rect(AMBIENT_ROI)
        ax[1, 0].add_patch(plt.Rectangle((x0, y0), w, h, fc = "none", ec = "c"))
        
        
        so2_cds = calib(aa_profile_corr[cond])*100**2 #per sqm
        ica = sum(so2_cds * pix_dists_line)
        flux_corr = ica * PLUME_VEL * SO2_MMOL / NA / 1000.0 #kg/s
        ax[1,1].plot(aa_profile_uncorr, "--b", label = r"Flux (uncorr): $\phi=%.2f$ kg/s" 
                                                                %(flux_uncorr))
        ax[1,1].plot(aa_profile_corr, "-g", label = r"Flux (corr): $\phi=%.2f$ kg/s" 
                                                                %(flux_corr))
        ax[1,1].set_title("Cross section profile", fontsize = 12)
        ax[1,1].legend(loc="best", framealpha=0.5, fancybox= True, fontsize = 10)
        
        fig.savefig(join(save_path, ("ex11_out_%d.png" %k)))
        k+=1
            

    
