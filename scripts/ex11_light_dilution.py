# -*- coding: utf-8 -*-
"""
piscope example script no. 10 - Image based light dilution correction
"""
import piscope as piscope
from geonum.base import GeoPoint
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from piscope.dilutioncorr import get_topo_dists_lines, get_extinction_coeff,\
    perform_dilution_correction
plt.close("all")

from ex1_measurement_setup_plume_data import img_dir
from ex10_bg_image_lists import get_bg_image_lists
skip_pix_line = 10

def create_dataset_dilution(start = datetime(2015, 9, 16, 6, 43, 00),\
                            stop = datetime(2015, 9, 16, 6, 47, 00)):
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

def prepare_list(lst, bg_corr_mode = 6):
    onlist = ds.get_list("on")
    offlist = ds.get_list("off")
    bg_onlist, bg_offlist = get_bg_image_lists() #dark_corr_mode already active
    
    onlist.dark_corr_mode = True
    onlist.add_gaussian_blurring(2)
    offlist.dark_corr_mode = True
    offlist.add_gaussian_blurring(2)
    
    onlist.set_bg_img(bg_onlist.current_img())
    offlist.set_bg_img(bg_offlist.current_img())
    onlist.bg_img.add_gaussian_blurring(2)
    offlist.bg_img.add_gaussian_blurring(2)
    onlist.bg_model.CORR_MODE = bg_corr_mode
    
if __name__ == "__main__":
    plt.close("all")
    ds = create_dataset_dilution()
    geom = correct_view_dir(ds.meas_geometry, which_crater = "ne")
    
    #INCLUDE DARK OFFSET CORR
    onlist = ds.get_list("on")
    offlist = ds.get_list("off")
    bg_onlist, bg_offlist = get_bg_image_lists() #dark_corr_mode already active
    
    onlist.darkcorr_mode = True
    onlist.add_gaussian_blurring(2)
    offlist.darkcorr_mode = True
    offlist.add_gaussian_blurring(2)
    
    onlist.set_bg_img(bg_onlist.current_img())
    offlist.set_bg_img(bg_offlist.current_img())
    onlist.bg_img.add_gaussian_blurring(2)
    offlist.bg_img.add_gaussian_blurring(2)
    onlist.bg_model.CORR_MODE = 6
    onlist.bg_model.guess_missing_settings(onlist.current_img())
    onlist.bg_model.xgrad_line_startcol = 10
    
    offlist.bg_model.update(**onlist.bg_model.settings_dict())
    

    onlist.tau_mode = True
    offlist.tau_mode = True
    
    tau_on = onlist.current_img().duplicate()
    tau_off = offlist.current_img().duplicate()
    
    onlist.bg_model.plot_tau_result(onlist.current_img()).suptitle(r"$\tau_{on}$")
    offlist.bg_model.plot_tau_result(offlist.current_img()).suptitle(r"$\tau_{off}$")
    
    onlist.aa_mode = True
    tau_mask = piscope.Img(onlist.current_img().img > 0.03)
    tau_mask.show()
    onlist.aa_mode = False
    
    bg_on = onlist.current_img() * np.exp(tau_on.img)
    bg_off = offlist.current_img() * np.exp(tau_off.img)
    
    go_on = 1
    if go_on:
        
        onlist.vigncorr_mode = True
        offlist.vigncorr_mode = True
        
        on_vigncorr= onlist.current_img()
        off_vigncorr = offlist.current_img()
        
        
    #==============================================================================
    #     lines = [piscope.processing.LineOnImage(1100, 650, 1100, 900,\
    #                                                             line_id= "far")]
    #==============================================================================
        lines = [piscope.processing.LineOnImage(1196, 650, 1196, 850,\
                                                                line_id= "far")]
        lines.append(piscope.processing.LineOnImage(1090, 990, 1100, 990,\
                                                            line_id= "close"))
    
        dists, access_mask, map, ax1 = get_topo_dists_lines(lines, geom, on_vigncorr,\
                                    skip_pix = skip_pix_line, plot = True)
        dists = dists[access_mask]
        rads_on = []
        rads_off = []
        for line in lines:
            rads_on.extend(line.get_line_profile(on_vigncorr))
            rads_off.extend(line.get_line_profile(off_vigncorr))
        
        rads_on = np.asarray(rads_on)[::int(skip_pix_line)][access_mask]
        rads_off = np.asarray(rads_off)[::int(skip_pix_line)][access_mask]    
        
        ia_roi = [1170, 340, 1300, 540]
        ia_on = on_vigncorr.crop(ia_roi, True).mean()
        ia_off = off_vigncorr.crop(ia_roi, True).mean()
        
        ext_on, i0_on, fit_res, ax2 = get_extinction_coeff(rads_on, dists, ia_on)
        ax2.set_title("310 nm")
        ext_off, i0_off, fit_res, ax3 = get_extinction_coeff(rads_off, dists, ia_off)
        ax3.set_title("330 nm")
        
        dist_img, plume_dist_img = geom.get_all_pix_to_pix_dists()  
        on_corr = perform_dilution_correction(on_vigncorr.img, ext_on,\
            bg_on.img, plume_dist_img.img, tau_mask.img)
        off_corr = perform_dilution_correction(off_vigncorr.img, ext_off,\
            bg_off.img, plume_dist_img.img, tau_mask.img)
        piscope.Img(off_corr).show()
        
        pcs_line = piscope.processing.LineOnImage(*[530, 586,910,200], line_id = "pcs")
        
        aa_uncorr = tau_on - tau_off
        aa_corr = piscope.Img(np.log(bg_on.img / on_corr)-
                                np.log(bg_off.img / off_corr))
        
        fig, ax = plt.subplots(1,3, figsize=(18,7))
        aa_uncorr.show(ax = ax[0])
        aa_corr.show(ax = ax[1])
        p0 = pcs_line.get_line_profile(aa_uncorr.img)
        p1 = pcs_line.get_line_profile(aa_corr.img)
        num = len(p0)
        
        phi0 = sum(p0)/num
        phi1 = sum(p1)/num
        ax[2].plot(p0, "-", label = r"Uncorrected: $\phi=%.3f$" %(phi0))
        ax[2].plot(p1, "-", label = r"Corrected: $\phi=%.3f$" %(phi1))
        

        
        

    
