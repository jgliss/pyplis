# -*- coding: utf-8 -*-
"""
piscope example script no. 2

Import plume image dataset from example script 1 and illustrate some important
features of the MeasGeometry class
"""
from geonum.base import GeoPoint
from matplotlib.pyplot import subplots, show, close
from os.path import join

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex01_measurement_setup_plume_data import create_dataset

### SCRIPT FUNCTION DEFINITIONS        
def correct_viewing_direction(meas_geometry, draw_result = True):
    """Correct viewing direction using location of Etna SE crater
    
    Defines location of Etna SE crater within images (is plotted into current
    plume onband image of dataset) and uses its geo location to retrieve the 
    camera viewing direction
    
    :param meas_geometry: :class:`MeasGeometry` object
    
    """
    se_crater_img_pos = [806, 736] #x,y pos in image
    
    # Create geo point with coordinates (extracted from Google Earth)
    se_crater = GeoPoint(37.747757, 15.002643, name = "SE crater")
    
    print "Retrieved altitude (SRTM): %s" %se_crater.altitude
    
    meas_geometry.geo_setup.add_geo_point(se_crater)
    
    _, _, _, basemap =\
    meas_geometry.correct_viewing_direction(pix_x=se_crater_img_pos[0], 
                                            pix_y=se_crater_img_pos[1],
                                            pix_pos_err=100, #for uncertainty estimate
                                            obj_id="SE crater",
                                            draw_result=draw_result,
                                            update=True)
    return meas_geometry, basemap
    
def plot_plume_distance_image(meas_geometry):
    """Determines and plots image where each pixel corresponds to the plume 
    distance"""
    dist_img, _, plume_dist_img = meas_geometry.get_all_pix_to_pix_dists()
    fig, ax = subplots(1, 2, figsize = (16,4))
    disp0 = ax[0].imshow(dist_img.img, cmap = "gray")
    ax[0].set_title("Parametrised pixel to pixel distances")
    cb0 = fig.colorbar(disp0, ax =ax[0], shrink = 0.9)
    cb0.set_label("Pixel to pixel distance [m]")
    disp1 = ax[1].imshow(plume_dist_img.img / 1000.0, cmap = "gray")
    cb1 = fig.colorbar(disp1, ax =ax[1], shrink = 0.9)
    cb1.set_label("Plume distance [km]")
    ax[1].set_title("Retrieved plume distances")
    return fig

### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    
    ds = create_dataset()
    geom_corr, map = correct_viewing_direction(ds.meas_geometry)

    fig =  plot_plume_distance_image(ds.meas_geometry)
    
    ### IMPORTANT STUFF FINISHED    
    
    if SAVEFIGS:
        map.ax.figure.savefig(join(SAVE_DIR, "ex02_out_1.%s" %FORMAT), format=FORMAT,
                              dpi=DPI)
        fig.savefig(join(SAVE_DIR, "ex02_out_2.%s" %FORMAT), format=FORMAT,
                    dpi=DPI)
    
    
    
    # Display images or not (nothing to understand here...)
    (options, args)   =  OPTPARSE.parse_args()
    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"
    
    
    
    
    
    