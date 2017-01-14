# -*- coding: utf-8 -*-
"""
piscope example script no. 2

Import plume image dataset from example script 1 and illustrate some important
features of the MeasGeometry class
"""
import piscope as piscope
from geonum.base import GeoPoint
from matplotlib.pyplot import subplots, close

from os.path import join, exists
from os import getcwd

from ex1_measurement_setup_plume_data import create_dataset

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")

if not exists(img_dir):
    raise IOError("Failed to access test data")

def correct_viewing_direction(meas_geometry):
    """Correct viewing direction using location of Etna SE crater
    
    Defines location of Etna SE crater within images (is plotted into current
    plume onband image of dataset) and uses its geo location to retrieve the 
    camera viewing direction
    
    :param meas_geometry: :class:`MeasGeometry` object
    
    """
    se_crater_img_pos = [806, 736] #x,y
    se_crater = GeoPoint(37.747757, 15.002643, name = "SE crater")
    se_alt_googleearth = 3.267 #km
    
    print "Retrieved altitude (SRTM): %s" %se_crater.altitude
    print "Altitude google earth: %s" %se_alt_googleearth
    
    meas_geometry.geo_setup.add_geo_point(se_crater)
    
    elev_new, az_new, _, map = meas_geometry.correct_viewing_direction(\
        se_crater_img_pos[0], se_crater_img_pos[1], obj_id = "SE crater",\
                                                        draw_result = True)
    return map
    
def plot_plume_distance_image(meas_geometry):
    """Determines and plots image where each pixel corresponds to the plume 
    distance"""
    dist_img, plume_dist_img = meas_geometry.get_all_pix_to_pix_dists()
    fig, ax = subplots(1, 2, figsize = (16,4))
    disp0 = ax[0].imshow(dist_img, cmap = "gray")
    ax[0].set_title("Parametrised pixel to pixel distances")
    cb0 = fig.colorbar(disp0, ax =ax[0], shrink = 0.9)
    cb0.set_label("Pixel to pixel distance [m]")
    disp1 = ax[1].imshow(plume_dist_img / 1000.0, cmap = "gray")
    cb1 = fig.colorbar(disp1, ax =ax[1], shrink = 0.9)
    cb1.set_label("Plume distance [km]")
    ax[1].set_title("Retrieved plume distances")
    return fig
    
if __name__ == "__main__":
    close("all")
    ds = create_dataset()
    map = correct_viewing_direction(ds.meas_geometry)
    fig = plot_plume_distance_image(ds.meas_geometry)
    
    map.ax.figure.savefig(join(save_path, "ex2_out_1.png"))
    fig.savefig(join(save_path, "ex2_out_2.png"))
    
    
    
    
    
    