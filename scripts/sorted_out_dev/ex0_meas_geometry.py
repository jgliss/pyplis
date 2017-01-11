# -*- coding: utf-8 -*-
"""
piscope example script no. 0

S
"""
import piscope as piscope
from datetime import datetime
from geonum.base import GeoPoint

from os.path import join, exists
from os import getcwd

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

### Define paths of example plume and background image
# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")

if not exists(img_dir):
    raise IOError("Failed to access test data")

def create_dataset():
    start = datetime(2015,9,16,7,6,00)
    stop = datetime(2015,9,16,7,22,00)
    
    ### Define camera
    cam_id = "ecII"
    
    #the camera filter setup
    filters= [piscope.utils.Filter(type = "on", acronym = "F01"),
              piscope.utils.Filter(type = "off", acronym = "F02")]
    
    #camera location and viewing direction (altitude will be retrieved automatically)                    
    geom_cam = {"lon"           :   15.1129,
                "lat"           :   37.73122,
                "elev"          :   15.0,
                "elevErr"       :   5.0,
                "azim"          :   274.0,
                "azimErr"       :   10.0,
                "alt_offset"    :   15.0} #altitude offset (above topography)
    
    
    #Camera height in m with respect to topographic altitude  at site
    #(We were standing on the roof of a building, guessed 20m)
    #create camera setup
    cam = piscope.setup.Camera(cam_id = cam_id, geom_data = geom_cam,\
                filter_list = filters, focal_length = 25.0)
    
    ### Load default information for Etna
    source = piscope.setup.Source("etna") 
    
    #### Provide wind direction
    wind_info= {"dir"     : 0.0,
               "dir_err"  : 15.0}


    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = piscope.setup.MeasSetup(img_dir, start, stop, camera=cam,\
                        source = source, wind_info = wind_info)
    
    ### Create analysis object (from BaseSetup)
    return piscope.dataset.Dataset(stp)

def correct_viewing_direction(dataset):
    """Correct viewing direction using location of Etna SE crater
    
    Defines location of Etna SE crater within images (is plotted into current
    plume onband image of dataset) and uses its geo location to retrieve the 
    camera viewing direction
    """
    geom = dataset.meas_geometry
    se_crater_img_pos = [806, 736] #x,y
    se_crater = GeoPoint(37.747757, 15.002643, name = "SE crater")
    se_alt_googleearth = 3.267 #km
    
    print "Retrieved altitude (SRTM): %s" %se_crater.altitude
    print "Altitude google earth: %s" %se_alt_googleearth
    
    geom.geo_setup.add_geo_point(se_crater)
    
    return geom.correct_viewing_direction(se_crater_img_pos[0],\
                        se_crater_img_pos[1], obj_id = "SE crater",\
                        draw_result = True)
    
if __name__ == "__main__":
    ds = create_dataset()
    
    elev, azim, geom_old = correct_viewing_direction(ds)
    