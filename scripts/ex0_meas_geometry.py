# -*- coding: utf-8 -*-
"""
piSCOPE example / test script no.

This script shows how to setup and work with 

:class:`piscope.Utils.MeasGeometry` objects

using the test dataset from Etna 2015 

Measurement location: Milo

"""
import piscope as piscope
from datetime import datetime
from geonum.base import GeoPoint
from matplotlib.pyplot import Line2D
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from os.path import join, exists
from os import getcwd

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

### Define paths of example plume and background image
# Image base path
img_dir = "../test_data/piscope_etna_testdata/images/"

if not exists(img_dir):
    raise IOError("Test image directory does not exist")

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


#==============================================================================
# ### Define location of Etna NE crater (both geographically and in pix coordinates)
# NE = GeoPoint(37.754788, 14.996673, name = "NE_crater") 
# NE_img_pos = [1105, 605] #position of NE crater in image
# 
# 
# 
# ### get MeasGeometry object
# geom = stp.meas_geometry
# 
# ### Add NE crater to GeoSetup of MeasGeometry object
# geom.geo_setup.add_geo_point(NE)
# 
# ### Make a copy of the MeasGeometry before correcting viewing direction
# geom_0 = deepcopy(geom) 
# 
# ### Correct viewing direction in the geometry object based on pos of NE crater
# geom.correct_viewing_direction(NE_img_pos[0], NE_img_pos[1],\
#                                            obj_id = "NE_crater")
# 
# 
# 
# #==============================================================================
# # ### Add the 3 lines to the plume dataset
# # for k in range(len(lines)):
# #     plume_data.forms["lines"].add(*lines[k], id = lineIds[k])
# #==============================================================================
# 
# results = []  
# 
# for k in range(len(lines)):     
#     
#     results.append(geom.get_distances_to_topo_line(lines[k], skip_pix = skip[k],\
#                                     view_above_topo_m = camZOffset))
# 
# ###
# """
# In the following, distances to the topography are estimated for the pixels
# on the two lines defined above ("flank_close". "flank_far"), this is being done
# in the following steps (for each line respectively:
# 
#     1. Get azimuth and elevation angles for all lines
# """
# #get azimuth and elevation angles for the both lines
# 
# ### Plot the results
# 
# #Plot current plume image (on band)
# im = plume_data.current_image("on")   
# fig_img = plt.figure()  
# ax = fig_img.add_subplot(111)
# 
# im = ax.imshow(im.img, cmap = "gray")
# #ax = plt.imshow(im.img, cmap = "gray")
# #Create 3D map of scene
# map3d = geom.draw_map_3d(0, 0, 0)
# #insert camera position into 3D map
# geom.cam_pos.plot_3d(map = map3d, add_name = True, dz_text = 40)
# 
# fig_dists, axes = plt.subplots(3,1, figsize = (6,8))
# 
# handles = []
# handles2 = []
# #now draw the lines into the plume raw image and into the 3D map
# for k in range(len(results)):
#     res = results[k]
#     color = c[k]
#     
#     #plot line into image
#     v = lines[k]
#     l = Line2D([v[0],v[2]],[v[1],v[3]], color = color, label = lineIds[k])
#     handles.append(l)
#     ax.add_artist(l)
#     
#     #boolean mask for accessing data for which distance retrieval worked
#     mask = res["ok"]
# 
#     y, yErr = res["dists"], res["dists_err"]
#     handles2.append(axes[k].plot(np.ma.masked_where(~mask, y), "--x",\
#                                     color = color, label = lineIds[k])[0])
#     axes[k].set_xlim([0, len(y)-1])
#     num = len(y)
#     dd = num*.03
#     axes[k].get_xaxis().set_ticks([0, num - 1])
#     axes[k].grid()
#     axes[k].get_xaxis().set_ticklabels(lineLabels[k], rotation = 15)
# 
#     pts = res["geo_points"][mask]
#     
#     xs, ys, zs = [], [], []
#     for p in pts:
#         if isinstance(p, GeoPoint):
#             px, py= map3d(p.lon.decimal_degree,p.lat.decimal_degree)
#             xs.append(px), ys.append(py), zs.append(p.altitude)
#             map3d.draw_geo_point_3d(p, marker = "x", s= 20, c = color)
#     
#     map3d.ax.plot(xs, ys, zs, "--", c = color, lw = 2, zorder = 100000)
# 
# map3d.ax.set_axis_off()
# ax.legend(handles = handles, loc = 'best', fancybox = True,\
#                                 framealpha = 0.5, fontsize = 16).draggable()
# ax.set_axis_off()
# ax.set_xlim([0,1343])
# ax.set_ylim([1023, 0])
# 
# axes[0].set_title("Distance retrievals")
# axes[2].set_xlabel("Position in image", fontsize = 16)
# axes[1].set_ylabel("Distance [km]", fontsize = 16)
# 
# #fig.savefig(join(save_path, "ex0_3d_map.png"))
# 
# fig_viewcorr, ax = plt.subplots(1,2, figsize=(18,6))
# m0 = geom_0.draw_map_2d(ax=ax[0])
# m0.ax.set_title("Viewing dir before correction")
# 
# m1 = geom.draw_map_2d(ax = ax[1])
# m1.ax.set_title("Viewing dir after correction")
#==============================================================================

if __name__ == "__main__":
    ds = create_dataset()
    