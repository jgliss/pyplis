# -*- coding: utf-8 -*-
"""
piscope example script no. 10 - Image based light dilution correction
"""
import piscope as piscope
from geonum.base import GeoPoint
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.ma import masked_where
from os.path import join

plt.close("all")

from ex1_measurement_setup_plume_data import img_dir, save_path

skip_pix_line = 15
color = "r"
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
                "alt_offset"    :   20} #meters above topography

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
#==============================================================================
# on_path = join(img_dir, "EC2_1106307_1R02_2015091606454457_F01_Etna.fts")
# off_path = join(img_dir, "EC2_1106307_1R02_2015091606454717_F02_Etna.fts")
#==============================================================================
ds = create_dataset_dilution()

#INCLUDE DARK OFFSET CORR
on_list = ds.get_list("on")
off_list = ds.get_list("off")

# Line definitions from manuscript

line = piscope.processing.LineOnImage(672, 624,672, 950, line_id= "cfov")

geom = ds.meas_geometry
se_crater_img_pos = [735, 575] #x,y
se_crater = GeoPoint(37.747757, 15.002643, name = "SE crater")
geom.geo_setup.add_geo_point(se_crater)

elev_new, az_new, _, map = geom.correct_viewing_direction(\
    se_crater_img_pos[0], se_crater_img_pos[1], obj_id = "SE crater",\
                                                    draw_result =  True)
                                                    
ax = on_list.show_current()
line.plot_line_on_grid(ax = ax, color = color, marker = "")
ax.set_xlim([0, 1343])
ax.set_ylim([1023, 0])

l = line.to_list() #line coords as list
res = geom.get_distances_to_topo_line(l, skip_pix = skip_pix_line)

#Create 3D map of scene
map3d = geom.draw_map_3d(0, 0, 0, 0)
#insert camera position into 3D map
geom.cam_pos.plot_3d(map = map3d, add_name = True, dz_text = 40)

fig_dists, ax = plt.subplots(1,1, figsize = (12,8))

handles = []
handles2 = []
#now draw the lines into the plume raw image and into the 3D map
    
#plot line into image
l2d = plt.Line2D([l[0], l[2]],[l[1],l[3]], color = color, label =\
                                                    line.line_id)
handles.append(l2d)

#boolean mask for accessing data for which distance retrieval worked
mask = res["ok"]

y, yErr = res["dists"], res["dists_err"]
num = len(y)
handles2.append(ax.plot(masked_where(~mask, y), "--x",\
                    color = color, label = line.line_id)[0])
ax.set_xlim([0, num - 1])
ax.get_xaxis().set_ticks([0, num - 1])
ax.grid()
ax.get_xaxis().set_ticklabels(["top", "bottom"], rotation = 15)

pts = res["geo_points"][mask]
map3d.add_geo_points_3d(pts, color = color)

ax.legend(handles = handles, loc = 'best', fancybox = True,\
                                framealpha = 0.5, fontsize = 16).draggable()

map3d.ax.set_axis_off()
ax.set_title("Distance retrievals")
ax.set_xlabel("Position in image", fontsize = 16)
ax.set_ylabel("Distance [km]", fontsize = 16)

ax.figure.savefig(join(save_path, "ex10_out_1.png"))
fig_dists.savefig(join(save_path, "ex10_out_2.png"))
map3d.ax.figure.savefig(join(save_path, "ex10_out_3.png"))
plt.show()
