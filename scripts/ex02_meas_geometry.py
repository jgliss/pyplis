# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliß (jonasgliss@gmail.com)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Pyplis example script no. 2 - Features of the MeasGeometry class

In this script, some important features of the MeasGeometry class are 
introduced. The class itself is automatically created in the MeasSetup
object which was created in example script ex01_analysis_setup.py and was 
passed as input for a Dataset object. The relevant MeasGeometry class is stored 
within the Dataset object and can be accessed via the ``meas_geometry`` 
attribute.

As a first feature, the viewing direction of the camera is retrieved from the 
image using the position of the south east (SE) crater of Mt.Etna. The result 
is then visualized in a 2D map to give an overview of the geometry. The map
further includes the initial viewing direction (see example script 
ex01_analysis_setup.py) which was logged in the field using a compass and a 
mechanical inclinometer.
 
Further, the distance to the plume is retrieved on a pixel basis (represented 
as image).
"""
from SETTINGS import check_version
# Raises Exception if conflict occurs
check_version()

from geonum.base import GeoPoint
from matplotlib.pyplot import subplots, show, close
from os.path import join

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex01_analysis_setup import create_dataset

### SCRIPT FUNCTION DEFINITIONS    
def find_viewing_direction(meas_geometry, draw_result=True):
    """Correct viewing direction using location of Etna SE crater
    
    Defines location of Etna SE crater within images (is plotted into current
    plume onband image of dataset) and uses its geo location to retrieve the 
    camera viewing direction
    
    :param meas_geometry: :class:`MeasGeometry` object
    
    """
    # Position of SE crater in the image (x, y)
    se_crater_img_pos = [806, 736] 
    
    # Geographic position of SE crater (extracted from Google Earth)
    # The GeoPoint object (geonum library) automatically retrieves the altitude
    # using SRTM data 
    se_crater = GeoPoint(37.747757, 15.002643, name = "SE crater")
    
    print "Retrieved altitude SE crater (SRTM): %s" %se_crater.altitude
    
    # The following method finds the camera viewing direction based on the
    # position of the south east crater. 
    new_elev, new_azim, _, basemap =\
    meas_geometry.find_viewing_direction(pix_x=se_crater_img_pos[0], 
                                         pix_y=se_crater_img_pos[1],
                                         pix_pos_err=100, #for uncertainty estimate
                                         geo_point=se_crater,
                                         draw_result=draw_result,
                                         update=True) #overwrite old settings 
                                         
    print ("Updated camera azimuth and elevation in MeasGeometry, new values: "
            "elev = %.1f, azim = %.1f" %(new_elev, new_azim))
            
    return meas_geometry, basemap
    
def plot_plume_distance_image(meas_geometry):
    """Determines and plots image plume distance and pix-to-pix distance images"""
    # This function returns three images, the first corresponding to pix-to-pix
    # distances in horizontal direction and the second (ignored here) to
    # the vertical (in this case they are the same since pixel height and 
    # pixel width are the same for this detector). The third image gives 
    # camera to plume distances on a pixel basis
    (dist_img, _, plume_dist_img) = meas_geometry.compute_all_integration_step_lengths()
    
    fig, ax = subplots(1, 2, figsize = (16,4))
    
    
    # Show pix-to-pix distance image
    dist_img.show(cmap="gray", ax=ax[0], zlabel="Pixel to pixel distance [m]")
    ax[0].set_title("Parametrised pixel to pixel distances")
    
    # Show plume distance image (convert pixel values to from m -> km)
    (plume_dist_img / 1000.0).show(cmap="gray", ax=ax[1], zlabel="Plume distance [km]")
    ax[1].set_title("Retrieved plume distances")
    return fig

### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    
    # Create the Dataset object (see ex01)
    ds = create_dataset()
    
    # execute function defined above (see above for definition and information)
    geom_corr, map = find_viewing_direction(ds.meas_geometry)
    
    # execute 2. script function (see above for definition and information)
    fig =  plot_plume_distance_image(ds.meas_geometry)
    
    ### IMPORTANT STUFF FINISHED    
    if SAVEFIGS:
        map.ax.figure.savefig(join(SAVE_DIR, "ex02_out_1.%s" %FORMAT), 
                              format=FORMAT, dpi=DPI)
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
    
    
    
    
    
    