# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)
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
"""Pyplis example script no. 2 - Features of the MeasGeometry class.

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
from geonum import GeoPoint
import pathlib
import matplotlib.pyplot as plt

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex01_analysis_setup import create_dataset


# SCRIPT FUNCTION DEFINITIONS
def find_viewing_direction(meas_geometry, draw_result=True):
    """Correct viewing direction using location of Etna SE crater.

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
    se_crater = GeoPoint(37.747757, 15.002643, name="SE crater", auto_topo_access=True)

    print("Retrieved altitude SE crater (SRTM): %s" % se_crater.altitude)

    # The following method finds the camera viewing direction based on the
    # position of the south east crater.
    new_elev, new_azim, _, basemap =\
        meas_geometry.find_viewing_direction(pix_x=se_crater_img_pos[0],
                                             pix_y=se_crater_img_pos[1],
                                             # for uncertainty estimate
                                             pix_pos_err=100,
                                             geo_point=se_crater,
                                             draw_result=draw_result,
                                             update=True)  # overwrite settings

    print("Updated camera azimuth and elevation in MeasGeometry, new values: "
          f"elev = {new_elev:.1f}, azim = {new_azim:.1f}")

    return meas_geometry, basemap


def plot_plume_distance_image(meas_geometry):
    """Determine and plot image plume distance and pix-to-pix distance imgs."""
    # This function returns three images, the first corresponding to pix-to-pix
    # distances in horizontal direction and the second (ignored here) to
    # the vertical (in this case they are the same since pixel height and
    # pixel width are the same for this detector). The third image gives
    # camera to plume distances on a pixel basis
    (dist_img, _, plume_dist_img) =\
        meas_geometry.compute_all_integration_step_lengths()

    fig, ax = plt.subplots(2, 1, figsize=(7, 8))

    # Show pix-to-pix distance image
    dist_img.show(cmap="gray", ax=ax[0], zlabel="Pix-to-pix distance [m]")
    ax[0].set_title("Parameterised pix-to-pix dists")

    # Show plume distance image (convert pixel values to from m -> km)
    (plume_dist_img / 1000.0).show(cmap="gray", ax=ax[1], zlabel="Plume distance [km]")
    ax[1].set_title("Plume dists")
    return fig


# SCRIPT MAIN FUNCTION
def main():
    plt.close("all")
    
    # Create the Dataset object (see ex01)
    ds = create_dataset()

    # execute function defined above (see above for definition and information)
    geom_corr, map_ = find_viewing_direction(ds.meas_geometry)

    # execute 2. script function (see above for definition and information)
    fig = plot_plume_distance_image(ds.meas_geometry)

    # You can compute the plume distance for the camera CFOV pixel column just
    # by calling the method plume_dist() without specifying the input azimuth
    # angle...
    plume_dist_cfov = geom_corr.plume_dist()[0][0]

    # ... and the corresponding uncertainty
    plume_dist_err_cfov = geom_corr.plume_dist_err()

    # You can also retrieve an array containing the camera azimuth angles for
    # each pixel column...
    all_azimuths = geom_corr.all_azimuths_camfov()

    # ... and use this to compute plume distances on a pixel column basis
    plume_dists_all_cols = geom_corr.plume_dist(all_azimuths)

    # If you want, you can update information about camera, source or
    # meteorolgy using either of the following methods (below we apply a
    # change in the wind-direction such that the plume propagation direction
    # is changed from S to SE )
    # geom_corr.update_cam_specs()
    # geom_corr.update_source_specs()

    geom_corr.update_wind_specs(dict(dir=315))
    geom_corr.draw_map_2d()  # this figure is only displayed and not saved

    # recompute plume distance of CFOV pixel
    plume_dist_cfov_new = geom_corr.plume_dist()[0][0]

    print("Comparison of plume distances after change of wind direction:\n"
          f"Previous: {plume_dist_cfov:.3f}m\n"
          f"New: {plume_dist_cfov_new:.3f}m")
    
    # IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        outfile = SAVE_DIR / f"ex02_out_1.{FORMAT}"
        map_.ax.figure.savefig(outfile,format=FORMAT,dpi=DPI)

        outfile = SAVE_DIR / f"ex02_out_2.{FORMAT}"
        fig.savefig(outfile,format=FORMAT,dpi=DPI)

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        # check some propoerties of the basemap (displayed in figure)

        # map basemap coordinates to lon / lat values
        lon, lat = map_(8335, 9392, inverse=True)
        npt.assert_allclose(actual=[lon, lat, map_.delta_lon, map_.delta_lat],
                            desired=[15.058292, 37.753504,  0.182902,  0.145056],
                            rtol=1e-5)

        actual=[geom_corr.cam_elev,
                geom_corr.cam_elev_err,
                geom_corr.cam_azim,
                geom_corr.cam_azim_err,
                plume_dist_cfov,
                plume_dist_err_cfov,
                plume_dists_all_cols.mean(),
                plume_dist_cfov_new
        ]
        
        desired=[
                1.547754e+01,
                1.064556e+00,
                2.793013e+02,
                1.065411e+00,
                1.073102e+04,
                1.645586e+02,
                1.076047e+04,
                9.961593e+03
        ]
        # check some basic properties / values of the geometry
        npt.assert_allclose(actual=actual,desired=desired,rtol=1e-5)
        print(f"All tests passed in script: {pathlib.Path(__file__).name}")
    try:
        if int(options.show) == 1:
            plt.show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()