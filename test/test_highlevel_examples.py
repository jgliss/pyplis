# -*- coding: utf-8 -*-
"""Pyplis high level test module.

This module contains some highlevel tests with the purpose to ensure
basic functionality of the most important features for emission-rate
analyses.

Note
----
The module is based on the dataset "testdata_minimal" which can be found
in the GitHub repo in the folder "pyplis/data/". The dataset is based on
the official Pyplis testdata set which is used for the example scripts.
This minimal version does not contain all images and the images are reduced
in size (Gauss-pyramid level 4).

Author: Jonas Gliss
Email: jonasgliss@gmail.com
License: GPLv3+
"""
import pyplis
from os.path import join
from datetime import datetime
import numpy.testing as npt
import pytest

BASE_DIR = join(pyplis.__dir__, "data", "testdata_minimal")
IMG_DIR = join(BASE_DIR, "images")

START_PLUME = datetime(2015, 9, 16, 7, 10, 00)
STOP_PLUME = datetime(2015, 9, 16, 7, 20, 00)

START_CALIB = datetime(2015, 9, 16, 6, 59, 00)
STOP_CALIB = datetime(2015, 9, 16, 7, 3, 00)

CALIB_CELLS = {'a37': [8.59e17, 2.00e17],
               'a53': [4.15e17, 1.00e17],
               'a57': [19.24e17, 3.00e17]}

PLUME_FILE = join(IMG_DIR,
                  'EC2_1106307_1R02_2015091607110434_F01_Etna.fts')
PLUME_FILE_NEXT = join(IMG_DIR,
                       'EC2_1106307_1R02_2015091607113241_F01_Etna.fts')
BG_FILE_ON = join(IMG_DIR, 'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')
BG_FILE_OFF = join(IMG_DIR, 'EC2_1106307_1R02_2015091607022216_F02_Etna.fts')

FUN = pyplis.custom_image_import.load_ecII_fits


@pytest.fixture(scope="function")
def plume_img():
    return pyplis.Img(PLUME_FILE, FUN).pyr_up(1)


@pytest.fixture(scope="function")
def plume_img_next():
    return pyplis.Img(PLUME_FILE_NEXT, FUN).pyr_up(1)


@pytest.fixture(scope="function")
def bg_img_on():
    return pyplis.Img(BG_FILE_ON, FUN).to_pyrlevel(0)


@pytest.fixture(scope="function")
def bg_img_off():
    return pyplis.Img(BG_FILE_OFF, FUN).to_pyrlevel(0)


def _make_setup():
    cam_id = "ecII"

    # Define camera (here the default ecII type is used)
    img_dir = IMG_DIR

    # Load default information for Etna
    source = pyplis.Source("etna")

    # Provide wind direction
    wind_info = {"dir": 0.0,
                 "dir_err": 1.0}

    # camera location and viewing direction (altitude will be retrieved
    # automatically)
    geom_cam = {"lon"           :   15.1129,   # noqa: E241 E203
                "lat"           :   37.73122,  # noqa: E241 E203
                'altitude'      :   800,       # noqa: E241 E203
                "elev"          :   20.0,      # noqa: E241 E203
                "elev_err"      :   5.0,       # noqa: E241 E203
                "azim"          :   270.0,     # noqa: E241 E203
                "azim_err"      :   10.0,      # noqa: E241 E203
                "alt_offset"    :   15.0,      # noqa: E241 E203
                "focal_length"  :   25e-3}     # noqa: E241 E203

    # the camera filter setup
    filters = [pyplis.Filter(type="on", acronym="F01"),
               pyplis.Filter(type="off", acronym="F02")]

    cam = pyplis.Camera(cam_id, filter_list=filters, **geom_cam)

    return pyplis.MeasSetup(img_dir,
                            camera=cam,
                            source=source,
                            wind_info=wind_info,
                            cell_info_dict=CALIB_CELLS,
                            auto_topo_access=False)


@pytest.fixture(scope="function")
def setup():
    return _make_setup()


@pytest.fixture(scope="function")
def calib_dataset(setup):
    """Initialize calibration dataset."""
    setup.start = START_CALIB
    setup.stop = STOP_CALIB

    return pyplis.CellCalibEngine(setup)


@pytest.fixture(scope="function")
def plume_dataset(setup):
    """Initialize measurement setup and create dataset from that."""
    setup.start = START_PLUME
    setup.stop = STOP_PLUME
    # Create analysis object (from BaseSetup)
    return pyplis.Dataset(setup)


@pytest.fixture(scope="function")
def aa_image_list(plume_dataset, bg_img_on, bg_img_off, viewing_direction):
    """Prepare AA image list for further analysis."""
    # Get on and off lists and activate dark correction
    lst = plume_dataset.get_list("on")
    lst.activate_darkcorr()  # same as lst.darkcorr_mode = 1

    off_list = plume_dataset.get_list("off")
    off_list.activate_darkcorr()

    # Prepare on and offband background images
    bg_img_on.subtract_dark_image(lst.get_dark_image().to_pyrlevel(0))
    bg_img_off.subtract_dark_image(off_list.get_dark_image().to_pyrlevel(0))

    # set the background images within the lists
    lst.set_bg_img(bg_img_on)
    off_list.set_bg_img(bg_img_off)

    # automatically set gas free areas
    # NOTE: this corresponds to pyramid level 3 as the test data is
    # stored in low res
    lst.bg_model.set_missing_ref_areas(lst.current_img())

    # Now update some of the information from the automatically set sky ref
    # areas
    lst.bg_model.xgrad_line_startcol = 1
    lst.bg_model.xgrad_line_rownum = 2

    off_list.bg_model.update(**lst.bg_model.settings_dict())

    lst.bg_model.mode = 0
    off_list.bg_model.mode = 0

    lst.calc_sky_background_mask()
    lst.aa_mode = True  # activate AA mode
# =============================================================================
#
#         m = lst.bg_model
#         ax = lst.show_current()
#         ax.set_title("MODE: %d" %m.mode)
#         m.plot_tau_result()
#
# =============================================================================
# =============================================================================
#         for mode in range(1,7):
#             lst.bg_model.CORR_MODE = mode
#             off_list.bg_model.CORR_MODE = mode
#             lst.load()
#             ax = lst.show_current()
#             ax.set_title("MODE: %d" %m.mode)
#             m.plot_tau_result()
# =============================================================================
    lst.meas_geometry = viewing_direction
    return lst


@pytest.fixture(scope="function")
def line():
    """Create an example retrieval line."""
    return pyplis.LineOnImage(630, 780, 1000, 350, pyrlevel_def=0,
                              normal_orientation="left")


@pytest.fixture(scope="function")
def geometry(plume_dataset):
    return plume_dataset.meas_geometry


@pytest.fixture(scope="function")
def viewing_direction(geometry):
    """Find viewing direction of camera based on MeasGeometry."""
    from geonum import GeoPoint
    # Position of SE crater in the image (x, y)
    se_crater_img_pos = [720, 570]  # [806, 736] (changed on 12/5/19)
    # Geographic position of SE crater (extracted from Google Earth)
    # The GeoPoint object (geonum library) automatically retrieves the altitude
    # using SRTM data
    se_crater = GeoPoint(37.747757, 15.002643, altitude=3103.0)

    # The following method finds the camera viewing direction based on the
    # position of the south east crater.
    new_elev, new_azim, _, basemap =\
        geometry.find_viewing_direction(
            pix_x=se_crater_img_pos[0],
            pix_y=se_crater_img_pos[1],
            pix_pos_err=100,  # for uncertainty estimate
            geo_point=se_crater,
            draw_result=False,
            update=True)  # overwrite old settings
    return geometry


def test_setup(setup):
    """Test some properties of the MeasSetup object."""
    s = setup.source
    vals_exact = [setup.save_dir == setup.base_dir,
                  setup.camera.cam_id]
    vals_approx = [sum([sum(x) for x in setup.cell_info_dict.values()]),
                   s.lon + s.lat + s.altitude]

    nominal_exact = [True, "ecII"]
    nominal_approx = [3.798e18, 3381.750]

    npt.assert_array_equal(vals_exact, nominal_exact)
    npt.assert_allclose(vals_approx, nominal_approx, rtol=1e-4)


def test_dataset(plume_dataset):
    """Test certain properties of the dataset object."""
    ds = plume_dataset
    keys = list(ds.img_lists_with_data.keys())
    vals_exact = [ds.img_lists["on"].nof + ds.img_lists["off"].nof,
                  sum(ds.current_image("on").shape),
                  keys[0], keys[1], ds.cam_id]

    nominal_exact = [178, 2368, "on", "off", "ecII"]

    npt.assert_array_equal(vals_exact, nominal_exact)


def test_find_viewdir(viewing_direction):
    """Correct viewing direction using location of Etna SE crater."""
    vals = [viewing_direction.cam_azim, viewing_direction.cam_azim_err,
            viewing_direction.cam_elev, viewing_direction.cam_elev_err]
    npt.assert_allclose(actual=vals,
                        desired=[280.21752138146036, 1.0656706289128692,
                                 13.72632050624192, 1.0656684171601736],
                        rtol=1e-7)


def test_imglists(plume_dataset):
    """Test some properties of the on and offband image lists."""
    on = plume_dataset._lists_intern["F01"]["F01"]
    off = plume_dataset._lists_intern["F02"]["F02"]

    vals_exact = [on.list_id, off.list_id]

    nominal_exact = ["on", "off"]

    npt.assert_array_equal(vals_exact, nominal_exact)


def test_line(line):
    """Test some features from example retrieval line."""
    n1, n2 = line.normal_vector
    l1 = line.convert(1, [100, 100, 1200, 1024])

    # compute values to be tested
    vals = [line.length(), line.normal_theta, n1, n2,
            l1.length() / line.length(), sum(l1.roi_def)]
    # set nominal values
    nominal = [567, 310.710846671181, -0.7580108737829234, -0.6522419146504225,
               0.5008818342151675, 1212]

    npt.assert_allclose(vals, nominal, rtol=1e-7)


def test_geometry(geometry):
    """Test important results from geometrical calculations."""
    res = geometry.compute_all_integration_step_lengths()
    vals = [res[0].mean(), res[1].mean(), res[2].mean()]
    npt.assert_allclose(actual=vals,
                        desired=[2.0292366, 2.0292366, 10909.873],
                        rtol=1e-7)


def test_optflow(plume_img, plume_img_next, line):
    """Test optical flow calculation."""
    flow = pyplis.OptflowFarneback()
    flow.set_images(plume_img, plume_img_next)
    flow.calc_flow()
    len_img = flow.get_flow_vector_length_img()
    angle_img = flow.get_flow_orientation_img()
    l = line.convert(plume_img.pyrlevel)
    res = flow.local_flow_params(line=l, dir_multi_gauss=False)
    nominal = [0.658797,
               -41.952854,
               -65.971787,
               22.437565,
               0.128414,
               0.086898,
               28.07,
               0.518644]
    vals = [len_img.mean(),
            angle_img.mean(), res["_dir_mu"],
            res["_dir_sigma"], res["_len_mu_norm"],
            res["_len_sigma_norm"], res["_del_t"],
            res["_significance"]]
    npt.assert_allclose(vals, nominal, rtol=1e-5)


def test_auto_cellcalib(calib_dataset):
    """Test if automatic cell calibration works."""
    calib_dataset.find_and_assign_cells_all_filter_lists()
    keys = ["on", "off"]
    nominal = [6., 845.50291, 354.502678, 3., 3.]
    mean = 0
    bg_mean = calib_dataset.bg_lists["on"].current_img().mean() +\
        calib_dataset.bg_lists["off"].current_img().mean()
    num = 0
    for key in keys:
        for lst in calib_dataset.cell_lists[key].values():
            mean += lst.current_img().mean()
            num += 1
    vals = [num, mean, bg_mean, len(calib_dataset.cell_lists["on"]),
            len(calib_dataset.cell_lists["off"])]
    npt.assert_allclose(nominal, vals, rtol=1e-7)


def test_bg_model(plume_dataset):
    """Test properties of plume background modelling.

    Uses the PlumeBackgroundModel instance in the on-band image
    list of the test dataset object (see :func:`plume_dataset`)
    """
    l = plume_dataset.get_list("on")
    m = l.bg_model

    m.set_missing_ref_areas(l.current_img())