from os.path import join, exists
from matplotlib.pyplot import close, show, rc_context

import pyplis
# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER, LINES

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

rc_context({'font.size': '18'})

# Raises Exception if conflict occurs
PCS = LINES[0]

# SCRIPT OPTONS
PYRLEVEL = 1
PLUME_VELO_GLOB = 4.29  # m/s
PLUME_VELO_GLOB_ERR = 1.5
# applies multi gauss fit to retrieve local predominant displacement
# direction, if False, then the latter is calculated from 1. and 2. moment
# of histogram (Faster but more sensitive to additional peaks in histogram)
HISTO_ANALYSIS_MULTIGAUSS = True
# molar mass of SO2
MMOL = 64.0638  # g/mol
# minimum required SO2-CD for emission-rate retrieval
CD_MIN = 5e16

START_INDEX = 0
STOP_INDEX = None
DO_EVAL = True

# activate background check mode, if True, emission rates are only retrieved
# for images showing SO2-CDs within specified interval around zero in BG
# reference rectangle LOG_ROI_SKY (see above). This can be used to ensure that
# significant systematic errors are induced in case the plume background
# retrieval failed. The latter could, for instance happen, if, for instance a
# cloud moves through one of the background reference areas used to model the
# background (cf. example script 3)
REF_CHECK_LOWER = -5e16
REF_CHECK_UPPER = 5e16
REF_CHECK_MODE = True

# the following ROI is in the upper right image corner, where no gas occurs in
# the time series. It is used to log mean, min and max for each analysed image
# this information can be used to check, whether the plume background retrieval
# worked well
LOG_ROI_SKY = [530, 30, 600, 100]  # correspond to pyrlevel 1

# RELEVANT DIRECTORIES AND PATHS

# DOAS calibration results from example script 6
CALIB_FILE = join(SAVE_DIR, "ex06_doascalib_aa.fts")

# AA sensitivity correction mask retrieved from cell calib in script 7
CORR_MASK_FILE = join(SAVE_DIR, "ex07_aa_corr_mask.fts")

# time series of predominant displacement vector from histogram analysis of
# optical flow field in ROI around the PCS line "young_plume" which is used
# here for the emission rate retrieval. These information is optional, and is
# calculated during the evaluation if not provided
RESULT_PLUMEPROPS_HISTO = join(SAVE_DIR, "ex09_plumeprops_young_plume.txt")

# SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    figs = []
    if not exists(CALIB_FILE):
        raise IOError("Calibration file could not be found at specified "
                      "location:\n%s\nPlease run example 6 first")
    if not exists(CORR_MASK_FILE):
        raise IOError("Cannot find AA correction mask, please run example"
                      "script 7 first")

    # convert the retrieval line to the specified pyramid level (script option)
    pcs = PCS.convert(to_pyrlevel=PYRLEVEL)

    # Load AA list
    # includes viewing direction corrected geometry
    aa_list = prepare_aa_image_list()
    aa_list.pyrlevel = PYRLEVEL

    # Load DOAS calbration data and FOV information (see example 6)
    doascalib = pyplis.doascalib.DoasCalibData()
    doascalib.load_from_fits(file_path=CALIB_FILE)
    doascalib.fit_calib_polynomial()

    # Load AA corr mask and set in image list(is normalised to DOAS FOV see
    # ex7)
    aa_corr_mask = pyplis.Img(CORR_MASK_FILE)
    aa_list.aa_corr_mask = aa_corr_mask

    # set DOAS calibration data in image list
    aa_list.calib_data = doascalib
    # you can check the settings first
    aa_list.gaussian_blurring = 1
    aa_list.calib_mode = False
    aa_list.aa_mode = False
    raw_disp = aa_list.show_current()
    raw = aa_list.current_img().duplicate()
    raw_disp.set_title("")
    figs.append(raw_disp.figure)
    aa_list.calib_mode = True

    from cv2 import erode, dilate
    import numpy as np
    calib = aa_list.current_img()
    mask = calib.img < -5e16
    mask = mask.astype(np.float32)
    size = 20

    kernel = np.ones((size, size))
    mask = erode(mask, kernel)
    mask = dilate(mask, kernel)
    pyplis.Img(mask).show()

    calib = aa_list.current_img().duplicate()
    calib.img[mask.astype(bool)] = 0
    c_disp = calib.show(vmin=-1e18, vmax=2e18, zlabel=r"$S_{SO2}\,[cm^{-2}]$",
                        zlabel_size=24)
    c_disp.set_title("")

    aa_list.calib_mode = 0
    aa_list.tau_mode = 1
    aa_list.optflow_mode = 1
    aa_list.optflow.settings.disp_skip = 20
    aa_list.optflow.settings.disp_len_thresh = 3
    aa_list.optflow.plot(extend_len_fac=2.0, color="#ff8c1a", ax=c_disp)
    pcs.color = "#00e600"
    pcs.x0 = 325
    pcs.y0 = 380
    pcs.plot_line_on_grid(ax=c_disp, annotate_normal=True,
                          include_normal=True)

    figs.append(c_disp.figure)

    if SAVEFIGS:
        for k in range(len(figs)):
            figs[k].savefig(join(SAVE_DIR, "priv_01_out_%d.%s"
                            % (k + 1, FORMAT)),
                            format=FORMAT, dpi=DPI)

    # Display images or not
    options = ARGPARSER.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")
