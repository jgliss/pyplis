Data import - Specifying custom camera information
==================================================

In order to use all features of pyplis, certain specifications related to camera
and image acquisition need to be defined. Basic information about the camera
(e.g. detector specifics) and the corresponding file convention (image type,
which data can be extracted from file names) are specified within
:class:`Camera` objects. This tutorial introduces the :class:`Camera` class and
how to set up your custom camera type based on your data format, including
definitions of your file naming convention.

Specifying your file naming convention
--------------------------------------

At the very beginning of the analysis, the images need to be imported and
separated by image type (e.g. on-band plume, off-band plume, dark, offset, on /
off-band cell calibration). In order to use the automated separation for a given
dataset (e.g. a single folder ``IMG_DIR`` containing images of all types) it is
required that the image type can be identified from the file names.

The relevant information for identifying different image types (e.g. plume
on-band, dark, offset) can be specified using either of the following two
classes:

  1. :class:`Filter`: specifies file access information for all image types that
are NOT dark or offset images (e.g. on / off images plume / background)
  #. :class:`DarkOffsetInfo`: specifies different types of dark images and
offset images.

Such a collection of :class:`Filter` and :class:`DarkOffsetInfo` objects is then
stored within a :class:`Camera` object.

These information is used to separate the individual image types when creating a
:class:`Dataset` object. The latter searches all valid image files in a given
folder ``IMG_DIR`` and creates :class:`ImgList` objects for each :class:`Filter`
and :class:`DarkImgList` objects for each :class:`DarkOffsetInfo` object defined
in the :class:`Camera`. Each of these lists is then filled with the file paths
of the corresponding image type located in ``IMG_DIR``. The :class:`Camera`
object furthermore includes relevant specs of the camera (e.g. pixel geometry,
lens).

The following list provides an overview of relevant parameters for filename
access information using exemplary filenames of the ECII camera type.

.. _tut_ecIIcam:

The ECII camera standard
------------------------

In the following, an exemplary :class:`Camera` class is specified based on the
ECII-camera standard and file naming convention (cf. :ref:`ex02`).

To start with, an empty :class:`Camera` instance is created::

  cam = pyplis.Camera()
  # prints the string representation which gives a nice overview over the
  # relevant parameters
  print cam

If you wish to store the camera as default you need to specify a unique camera
ID (string) which is not yet used for any of the pyplis default cameras stored
in the file *cam_info.txt* (package data). You can check all existing IDs
using::

  print pyplis.inout.get_all_valid_cam_ids()

Let's call our new camera "ecII_test"::

  cam.cam_id = "ecII_test"

Now specify some relevant attributes of the camera, starting with the image file
type::

    cam.file_type = "fts"

You can also provide information about detector and camera lens::

    cam.focal_length = 25e-3 #m

    # Detector geometry
    cam.pix_height = 4.65e-6 # pixel height in m
    cam.pix_width = 4.65e-6 # pixel width in m
    cam.pixnum_x = 1344
    cam.pixnum_y = 1024

In the following, the camera file naming convention is specified. This enables
to extract certain information from the image file names (e.g. image acq. time,
image type, exposure time).

Start with setting the file name delimiter of your file naming convention::

    cam.delim = "_"

Based on that, specify the position of acquisition time (and date) in the image
file names after splitting with delimiter::

    cam.time_info_pos = 3

The acq. time strings in the file names need to be converted into ``datetime``
objects thus, specify the string for internal conversion (is done using
:func:`datetime.strptime`)::

    cam.time_info_str = "%Y%m%d%H%M%S%f"

If the file name also includes the image exposure time, this can also be
specified::

    cam.texp_pos = "" #the ECII does not...

as well as the unit (choose from "s" or "ms" if applicable)::

    cam.texp_unit = ""

Furthermore, the image type identification can (and should) be specified in the
camera, in order to make life easier. This ensures, that images of different
types (e.g. on / off-band, dark, offset) can be identified and separated
directly from the filename. The relevant information is specified in a
collection of :class:`Filter` and :class:`DarkOffsetInfo` objects.
Let's start off with defining the different image access types for on and
off-band images (these are stored in :class:`Filter` objects, while dark /
offset image access information is stored in :class:`DarkOffsetInfo` objects,
follows below)::

  # On-band images
  on = pyplis.Filter(id="on", type="on", acronym="F01",
                     meas_type_acro="F01", center_wavelength=310)
  # Off-band images
  off = pyplis.Filter(type="off", acronym="F02",
                      meas_type_acro="F02", center_wavelength=330)

Now add the two filters to the camera (i.e. put them into a list and assign  it
to the camera)::

    filters = [on, off]

    cam.default_filters = filters

    # Checks and sets filters in cam
    cam.prepare_filter_setup()

Tell the camera, which of the filters is the "central" filter for the emission
rate analysis (usually "on")::

    cam.main_filter_id = "on"

The latter information is used for internal linking of image lists within a
:class:`Dataset` object, for instance, if the camera contains multiple
``type="on"`` filters (i.e. on-band SO2).

.. note::

  This parameter ``main_filter_id`` is irrelevant for standard setups like here
(i.e. one on and one off-band filter)

Similar to the filter setup (which specifies access to the actual images to be
analysed), the filename access information for dark (``type=dark``) and offset
(``type=offset``) image identification needs to be specified using
:class:`DarkOffsetInfo` instances::

    offset_low_gain  = pyplis.DarkOffsetInfo(id="offset0",type="offset",
                                            acronym="D0L",
                                            meas_type_acro="D0L",
                                            read_gain=0)

    offset_high_gain = pyplis.DarkOffsetInfo(id="offset1",type="offset",
                                             acronym="D0H", read_gain=1)
    dark_low_gain    = pyplis.DarkOffsetInfo(id="dark0",type="dark",
                                             acronym="D1L", read_gain=0)
    dark_high_gain   = pyplis.DarkOffsetInfo(id="dark1",type="dark",
                                             acronym="D1H", read_gain=1)

    # put the 4 dark info objects into a list and assign to the camera
    dark_info = [offset_low_gain, offset_high_gain,
                 dark_low_gain, dark_high_gain]

    cam.dark_info = dark_info

.. note::

  You might have recognised, that in the last 3 :class:`DarkOffsetInfo``
objects, the meas_type_acro was not specified. This is because it is actually
irrelevant for the ECII camera which does not include a sub string specifying
different measurement modi like, for instance, the HD-Custom camera (i.e. K, M,
D).

Now that all different image types are specified, the camera needs to know where
to find the actual information in the file names (after splitting using
``delim``).
The position of the strings specified in the attribute ``acronym`` (see
definitions of the ``Filter`` and ``DarkOffsetInfo`` objects above) can be set
using::

    cam.acronym_pos = 4

and the position of the strings specified in attribute ``meas_type_acro``::

    cam.meas_type_acro_pos = 4

.. note::

  If ``meas_type_acro`` is irrelevant (like for this camera) it is required to
be set equal ``acronym_pos``

Furthermore, the dark correction type needs to be specified, pyplis includes two
options for that, the ECII uses option 1::

    cam.DARK_CORR_OPT = 1

.. todo::

  Include information about the two different dark correction modes

That's it! You might want to check if everything is in place::

  print cam

If you are happy, you might want to check if the data access from the file names
works. You can do a fast check using a file path ``IMG_PATH`` to one of your
images and run::

  acq_time, filter_id, meas_type, texp, warnings =\
                    cam.get_img_meta_from_filename(IMG_PATH)

You might also test it for a whole dataset of images located in a directory
``IMG_DIR`` and check if pyplis can identify the different image types. You can
do this, for instance, by creating a :class:`Dataset` object. First, create a
measurement setup with minimum information::

  meas_setup = pyplis.MeasSetup(base_dir=IMG_DIR, camera=cam)

and create a Dataset from that::

  ds = pyplis.Dataset(meas_setup)

The :class:`Dataset` object should now detect all individual image types and
puts them into separate lists, which can be accessed using the IDs of the
corresponding :class:`Filter` objects, e.g.::

  lst = ds.get_list("on")
  print "Number of images in list: %d" %lst.nof

These lists are of type ``ImgList``. Similarly, dark and offset image lists
(:class:`DarkImgList` classes) were created using the information stored in the
:class:`DarkOffsetInfo` objects specified in our camera::

  dark_list_low_gain = ds.get_list("dark0")
  offset_list_low_gain = ds.get_list("offset0")

You can also easily access all lists, that actually contain images (i.e. for
which image matches could be found in ``IMG_DIR``), e.g. all lists that contain
images and correspond to one of the ``Filter`` objects::

  all_imglists = ds.img_lists_with_data #this is a dictionary
  print all_imglists.keys() #prints the list / Filter IDs

and similar, all :class:`DarkImgList` objects that contain data::

  all_darklists = ds.dark_lists_with_data #this is a dictionary
  print all_darklists.keys() #prints the list IDs

If everything works out nicely, you can add the camera as new default using::

  cam.save_as_default()

After saving the camera as new default, you can load it using::

  import pyplis
  cam = pyplis.Camera(cam_id="ecII_test")
  print cam

Done!
