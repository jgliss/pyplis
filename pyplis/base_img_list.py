from pyplis import logger, print_log
from pyplis.geometry import MeasGeometry
from pyplis.glob import DEFAULT_ROI
from pyplis.helpers import check_roi, exponent, get_pyr_factor_rel, isnum, map_roi
from pyplis.image import Img
from pyplis.processing import ImgStack, PixelMeanTimeSeries
from pyplis.setupclasses import Camera
from pyplis.utils import LineOnImage

from matplotlib.pyplot import draw, figure
from numpy import arange, argmin, asarray, float32, ndarray, zeros
from numpy.ma import nomask
from scipy.ndimage.filters import gaussian_filter

from copy import deepcopy
from datetime import date, datetime
from os.path import basename, exists
from traceback import format_exc

class BaseImgList:
    """Basic image list object.

    Basic class for image list objects providing indexing and image loading
    functionality

    In this class, only the current image is loaded at a time while
    :class:`ImgList` loads current and next image whenever the index is
    changed (e.g. required for :attr:`optflow_mode`)

    This object and all objects inheriting from this are fundamentally based
    on a list of image file paths, which are dynamically loaded and processed
    during usage.

    Parameters
    ----------
    files : list, optional
        list with image file paths
    list_id : str, optional
        a string used to identify this list (e.g. "second_onband")
    list_type : str, optional
        type of images in list (please use "on" or "off")
    camera : Camera, optional
        camera specifications
    init : bool
        if True, list will be initiated and files loaded (given that image
        files are provided on input)
    **img_prep_settings
        additional keyword args specifying image preparation settings applied
        on image load
    """

    def __init__(self, files=None, list_id=None, list_type=None,
                 camera=None, geometry=None, init=True, **img_prep_settings):
        # this list will be filled with filepaths
        self.files = []
        # id of this list
        self.list_id = list_id
        self.list_type = list_type

        self.filter = None  # can be used to store filter information
        self._meas_geometry = None

        # these variables can be accessed using corresponding @property
        # attributes
        self._integration_step_lengths = None
        self._plume_dists = None

        self.set_camera(camera=camera, cam_id=None)

        self._update_cam_geodata = False
        self._edit_active = True

        # the following dictionary contains settings for image preparation
        # applied on image load
        self.img_prep = {"blurring": 0,  # width of gauss filter
                         "median": 0,  # width of median filter
                         "crop": False,
                         "pyrlevel": 0,  # int, gauss pyramide level
                         "8bit": 0}  # to 8bit

        self._roi_abs = DEFAULT_ROI  # in original img resolution
        self._auto_reload = True

        self._list_modes = {}  # init for :class:`ImgList` object

        self._vign_mask = None  # vignetting correction mask can be stored here
        self.__sky_mask = nomask  # mask for invalid pixel
        self.loaded_images = {"this": None}

        # used to store the img edit state on load
        self._load_edit = {"this": {},
                           "next": {}}

        self.index = 0
        self._skip_files = 0  # if 0, no files are skipped
        self.next_index = 0
        self.prev_index = 0

        # Other image lists can be linked to this and are automatically updated
        self.linked_lists = {}
        # this dict (linked_indices) is filled in :func:`link_imglist` to
        # increase the linked reload image performance
        self._linked_indices = {}
        # contains info about the always_reload option of linked image lists
        # is updated whenever a new list is linked to this one
        self._always_reload = {}

        # update image preparation settings (if applicable)
        for key, val in img_prep_settings.items():
            if key in self.img_prep:
                self.img_prep[key] = val

        if bool(files):
            self.add_files(files, load=False)

        if isinstance(geometry, MeasGeometry):
            self.meas_geometry = geometry

        if self.data_available and init:
            self.load()

    @property
    def start(self):
        """Acquisistion time of first image."""
        try:
            return self.start_acq[0]
        except IndexError:
            raise IndexError("No data available")

    @property
    def stop(self):
        """Start acqusition time of last image."""
        try:
            return self.start_acq[-1]
        except IndexError:
            raise IndexError("No data available")

    @property
    def this(self):
        """Return current image."""
        return self.current_img()

    @property
    def edit_active(self):
        """Define whether images are edited on image load or not.

        If False, images will be loaded as raw, i.e. without any editing or
        further calculations (e.g. determination of optical flow, or updates of
        linked image lists). Images will be reloaded.
        """
        return self._edit_active

    @edit_active.setter
    def edit_active(self, value):
        if value == self._edit_active:
            return
        self._edit_active = value
        self.load()

    @property
    def skip_files(self):
        """Integer specifying the image iter step in the file list.

        Defaults to 1: every file is used, 2 means, that every second file is
        used.
        """
        return self._skip_files

    @skip_files.setter
    def skip_files(self, val):
        if not val >= 0:
            raise ValueError("Value must be 0 or positive")
        self._skip_files = int(val)
        self.iter_indices(self.index)
        self.load()

    @property
    def meas_geometry(self):
        """Return measurement geometry."""
        return self._meas_geometry

    @meas_geometry.setter
    def meas_geometry(self, val):
        if not isinstance(val, MeasGeometry):
            raise TypeError("Could not set meas_geometry, need MeasGeometry "
                            "object")
        self._meas_geometry = val

    @property
    def update_cam_geodata(self):
        """Update measurement geometry whenever list index is changed."""
        return self._update_cam_geodata

    @update_cam_geodata.setter
    def update_cam_geodata(self, value):
        try:
            self._update_cam_geodata = bool(value)
        except BaseException:
            raise

    @property
    def plume_dists(self):
        """Distance to plume.

        Can be an image were each pixel value corresponds to the plume distance
        at each pixel position (e.g. computed using the MeasGeometry) or can
        also be a single value, which may be appropriate under certain
        measurement setups (e.g. distant plume perpendicular to CFOV of camera)

        Note
        ----
        This method checks if a value is accessible in :attr:`_plume_dists` and
        if not tries to compute plume distances by calling
        :func:`compute_all_integration_step_lengths` of the
        :class:`MeasGeometry` object assigned to this ImgList. If this fails,
        then an AttributeError is raised

        Returns
        -------
        float or Img or ndarray
            Plume distances in m. If plume distances are accessible per image
            pixel. Note that the corresponding data is converted to pyramid
            level 0 (required for dilution correction).

        """
        v = self._plume_dists
        if isnum(v):
            return v
        elif isinstance(v, Img):
            return v.to_pyrlevel(0)
        self._get_and_set_geometry_info()
        return self._plume_dists

    @plume_dists.setter
    def plume_dists(self, value):
        if not (isnum(value) or isinstance(value, Img)):
            raise TypeError(
                "Need Img or numerical data type (e.g. float, int)")
        if isinstance(value, Img):
            value = value
        self._plume_dists = value

    @property
    def vign_mask(self):
        """Return current vignetting correction mask."""
        if not any([isinstance(self._vign_mask, x) for x in (Img, ndarray)]):
            raise AttributeError("Vignetting mask is not available in list")
        return self._vign_mask

    @vign_mask.setter
    def vign_mask(self, value):
        if not any([isinstance(value, x) for x in (Img, ndarray)]):
            raise AttributeError("Invalid input for vignetting mask, need "
                                 "Img object or numpy ndarray")
        try:
            value = Img(value)
        except BaseException:
            pass
        pyrlevel_rel = get_pyr_factor_rel(self.current_img().img, value.img)
        if pyrlevel_rel != 0:
            if pyrlevel_rel < 0:
                value.pyr_down(pyrlevel_rel)
            else:
                value.pyr_up(pyrlevel_rel)
        self._vign_mask = value

    @property
    def sky_mask(self):
        """Return sky access mask.

        0 for sky,
        1 for non-sky (=invalid)
        (in masked arrays, entries marked with 1 are invalid)
        """
        return self.__sky_mask

    @sky_mask.setter
    def sky_mask(self, value):
        # TODO: Check if the mask has the same dimension as the images
        # TODO: maybe load as pyplis img
        if not isinstance(value, ndarray):
            raise TypeError("Could not set sky_mask, need MeasGeometry "
                            "object")
        self.__sky_mask = deepcopy(value)

    @property
    def integration_step_length(self):
        """Return integration step length for emission-rate analyses.

        The intgration step length corresponds to the physical distance in
        m between two pixels within the plume and is central for computing
        emission-rate. It may be an image were each pixel value corresponds to
        the integreation step length at each pixel position (e.g. computed
        using the MeasGeometry) or it can also be a single value, which may be
        appropriate under certain measurement setups (e.g. distant plume
        perpendicular to CFOV of camera).

        Note
        ----
        This method checks if a value is accessible in
        :attr:`_integration_step_lengths` and if not tries to compute them by
        calling :func:`compute_all_integration_step_lengths` of the
        :class:`MeasGeometry` object assigned to this ImgList. If this fails,
        an AttributeError is raised

        Returns
        -------
        float or Img or ndarray
            Integration step lengths in m. If plume distances are accessible
            per image pixel, then the corresponding data IS converted to the
            current pyramid level

        """
        v = self._integration_step_lengths
        if isnum(v):
            return v
        elif isinstance(v, Img):
            return v.to_pyrlevel(self.pyrlevel)
        self._get_and_set_geometry_info()
        return self._integration_step_lengths

    @integration_step_length.setter
    def integration_step_length(self, value):
        if not (isnum(value) or isinstance(value, Img)):
            raise TypeError(
                "Need Img or numerical data type (e.g. float, int)")
        if isinstance(value, Img):
            value = value.to_pyrlevel(self.pyrlevel)
            if not value.shape == self.current_img().shape:
                raise ValueError("Cannot set plume distance image: shape "
                                 "mismatch between input and images in list")
        self._integration_step_lengths = value

    @property
    def auto_reload(self):
        """Activate / deactivate automatic reload of images."""
        return self._auto_reload

    @auto_reload.setter
    def auto_reload(self, val):
        self._auto_reload = val
        if bool(val):
            logger.info("Reloading images...")
            self.load()

    @property
    def crop(self):
        """Activate / deactivate crop mode."""
        return self.img_prep["crop"]

    @crop.setter
    def crop(self, value):
        """Set crop."""
        self.img_prep["crop"] = bool(value)
        self.load()

    @property
    def pyrlevel(self):
        """Return current Gauss pyramid level.

        Note
        ----
        images are reloaded on change
        """
        return self.img_prep["pyrlevel"]

    @pyrlevel.setter
    def pyrlevel(self, value):
        logger.info("Updating pyrlevel and reloading")
        if value != self.pyrlevel:
            self.img_prep["pyrlevel"] = int(value)
            self.load()

    @property
    def gaussian_blurring(self):
        """Return current blurring level.

        Note
        ----
        images are reloaded on change
        """
        return self.img_prep["blurring"]

    @gaussian_blurring.setter
    def gaussian_blurring(self, val):
        if val < 0:
            raise ValueError("Negative smoothing kernel does not make sense..")
        elif val > 10:
            print_log.warning("Activate gaussian blurring with kernel size exceeding 10, "
                 "this might significantly slow down things..")
        self.img_prep["blurring"] = val
        self.load()

    @property
    def roi(self):
        """Return current ROI (in relative coordinates).

        The ROI is returned with respect to the current :attr:`pyrlevel`
        """
        return map_roi(self._roi_abs, self.pyrlevel)

    @roi.setter
    def roi(self):
        raise AttributeError("Please use roi_abs to set the current ROI in "
                             "absolute image coordinates. :func:`roi` is used "
                             "to access the current ROI for the actual "
                             "pyramide level.")

    @property
    def roi_abs(self):
        """Return current roi in absolute detector coords (cf. :attr:`roi`)."""
        # return map_roi(self._roi_abs, self.img_prep["pyrlevel"])
        return self._roi_abs

    @roi_abs.setter
    def roi_abs(self, val):
        if check_roi(val):
            self._roi_abs = val
            self.load()

    @property
    def cfn(self):
        """Return current index (file number in ``files``)."""
        return self.index

    @property
    def nof(self):
        """Return number of files in this list."""
        return len(self.files)

    @property
    def last_index(self):
        """Return index of last image."""
        return len(self.files) - 1

    @property
    def data_available(self):
        """Return wrapper for :func:`has_files`."""
        return self.has_files()

    @property
    def has_images(self):
        """Return wrapper for :func:`has_files`."""
        return self.has_files()

    @property
    def start_acq(self):
        """Array containing all image acq. time stamps of this list.

        Note
        ----
        The time stamps are extracted from the file names
        """
        ts = self.get_img_meta_all_filenames()[0]
        return ts

    def timestamp_to_index(self, val=datetime(1900, 1, 1)):
        """Convert a datetime to the list index.

        Returns the list index that is closest in time to the input time
        stamp.

        Parameters
        ----------
        val : datetime
            time stamp

        Raises
        ------
        AttributeError
            if time stamps of images in list cannot be accessed from their
            file names

        Returns
        -------
        int
            corresponding list index

        """
        times = self.start_acq
        if not len(times) == self.nof:
            raise AttributeError("Failed to access all acq. time stamps could "
                                 "not be accessed")
        return argmin(abs(val - times))

    def index_to_timestamp(self, val=0):
        """Get timestamp of input list index.

        Parameters
        ----------
        val : index
            time stamp

        Raises
        ------
        AttributeError
            if time stamps of images in list cannot be accessed from their
            file names

        Returns
        -------
        int
            corresponding list index

        """
        times = self.start_acq
        if not len(times) == self.nof:
            raise AttributeError("Acq. time stamps could not be accessed")
        if not 0 <= val <= self.last_index:
            raise IndexError("List index out of range")
        return times[val]

    def add_files(self, files, load=True):
        """Add images to this list.

        Parameters
        ----------
        file_list : list
            list with file paths

        Returns
        -------
        bool
            success / failed

        """
        if files is None:
            files = []
        elif isinstance(files, str):
            files = [files]
        if not isinstance(files, list):
            raise TypeError("Error: file paths could not be added to image "
                            "list, wrong input type %s" % type(files))

        self.files.extend(files)
        self.init_filelist(at_index=self.index)
        if load and self.data_available:
            logger.info("Added %d files in list %s, load %s" % (len(files),
                                                       self.list_id, load))
            self.load()

    def init_filelist(self, at_index=0):
        """Initialize the filelist.

        Sets current list index and resets loaded images

        Parameters
        ----------
        at_index : int
            desired image index, defaults to 0

        """
        self.iter_indices(to_index=at_index)
        for key in self.loaded_images:
            self.loaded_images[key] = None

        if self.nof > 0:
            logger.info("\nInit ImgList %s" % self.list_id)
            logger.info("-------------------------")
            logger.info("Number of files: " + str(self.nof))
            logger.info("-----------------------------------------")

    def iter_indices(self, to_index):
        """Change the current image indices for previous, this and next img.

        Note
        ----
        This method only updates the actual list indices and does not perform
        a reload.
        """
        try:
            self.index = to_index % self.nof
            self.next_index = (self.index + self.skip_files + 1) % self.nof
            self.prev_index = (self.index - self.skip_files - 1) % self.nof

        except:
            self.index, self.prev_index, self.next_index = 0, 0, 0

    def load(self):
        """Load current image.

        Try to load the current file ``self.files[self.cfn]`` and if remove the
        file from the list if the import fails

        Returns
        -------
        bool
            if True, image was loaded, if False not
        """
        if not self._auto_reload:
            logger.info(f"Automatic image reload deactivated in image list {self.list_id}")
            return False
        try:
            img = self._load_image(self.index)
            self._load_edit["this"].update(img.edit_log)
            self.loaded_images["this"] = img
            if img.vign_mask is not None:
                self.vign_mask = img.vign_mask

            if self.update_cam_geodata:
                self.meas_geometry.update_cam_specs(**self.current_img().meta)

            self._apply_edit("this")

        except IOError:
            print_log.warning("Invalid file encountered at list index %s, file will"
                 " be removed from list" % self.index)
            self.pop()
            if self.nof == 0:
                raise IndexError("No filepaths left in image list...")
            self.load()

        except IndexError:
            try:
                self.init_filelist()
                self.load()
            except BaseException:
                raise IndexError("Could not load image in list %s: file list "
                                 " is empty" % (self.list_id))

        return True

    def goto_next(self):
        """Goto next index in list."""
        if self.nof < 2:
            print_log.warning("Only one image available, no index change or "
                 "reload performed")
            return self.current_img()
        self.iter_indices(to_index=self.next_index)
        self.load()
        return self.current_img()

    def goto_prev(self):
        """Load previous image in list."""
        if self.nof < 2:
            print_log.warning("Only one image available, no index change or "
                 "reload performed")
            return self.current_img()
        self.iter_indices(to_index=self.prev_index)
        self.load()
        return self.current_img()

    def goto_img(self, to_index, reload_here=False):
        """Change the index of the list, load and prepare images at new index.

        Parameters
        ----------
        to_index : float
             new list index
        reload_here : bool
            applies only if :param:`to_index` is the current list index. If
            True, then the current images are reloaded, if False, nothing is
            done.

        """
        if not -1 < to_index < self.nof:
            raise IndexError("Invalid index %d. List contains only %d files"
                             % (to_index, self.nof))

        elif to_index == self.index:
            if reload_here:
                self.load()
            return self.current_img()
        elif to_index == self.next_index:
            self.goto_next()
        elif to_index == self.prev_index:
            self.goto_prev()
        else:
            self.iter_indices(to_index)
            self.load()

        return self.loaded_images["this"]

    def pop(self, idx=None):
        """Remove one file from this list."""
        raise NotImplementedError("pop method of ImgList is currently not "
                                  "available...")
        logger.warning("Removing image at index %n from image list")
        if idx is None:
            idx = self.index
        self.files.pop(idx)

    def has_files(self):
        """Return boolean whether or not images are available in list."""
        return bool(self.nof)

    def plume_dist_access(self):
        """Check if measurement geometry is available."""
        if not isinstance(self.meas_geometry, MeasGeometry):
            return False
        try:
            plume_dist_img = self.meas_geometry.\
                compute_all_integration_step_lengths()[2]
            logger.info("Plume distances available, dist_avg = %.2f"
                  % plume_dist_img.mean())
        except BaseException:
            return False

    def update_img_prep(self, **settings):
        """Update image preparation settings and reload.

        Parameters
        ----------
        **settings
            key word args specifying settings to be updated (see keys of
            ``img_prep`` dictionary)

        """
        for key, val in settings.items():
            if key in self.img_prep and\
                    isinstance(val, type(self.img_prep[key])):
                self.img_prep[key] = val
        try:
            self.load()
        except IndexError:
            pass

    def clear(self):
        """Empty this list (i.e. :attr:`files`)."""
        self.files = []

    def separate_by_substr_filename(self, sub_str, sub_str_pos, delim="_"):
        """Separate this list by filename specifications.

        The function checks all current filenames, and keeps those, which have
        a certain sub string at a certain position in the file name after
        splitting using a provided delimiter. All other files are added to a
        new image list which is returned.

        Parameters
        ----------
        sub_str : str
            string identification used to identify the image type which is
            supposed to be kept within this list object
        sub_str_pos : int
            position of sub string after filename was split (using input
            param delim)
        delim : str
            filename delimiter, defaults to "_"

        Returns
        -------
        tuple
            2-element tuple containing

            - :obj:`ImgList`, list contains images matching the requirement
            - :obj:`ImgList`, list containing all other images

        """
        match = []
        rest = []
        for p in self.files:
            spl = basename(p).split(".")[0].split(delim)
            if spl[sub_str_pos] == sub_str:
                match.append(p)
            else:
                rest.append(p)

        lst_match = self.__class__(match, list_id="match", camera=self.camera)
        lst_rest = self.__class__(rest, list_id="rest", camera=self.camera)
        return (lst_match, lst_rest)

    def set_camera(self, camera=None, cam_id=None):
        """Set the current camera.

        Two options:

            1. set :obj:`Camera` directly
            2. provide one of the default camera IDs (e.g. "ecII", "hdcam")

        Parameters
        ----------
        camera : Camera
            the camera used
        cam_id : str
            one of the default cameras (use
            :func:`pyplis.inout.get_all_valid_cam_ids` to get the default
            camera IDs)

        """
        if camera is not None:
            if not isinstance(camera, Camera):
                raise TypeError("Camera argument for image list was not "
                                "correctly initialised with an object of type "
                                "pyplis.Camera")

            self.camera = camera

        else:
            if cam_id is not None:
                self.camera = Camera(cam_id)

    def reset_img_prep(self):
        """Init image pre-edit settings."""
        self.img_prep = dict.fromkeys(self.img_prep, 0)
        self._roi_abs = DEFAULT_ROI
        if self.nof > 0:
            self.load()

    def get_img_meta_from_filename(self, file_path):
        """Load and prepare img meta input dict for Img object.

        Args:
            file_path: image file path

        Returns:
            dict: dictionary containing start acquisition time and exposure time
        """
        info = self.camera.get_img_meta_from_filename(file_path)
        return {"start_acq": info[0], "texp": info[3]}

    def get_img_meta_all_filenames(self):
        """Try to load acquisition and exposure times from filenames.

        Note
        ----
        Only works if relevant information is specified in ``self.camera`` and
        can be accessed from the file names

        Returns
        -------
        tuple
            2-element tuple containing

            - list, list containing all retrieved acq. time stamps
            - list, containing all retrieved exposure times

        """
        times, texps = asarray([None] * self.nof), asarray([None] * self.nof)

        for k in range(self.nof):
            try:
                info = self.camera.get_img_meta_from_filename(self.files[k])
                times[k] = info[0]
                texps[k] = info[3]
            except BaseException:
                pass
        try:
            if times[0].date() == date(1900, 1, 1):
                d = self.current_img().meta["start_acq"].date()
                print_log.warning("Warning accessing acq. time stamps from file names in "
                     "ImgList: date information could not be accessed, using "
                     "date of currently loaded image meta info: %s" % d)
                times = asarray([datetime(d.year, d.month, d.day, x.hour,
                                          x.minute, x.second, x.microsecond)
                                 for x in times])
        except BaseException:
            pass
        return times, texps

    def assign_indices_linked_list(self, lst: "BaseImgList") -> ndarray:
        """Create a look up table for fast indexing between image lists.

        This method links the image indices of another list 
        to the indices of this list based on the acquisition times of
        the images in both lists, that is, it ensures that for each
        image in this list, the index of the image in the input list 
        closest in time is known.

        Parameters
        ----------
        lst : BaseImgList
            image list supposed to be linked

        Returns
        -------
        array
            array containing linked indices
        """
        idx_array = zeros(self.nof, dtype=int)
        times, _ = self.get_img_meta_all_filenames()
        times_lst, _ = lst.get_img_meta_all_filenames()
        if (any([x is None for x in times]) or
              any([x is None for x in times_lst])):
            print_log.warning("Image acquisition times could not be accessed from file "
                 "names, assigning by indices")
            lst_idx = arange(lst.nof)
            for k in range(self.nof):
                idx_array[k] = abs(k - lst_idx).argmin()
        else:
            for k in range(self.nof):
                idx = abs(times[k] - times_lst).argmin()
                idx_array[k] = idx
        return idx_array

    def same_preedit_settings(self, settings_dict):
        """Compare input settings dictionary with self.img_prep.

        Parameters
        ----------
        **settings_dict
            keyword args specifying settings to be compared

        Returns
        -------
        bool
            False if not the same, True else

        """
        sd = self.img_prep
        for key, val in settings_dict.items():
            if key in sd:
                if not sd[key] == val:
                    return False
        return True

    def make_stack(self, stack_id=None, pyrlevel=None, roi_abs=None,
                   start_idx=0, stop_idx=None, ref_check_roi_abs=None,
                   ref_check_min_val=None, ref_check_max_val=None,
                   dtype=float32):
        """Stack all images in this list.

        The stacking is performed using the current image preparation
        settings (blurring, dark correction etc). Only stack ROI and pyrlevel
        can be set explicitely.

        Note
        ----
        In case of ``MemoryError`` try stacking less images (specifying
        start / stop index) or reduce the size setting a different Gauss
        pyramid level.

        Parameters
        ----------
        stack_id : :obj:`str`, optional
            identification string of the image stack
        pyrlevel : :obj:`int`, optional
            Gauss pyramid level of stack
        roi_abs : list
            build stack of images cropped in ROI
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`, optional
            index of last considered image (if None, the last image in this
            list is used). Note that the timestamp option only works if acq.
            times can be accessed from filenames for all files in the list
            (using method :func:`timestamp_to_index`)
        ref_check_roi_abs : :obj:`list`, optional
            rectangular area specifying a reference area which can be specified
            in combination with the following 2 parameters in order to include
            only images in the stack that are within a certain intensity range
            within this ROI (Note that this ROI needs to be specified in
            absolute coordinate, i.e. corresponding to pyrlevel 0).
        ref_check_min_val : :obj:`float`, optional
            if attribute ``roi_ref_check`` is a valid ROI, then only images
            are included in the stack that exceed the specified intensity
            value (can e.g. be optical density or minimum gas CD in calib
            mode)
        ref_check_max_val : :obj:`float`, optional
            if attribute ``roi_ref_check`` is a valid ROI, then only images
            are included in the stack that are smaller than the specified
            intensity value (can e.g. be optical density or minimum gas CD in
            calib mode)
        dtype
            data type of stack

        Returns
        -------
        ImgStack
            image stack containing stacked images

        """
        self.edit_active = True
        cfn = self.cfn
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        num = self._iter_num(start_idx, stop_idx)
        # remember last image shape settings
        _roi = deepcopy(self._roi_abs)
        _pyrlevel = self.pyrlevel
        _crop = self.crop

        self.auto_reload = False
        if pyrlevel is not None and pyrlevel != _pyrlevel:
            logger.info(f"Changing image list pyrlevel from {_pyrlevel} to {pyrlevel}")
            self.pyrlevel = pyrlevel
        if check_roi(roi_abs):
            logger.info(f"Activate cropping in ROI {roi_abs} (absolute coordinates)")
            self.roi_abs = roi_abs
            self.crop = True

        if stack_id is None:
            stack_id = self.list_id

        self.goto_img(start_idx)

        self.auto_reload = True
        h, w = self.current_img().shape
        stack = ImgStack(h, w, num, dtype, stack_id, camera=self.camera,
                         img_prep=self.current_img().edit_log)
        lid = self.list_id
        ref_check = True
        if not check_roi(ref_check_roi_abs):
            ref_check = False
        if ref_check_min_val is not None:
            ref_check_min_val = float(ref_check_min_val)
        else:
            ref_check = False
        if ref_check_max_val is not None:
            ref_check_max_val = float(ref_check_max_val)
        else:
            ref_check = False
        exp = int(10**exponent(num) / 4.0)
        if not exp:
            exp = 1
        for k in range(num):
            if k % exp == 0:
                print_log.info(f"Building img-stack from list {lid}, progress: ({k} | {num - 1})")
            img = self.loaded_images["this"]
            append = True
            if ref_check:
                sub_val = img.crop(roi_abs=ref_check_roi_abs, new_img=1).mean()
                if not ref_check_min_val <= sub_val <= ref_check_max_val:
                    print_log.warning("Exclude image no. %d from stack, got value=%.2f in "
                          "ref check ROI (out of specified range)"
                          % (k, sub_val))
                append = False
            if append:
                stack.add_img(img.img, img.meta["start_acq"],
                              img.meta["texp"])
            self.goto_next()
            k += 1
        stack.start_acq = asarray(stack.start_acq)
        stack.texps = asarray(stack.texps)
        stack.roi_abs = self._roi_abs

        print_log.info("Img stack calculation finished, rolling back to intial list"
              "state:\npyrlevel: %d\ncrop modus: %s\nroi (abs coords): %s "
              % (_pyrlevel, _crop, _roi))
        self.auto_reload = False
        self.pyrlevel = _pyrlevel
        self.crop = _crop
        self.roi_abs = _roi
        self.goto_img(cfn)
        self.auto_reload = True
        if not sum(stack._access_mask) > 0:
            raise ValueError("Failed to build stack, stack is empty...")
        return stack

    def get_mean_img(self, start_idx=0, stop_idx=None):
        """Determine an average image from a number of list images.

        Parameters
        ----------
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`, optional
            index of last considered image (if None, the last image in this
            list is used). Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)

        Returns
        -------
        Img
            average image

        """
        cfn = self.index
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        self.goto_img(start_idx)
        num = self._iter_num(start_idx, stop_idx)
        img = Img(zeros(self.current_img().shape))
        img.edit_log = self.current_img().edit_log
        img.meta["start_acq"] = self.current_time()
        added = 0
        texps = []
        for k in range(num):
            try:
                cim = self.current_img()
                img.img += cim.img
                try:
                    texps.append(cim.texp)
                except BaseException:
                    pass
                self.goto_next()
                added += 1
            except BaseException:
                print_log.warning("Failed to add image at index %d" % k)
        img.img = img.img / added
        img.meta["stop_acq"] = self.current_time()
        if len(texps) == added:
            img.meta["texp"] = asarray(texps).mean()
        self.goto_img(cfn)
        return img

    def get_mean_tseries_rects(self, start_idx, stop_idx, *rois):
        """Similar to :func:`get_mean_value` but for multiple rects.

        Parameters
        ----------
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`
            index of last considered image (if None, the last image in this
            list is used). Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        *rois
            non keyword args specifying rectangles for data access

        Returns
        -------
        tuple
            N-element tuple containing :class:`PixelMeanTimeSeries` objects
            (one for each ROI specified on input)

        """
        if not self.data_available:
            raise IndexError("No images available in ImgList object")
        dat = []
        num_rois = len(rois)
        if num_rois == 0:
            raise ValueError("No ROIs provided...")
        for roi in rois:
            dat.append([[], [], [], []])
        cfn = self.cfn
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        self.goto_img(start_idx)
        num = self._iter_num(start_idx, stop_idx)

        lid = self.list_id
        pnum = int(10**exponent(num) / 2.0)
        for k in range(num):
            try:
                if k % pnum == 0:
                    print_log.info("Calc pixel mean t-series in list %s (%d | %d)"
                          % (lid, (k + 1), num))
            except BaseException:
                pass
            img = self.loaded_images["this"]
            for i in range(num_rois):
                roi = rois[i]
                d = dat[i]
                d[0].append(img.meta["texp"])
                d[1].append(img.meta["start_acq"])
                sub = img.img[roi[1]:roi[3], roi[0]:roi[2]]
                d[2].append(sub.mean())
                d[3].append(sub.std())

            self.goto_next()

        self.goto_img(cfn)
        means = []
        for i in range(num_rois):
            d = dat[i]
            mean = PixelMeanTimeSeries(d[2], d[1], d[3], d[0], rois[i],
                                       img.edit_log)
            means.append(mean)
        return means

    def get_mean_value(self, start_idx=0, stop_idx=None, roi=DEFAULT_ROI,
                       apply_img_prep=True):
        """Determine pixel mean value time series in ROI.

        Determines the mean pixel value (and standard deviation) for all images
        in this list. Default ROI is the whole image and can be set via
        input param roi, image preparation can be turned on or off.

        Parameters
        ----------
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`
            index of last considered image (if None, the last image in this
            list is used). Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        roi : list
            rectangular region of interest ``[x0, y0, x1, y1]``,
            defaults to [0, 0, 9999, 9999] (i.e. whole image)
        apply_img_prep : bool
            if True, img preparation is performed as specified in
            ``self.img_prep`` dictionary, defaults to True

        Returns
        -------
        PixelMeanTimeSeries
            time series of retrieved values

        """
        if not self.data_available:
            raise IndexError("No images available in ImgList object")
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        self.edit_active = apply_img_prep
        self.goto_img(start_idx)
        num = self._iter_num(start_idx, stop_idx)

        cfn = self.cfn
        vals, stds, texps, acq_times = [], [], [], []
        lid = self.list_id
        pnum = int(10**exponent(num) / 4.0)
        for k in range(num):
            try:
                if k % pnum == 0:
                    print_log.info("Calc pixel mean t-series in list %s (%d | %d)"
                          % (lid, (k + 1), num))
            except BaseException:
                pass
            img = self.loaded_images["this"]
            texps.append(img.meta["texp"])
            acq_times.append(img.meta["start_acq"])
            sub = img.img[roi[1]:roi[3], roi[0]:roi[2]]
            vals.append(sub.mean())
            stds.append(sub.std())

            self.goto_next()

        self.goto_img(cfn)

        return PixelMeanTimeSeries(vals, acq_times, stds, texps, roi,
                                   img.edit_log)

    def current_edit(self):
        """Return :attr:`edit_log` of current image."""
        return self.current_img().edit_log

    def edit_info(self):
        """Print the current image preparation settings."""
        d = self.current_img().edit_log
        print_log.info("\nImgList %s, image edit info\n----------------------------"
              % self.list_id)
        for key, val in d.items():
            print_log.info("%s: %s" % (key, val))

    def add_gaussian_blurring(self, sigma=1):
        """Increase amount of gaussian blurring on image load.

        :param int sigma (1): Add width gaussian blurring kernel
        """
        self.img_prep["blurring"] += sigma
        self.load()

    def cam_id(self):
        """Get the current camera ID (if camera is available)."""
        return self.camera.cam_id

    def current_time(self):
        """Get the acquisition time of the current image from image meta data.

        Raises
        ------
        IndexError
            if list does not contain images

        Returns
        -------
        datetime
            start acquisition time of currently loaded image

        """
        return self.current_img().meta["start_acq"]

    def current_time_str(self, format='%H:%M:%S'):
        """Return a string of the current acq time."""
        return self.current_img().meta["start_acq"].strftime(format)

    def current_img(self, key="this"):
        """Get the current image object.

        Parameters
        ----------
        key : str
            this" or "next"

        Returns
        -------
        Img
            currently loaded image in list

        """
        img = self.loaded_images[key]
        if not isinstance(img, Img):
            logger.info("CALLING LOAD IN CURRENT_IMG %s, list %s"
                  % (key, self.list_id))
            self.load()
            img = self.loaded_images[key]
        return img

    def show_current(self, **kwargs):
        """Show the current image."""
        return self.current_img().show(**kwargs)

    def append(self, file_path):
        """Append image file to list.

        :param str file_path: valid file path
        """
        if not exists(file_path):
            raise IOError("Image file path does not exist %s" % file_path)

        self.files.append(file_path)

    def plot_mean_value(self, roi=DEFAULT_ROI, yerr=False, ax=None):
        """Plot mean value of image time series.

        Parameters
        ----------
        roi : list
            rectangular ROI in which mean is determined (default is
            ``[0, 0, 9999, 9999]``, i.e. whole image)
        yerr : bool
            include errorbars (std), defaults to False
        ax : :obj:`Axes`, optional
            matplotlib axes object

        Returns
        -------
        Axes
            matplotlib axes object

        """
        if ax is None:
            fig = figure()  # figsize=(16, 6))
            ax = fig.add_subplot(1, 1, 1)

        mean = self.get_mean_value()
        ax = mean.plot(yerr=yerr, ax=ax)
        return ax

    def plot_tseries_vert_profile(self, pos_x, start_y=0, stop_y=None,
                                  step_size=0.1, blur=4):
        """Plot the temporal evolution of a line profile.

        Parameters
        ----------
        pos_x : int
            number of pixel column
        start_y : int
            Start row of profile (y coordinate, default: 10)
        stop_y : int
            Stop row of profile (is set to rownum - 10pix if input is None)
        step_size : float
            stretch between different line profiles of the evolution (0.1)
        blur : int
            blurring of individual profiles (4)

        Returns
        -------
        Figure
            figure containing result plot

        """
        cfn = deepcopy(self.index)
        self.goto_img(0)
        name = "vertAtCol" + str(pos_x)
        h, w = self.get_img_shape()
        h_rel = float(h) / w
        width = 18
        height = int(9 * h_rel)
        if stop_y is None:
            stop_y = h - 10
        l = LineOnImage(pos_x, start_y, pos_x, stop_y, name)
        fig = figure(figsize=(width, height))
        # fig,axes=plt.subplots(1,2,sharey=True,figsize=(width,height))
        cidx = 0
        img_arr = self.loaded_images["this"].img
        rad = gaussian_filter(l.get_line_profile(img_arr), blur)
        del_x = int((rad.max() - rad.min()) * step_size)
        y_arr = arange(start_y, stop_y, 1)
        ax1 = fig.add_axes([0.1, 0.1, 0.35, 0.8])
        times = self.get_img_meta_all_filenames()[0]
        if any([x is None for x in times]):
            raise ValueError("Cannot access all image acq. times")
        idx = []
        idx.append(cidx)
        for k in range(1, self.nof):
            rad = rad - rad.min() + cidx
            ax1.plot(rad, y_arr, "-b")
            img_arr = self.goto_next().img
            rad = gaussian_filter(l.get_line_profile(img_arr), blur)
            cidx = cidx + del_x
            idx.append(cidx)
        idx = asarray(idx)
        ax1.set_ylim([0, h])
        ax1.invert_yaxis()
        draw()
        new_labels = []
        ticks = ax1.get_xticklabels()
        new_labels.append("")
        for k in range(1, len(ticks) - 1):
            tick = ticks[k]
            index = argmin(abs(idx - int(tick.get_text())))
            new_labels.append(times[index].strftime("%H:%M:%S"))
        new_labels.append("")
        ax1.set_xticklabels(new_labels)
        ax1.grid()
        self.goto_img(cfn)
        ax2 = fig.add_axes([0.55, 0.1, 0.35, 0.8])
        l.plot_line_on_grid(self.loaded_images["this"].img, ax=ax2)
        ax2.set_title(self.loaded_images["this"].meta["start_acq"].strftime(
            "%d.%m.%Y %H:%M:%S"))
        return fig

    def _this_raw_fromfile(self):
        """Reload and return current image.

        This method is used for test purposes and does not change the list
        state. See for instance :func:`activate_dilution_corr` in
        :class:`ImgList`

        Returns
        -------
        Img
            the current image loaded and unmodified from file

        """
        return self._load_image(self.index)

    def _load_image(self, list_index):
        """Load the actual image data for a given index.

        Parameters
        ----------
        list_index : int
            Index of image in file list ``self.files``

        Returns
        -------
        Img
            the loaded image data (unmodified)

        """
        file_path = self.files[list_index]
        try:
            meta = self.get_img_meta_from_filename(file_path)
        except:
            print_log.warning("Failed to retrieve image meta information from file path %s"
                 % file_path)
            meta = {}
        meta["filter_id"] = self.list_id
        return Img(file_path,
                   import_method=self.camera.image_import_method,
                   **meta)

    def _apply_edit(self, key):
        """Apply the current image edit settings to image.

        :param str key: image id (e.g. this)
        """
        if not self.edit_active:
            logger.debug(f"Edit not active in img_list {self.list_id}: no image preparation will be performed")
            return
        img = self.loaded_images[key]
        img.to_pyrlevel(self.img_prep["pyrlevel"])
        if self.img_prep["crop"]:
            img.crop(self.roi_abs)
        img.add_gaussian_blurring(self.img_prep["blurring"])
        img.apply_median_filter(self.img_prep["median"])
        if self.img_prep["8bit"]:
            img._to_8bit_int(new_im=False)
        self.loaded_images[key] = img

    def _iter_num(self, start_idx, stop_idx):
        """Return the number of iterations for a loop.

        The number of iterations is based on the current attribute
        ``skip_files``.

        Parameters
        ----------
        start_idx : int
            start index of loop
        stop_idx : int
            stop index of loop

        Returns
        -------
        int
            number of required iterations

        """
        # the int(x) function rounds down, so no floor(x) needed
        return int((stop_idx - start_idx) / (self.skip_files + 1.0))

    def _first_file(self):
        """Get first file path of image list."""
        if not bool(self.files):
            raise IndexError('ImgList is empty...')
        return self.files[0]

    def _last_file(self):
        """Get last file path of image list."""
        if not bool(self.files):
            raise IndexError('ImgList is empty...')
        return self.files[self.nof - 1]

    def _make_header(self):
        """Make header for current image (based on image meta information)."""
        try:
            im = self.current_img()
            if not isinstance(im, Img):
                raise Exception("Current image not accessible in ImgList...")

            s = ("%s (Img %s of %s), read_gain %s, texp %.2f s"
                 % (self.current_time().strftime('%d/%m/%Y %H:%M:%S'),
                    self.index + 1, self.nof, im.meta["read_gain"],
                    im.meta["texp"]))
            return s

        except Exception as e:
            logger.warning(repr(e))
            return "Creating img header failed..."

    def _get_and_set_geometry_info(self):
        """Compute and write plume and pix-to-pix distances from MeasGeometry.
        """
        try:
            (int_steps, _,
             dists) = self.meas_geometry.compute_all_integration_step_lengths()
            self._plume_dists = dists  # .to_pyrlevel(0)
            self._integration_step_lengths = int_steps
            logger.info("Computed and updated list attributes plume_dist and "
                  "integration_step_length in ImgList from MeasGeometry")
        except BaseException:
            raise ValueError("Measurement geometry not ready for access "
                             "of plume distances and integration steps in "
                             "image list %s."
                             % self.list_id)

    def __str__(self):
        s = "\npyplis ImgList\n----------------------------------\n"
        s += "ID: %s\nType: %s\n" % (self.list_id, self.list_type)
        s += "Number of files (imgs): %s\n\n" % self.nof
        s += "Current image prep settings\n.................................\n"
        if not self.has_files():
            return s
        try:
            for k, v in self.current_img().edit_log.items():
                s += "%s: %s\n" % (k, v)
            if self.crop is True:
                s += "Cropped in ROI\t[x0, y0, x1, y1]:\n"
                s += "  Absolute coords:\t%s\n" % self.roi_abs
                s += "  @pyrlevel %d:\t%s\n" % (self.pyrlevel, self.roi)
        except BaseException:
            s += "FATAL: Image access failed, msg\n: %s" % format_exc()
        return s

    def __call__(self, num=0):
        """Change current file number, load and return image.

        :param int num: file number
        """
        return self.goto_img(num)

    def __getitem__(self, name):
        """Get item method."""
        if name in self.__dict__:
            return self.__dict__[name]
        for k, v in self.__dict__.items():
            try:
                if name in v:
                    return v[name]
            except BaseException:
                pass