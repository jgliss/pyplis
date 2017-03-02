# -*- coding: utf-8 -*-
"""
Module containing the :class:`Dataset` object which is important for 
automated separation of image files by their type (e.g. on-band, off-band, 
dark, offset) using information from a file naming convention specified 
within a :class:`Camera` object.
"""
from os.path import exists, join, isfile
from os import listdir, walk
from warnings import warn
from datetime import datetime, date
from numpy import inf
from matplotlib.pyplot import subplots, FuncFormatter, tight_layout, Line2D
from matplotlib.patches import Rectangle

from copy import deepcopy
from traceback import format_exc
from collections import OrderedDict as od

from .imagelists import ImgList, DarkImgList
from .helpers import shifted_color_map
from .setupclasses import MeasSetup
   
class Dataset(object):
    """Class for data import management
    
    Default input is a :class:`pyplis.setupclasses.MeasSetup` object, which
    specifies the camera used (e.g. file naming convention, detector specifics)
    the measurement geometry and information about the source and
    meteorological wind direction, start / stop time stamps and the image 
    base directory.
    
    This object finds all images in the 
    """
    def __init__(self, input=None, init=1):
        self.setup = None
    
        self._lists_intern = od()
        self.lists_access_info = od()
    
        self.load_input(input)
    
        if init:                                               
            self.init_image_lists()
                                                        
    def load_input(self, input):
        """Extract information from input and set / update self.setup"""
        print "Checking and loading input in Dataset"
        if self.set_setup(input):
            return 1
        msg = ("Input is not MeasSetup, create new (default) MeasSetup object")
        print msg
        self.setup = MeasSetup()
        if input == None:
            return 1
        if isinstance(input, str) and exists(input):            
            msg = ("Input is filepath, assuming this to be image base path: "
                "updating base_dir variable in self.setup")
            print msg
            self.change_img_base_dir(input)
        else:
            msg=("Unkown input setup type: " + str(type(input)))
            raise TypeError(msg)   
        
        return 1
    
    def set_setup(self, stp):
        """Set the current :class:`MeasSetup` object"""
        if isinstance(stp, MeasSetup):
            print "Updating setup in Dataset...."
            self.setup = stp
            return 1
        return 0
        
    def reset_all_img_lists(self):
        """Initialisation of all image lists, old lists are deleted"""
        self._lists_intern = od()
        for key, f in self.filters.filters.iteritems():
            l = ImgList(list_id=key, list_type=f.type, camera=self.camera)
            l.filter = f
            if not self._lists_intern.has_key(f.meas_type_acro):
                self._lists_intern[f.meas_type_acro] = od()
            self._lists_intern[f.meas_type_acro][f.acronym] = l
            self.lists_access_info[f.id] = [f.meas_type_acro, f.acronym]
            
        if not bool(self.camera.dark_info):
            msg = ("Warning: dark image lists could not be initiated, no dark "
                "image file information available in `self.camera`")
            print msg
            return 0    
        
        for item in self.camera.dark_info:
            l = DarkImgList(list_id=item.id, list_type=item.type,
                            read_gain=item.read_gain, camera=self.camera)
            l.filter = item
            if not self._lists_intern.has_key(item.meas_type_acro):
                self._lists_intern[item.meas_type_acro] = od()
            self._lists_intern[item.meas_type_acro][item.acronym] = l
            self.lists_access_info[item.id] = [item.meas_type_acro,\
                                                            item.acronym]
    
    def get_all_filepaths(self):
        """Gets all valid file paths"""
        print "\nSEARCHING VALID FILE PATHS IN\n%s\n" %self.base_dir
        
        p = self.base_dir
        if not isinstance(self.camera.file_type, str):
            print ("file_type not specified in Dataset..."
                "Using all files and file_types")
            self.setup.options["USE_ALL_FILES"] = True
            self.setup.options["USE_ALL_FILE_TYPES"] = True
     
        if p is None or not exists(p):
            message = ('Error: path %s does not exist' %p)
            print message 
            return []
        
        if not self.INCLUDE_SUB_DIRS:
            print "Exclude files from subdirectories"
            if self.USE_ALL_FILE_TYPES:
                print "Using all file types"
                all_paths = [join(p, f) for f in listdir(p) if\
                                                    isfile(join(p, f))]
            else:
                print "Using only %s files" %self.file_type
                all_paths = [join(p, f) for f in listdir(p) if\
                        isfile(join(p, f)) and f.endswith(self.file_type)]
            
        else:
            print "Include files from subdirectories"
            all_paths = []
            if self.USE_ALL_FILE_TYPES:
                print "Using all file types"
                for path, subdirs, files in walk(p):
                   for filename in files:
                       all_paths.append(join(path, filename))
            else:
                print "Using only %s files" %self.file_type
                for path, subdirs, files in walk(p):
                    for filename in files:
                        if filename.endswith(self.file_type):
                            all_paths.append(join(path, filename))
    
        all_paths.sort() 
        print ("Total number of files found %s" %len(all_paths))
        
        return all_paths
        
    def init_image_lists(self):
        """Import all images and create image list objects"""
        
        print "\n+++++++++++++++++++++++++++++++++++++++++++++++++"
        print "+++++++++ INIT IMAGE LISTS IN DATASET +++++++++++"
        print "+++++++++++++++++++++++++++++++++++++++++++++++++\n"
        
        warnings = []
        cam = self.camera
        
        #: create img list objects for each filter and for dark / offset lists
        self.reset_all_img_lists() 
        #: check if image filetype is specified and if not, set option to use 
        #: all file types
        self._check_file_type() 
        if self.base_dir is None or not exists(self.base_dir):
            s = ("Warning: image base directory does not exist, method "
                "init_image_lists aborted in Dataset")
            warnings.append(s)
            warn(s)
            return False
        #: load all file paths
        paths = self.get_all_filepaths()
        # paths now includes all valid paths dependent on whether file_type is
        # specified or not and whether also subdirectories were considered
        if not bool(paths):
            s= ("Warning: lists could not be initiated, no valid files found "
                "method init_image_lists aborted in Dataset")
            warnings.append(s)
            warn(s)
            return False
        # check which image meta information can be accessed from first file in
        # list (updates ``_fname_access_flags`` in :class:`Camera`)
        self.check_filename_info_access(paths[0])
        
        #get the current meta access flags
        flags = cam._fname_access_flags
        if self.USE_ALL_FILES and flags["start_acq"]:
            #take all files in the basefolder (i.e. set start and stop date the 
            #first and last date of the files in the folder)
            self.setup.start = self.camera.get_img_meta_from_filename(
                                                                paths[0])[0]
            self.setup.stop = self.camera.get_img_meta_from_filename(
                                                                paths[-1])[0]        
        
        #: Set option to use all files in case acquisition time stamps cannot
        #: be accessed from filename
        if not flags["start_acq"]:
            print ("Acquisition time access from filename not possible, using "
                "all files")
            self.setup.options["USE_ALL_FILES"] = True
        
        #: Separate the current list based on specified time stamps
        if not self.setup.options["USE_ALL_FILES"]:
            paths_temp = self.extract_files_time_ival(paths)
            if not bool(paths_temp): #check if any files were found in specified t-window
                s = ("No images found in specified time interval "
                    "%s - %s, mode was changed to: USE_ALL_FILES=True" 
                    %(self.start, self.stop))
                warnings.append(s)
                self.setup.options["USE_ALL_FILES"] = True
            else:
                paths = paths_temp
                
        if not (flags["filter_id"] and flags["meas_type"]):
            #: it is not possible to separate different image types (on, off, 
            #: dark..) from filename, thus all are loaded into on image list
            warnings.append("Images can not be separated by type / meas_type "
                "(e.g. on, off, dark, offset...) from filename info, loading "
                "all files into on-band list")
            self.setup.options["SEPARATE_FILTERS"] = False
            i = self.lists_access_info[self.filters.default_key_on]
            self._lists_intern[i[0]][i[1]].add_files(paths)
            [warn(x) for x in warnings]
            return True
            
        not_added = 0
        #: now perform separation by meastype and filter
        for p in paths:
            try:
                _, filter_id, meas_type, _, _ = self.camera.\
                                        get_img_meta_from_filename(p)
                self._lists_intern[meas_type][filter_id].files.append(p)
            except Exception as e:
                print repr(e)
                not_added += 1
        
        for meas_type, sub_dict in self._lists_intern.iteritems():
            for filter_id, lst in sub_dict.iteritems():
                lst.init_filelist()
        
        self.assign_dark_offset_lists()
        
        try:
            off_list = self.get_list(self.filters.default_key_off)
            self.get_list(self.filters.default_key_on).link_imglist(off_list)
        except:
            pass
        [warn(x) for x in warnings]
        return True
        
    def check_filename_info_access(self, filepath):
        """Checks which information can be accessed from file name
        
        :param str filepath: image file path        
        """
        err = self.camera.get_img_meta_from_filename(filepath)[4]
        for item in err:
            print item
        return self.camera._fname_access_flags
        
    def change_img_base_dir(self, img_dir):
        """Set or update the current base_dir. 
        
        :param str p: new path
        """
        if not exists(img_dir):
            msg = ("Could not update base_dir, input path %s does not exist" 
                                                                    %img_dir)
            print msg
            self.warnings.append(msg)
            return 0
        self.setup.base_dir = img_dir
    
    def _check_file_type(self):
        """Check if filtype information is available 
        
        Sets::
        
            self.USE_ALL_FILE_TYPES = True 
            
        if file type information can not be accessed
        
        """
        info = self.camera
        val = True
        if isinstance(info.file_type, str):
            val = False
        self.setup.USE_ALL_FILE_TYPES = val
    
    def extract_files_time_ival(self, all_paths):
        """Extracts all files belonging to specified time interval
        
        :param list all_paths: list of image filepaths
        """
        if not self.camera._fname_access_flags["start_acq"]:
            warn("Acq. time information cannot be accessed from file names")
            return all_paths
        acq_time0 = self.camera.get_img_meta_from_filename(all_paths[0])[0]
        if acq_time0.date() == date(1900, 1, 1):
            paths=self._find_files_ival_time_only(all_paths)
        else:
            paths=self._find_files_ival_datetime(all_paths)
        
        if not bool(paths):
            warn("Error: no files could be found in specified time "
                "interval %s - %s" %(self.start, self.stop))
            self.USE_ALL_FILES = True
        else:
            print("%s files of type were found in specified time interval %s "
                "- %s" %(len(paths), self.start, self.stop))
        return paths
    def _find_files_ival_time_only(self, all_paths):
        """Extracts all files belonging to specified time interval
        
        :param list all_paths: list of image filepaths
        """
        paths = []   
        start = self.start.time()
        stop = self.stop.time()
        for path in all_paths:    
            acq_time = self.camera.get_img_meta_from_filename(path)[0].time()
            if start <= acq_time <= stop:
                paths.append(path)         
        
        
    
    def _find_files_ival_datetime(self, all_paths):
        """Extracts all files belonging to specified time interval
        
        This function considers the datetime stamps of ``self.start`` and
        ``self.stop``, see also :func:`_find_files_ival_time_only` which only
        uses the actual time to find valid files.
        
        :param list all_paths: list of image filepaths
        """
        paths = []   
        for path in all_paths:    
            acq_time = self.camera.get_img_meta_from_filename(path)[0]
            if self.start <= acq_time <= self.stop:
                paths.append(path)         

        if not bool(paths):
            warn("Error: no files could be found in specified time "
                "interval %s - %s" %(self.start, self.stop))
        else:
            print("%s files of type were found in specified time interval %s "
                "- %s" %(len(paths), self.start, self.stop))
        return paths
        
    def find_closest_img(self, filename, in_list, acronym, meas_type_acro):
        """Find closest-in-time image to input image file
        
        :param str filename: image filename
        :param str in_list: input list with filepaths 
        :param str acronym: the acronym of the image type to be searched (e.g.
            an acronym for a dark image as specified in camera)
        :param str meas_type_acro: meas type acronym of image type to be 
            searched (e.g. an acronym for a dark image as specified in 
            camera)
        """   
        t0 = self.camera.get_img_meta_from_filename(filename)[0]
        del_t = inf
        idx = -1
        for k in range(len(in_list)):
            t1, f1, tp1, _, _ = self.camera.get_img_meta_from_filename(\
                                                                in_list[k])
            if f1 == acronym and abs(t1 - t0).total_seconds() < del_t and\
                        meas_type_acro == tp1:
                del_t = abs(t1 - t0).total_seconds()
                idx = k
        if idx == -1 or del_t == inf:
            raise Exception("Error in func find_closest_img: no match")
     
        return in_list[idx]
    
    def all_lists(self):
        """Returns list containing all available image lists
        
        Loops over ``self._lists_intern`` and the corresponding sub directories
        """
        lists = []
        for meas_type, sub_dict in self._lists_intern.iteritems():
            for filter_id, lst in sub_dict.iteritems():
                lists.append(lst)
        return lists        
        
    @property
    def dark_ids(self):
        """Get all dark IDs"""
        ids = []
        for info in self.camera.dark_info:
            ids.append(info.id)
        return ids
    
    
    def assign_dark_offset_lists(self, into_list=None):
        """Assign dark and offset lists to image lists ``self.lists``
        
        Assign dark and offset lists in filter lists for automatic dark and
        offset correction. The lists are set dependent on the read_gain mode of
        the detector
        
        :param ImgList into_list (None): optional input, if specified, the dark 
            assignment is performed only in the input list
        """    
        if isinstance(into_list, ImgList):            
            into_list.link_dark_offset_lists(self.dark_lists_with_data)
            return True
        
        no_dark_ids = self.check_dark_lists()
        if len(no_dark_ids) > 0:
            self.find_master_darks(no_dark_ids)
            
        for filter_id, lst in self.img_lists.iteritems():
            if lst.nof > 0:
                print ("Assigning dark and offset lists in image list %s" 
                                                                %filter_id)
                lists = od()
                if self.camera.meas_type_pos != self.camera.filter_id_pos:
                    for dark_acro in self.camera.dark_meas_type_acros:
                        try:
                            dark_lst = self._lists_intern[dark_acro]\
                                                    [lst.filter.acronym]
                            lists[dark_lst.list_id] = dark_lst
                            print ("Found dark list match for image list %s,"
                             "dark ID: %s" %(lst.list_id, dark_lst.list_id))
                        except:
                            pass
                    for offset_acro in self.camera.offset_meas_type_acros:
                        try:
                            offs_lst = self._lists_intern[offset_acro]\
                                                    [lst.filter.acronym]
                            lists[offs_lst.list_id] = offs_lst
                            print ("Found offset list match for image list %s:"
                             "dark ID: %s" %(lst.list_id, offs_lst.list_id))
                        except:
                            pass
                if not lists:
                    lists = self.dark_lists_with_data
                
                lst.link_dark_offset_lists(lists)
        return True
        
    def get_all_dark_offset_lists(self):
        """Get all dark and offset image lists"""
        lists = od()
        for dark_id in self.dark_ids:
            info = self.lists_access_info[dark_id]
            lists[dark_id] = self._lists_intern[info[0]][info[1]]
        return lists
        
    @property
    def dark_lists(self):
        """Wrapper for :func:`get_all_dark_offset_lists`"""
        return self.get_all_dark_offset_lists()
    
    @property
    def dark_lists_with_data(self):
        """Returns all dark/offset list that include image data"""
        lists = od()
        for dark_id, lst in self.dark_lists.iteritems():
            if lst.nof > 0:
                lists[dark_id] = lst
        return lists
                
    @property
    def filter_ids(self):
        """Get all dark IDs"""
        return self.filters.filters.keys()
    
    def get_all_image_lists(self):
        """Get all image lists (without dark and offset lists)"""
        lists = od()
        for filter_id in self.filter_ids:
            info = self.lists_access_info[filter_id]
            lists[filter_id] = self._lists_intern[info[0]][info[1]]
        return lists
        
    @property
    def img_lists(self):
        """Wrapper for :func:`get_all_image_lists`"""
        return self.get_all_image_lists()
    
    @property
    def img_lists_with_data(self):
        """Wrapper for :func:`get_all_image_lists`"""
        lists = od()
        for key, lst in self.img_lists.iteritems():
            if lst.nof > 0:
                lists[key] = lst
        return lists
        
    def check_dark_lists(self):
        """Checks all dark lists whether they contain images or not"""
        no_data_ids = []
        for dark_id, lst in self.dark_lists.iteritems():
            if not lst.nof > 0:
                no_data_ids.append(lst.list_id)
        return no_data_ids
    
    def find_master_dark(self):
        """Search master dark image for specific dark list
        
        Search a master dark image for all dark image lists that do not
        contain images
        """
        print "\nCHECKING DARK IMAGE LISTS IN DATASET"
        flags = self.camera._fname_access_flags
        if not (flags["filter_id"] and flags["meas_type"]):
            #: it is not possible to separate different image types (on, off, 
            #: dark..) from filename, thus dark or offset images can not be searched
            return []
            
        all_files = self.get_all_filepaths()
        l = self.get_list(self.filters.default_key_on)
        if l.data_available:
            f_name = l.files[int(l.nof/2)]
        else:
            f_name = all_files[int(len(all_files)/2.)]
        failed_ids = [] 
        if not bool(dark_ids):
            dark_ids = self.dark_lists.keys()
        for dark_id in dark_ids:
            lst = self.dark_lists[dark_id]
            if not lst.nof > 0:
                meas_type_acro, acronym = self.lists_access_info[dark_id]
                print ("\nSearching master dark image for\nID: %s\nacronym: %s"
                  "\nmeas_type_acro: %s" %(dark_id, acronym, meas_type_acro))
                try:
                    p = self.find_closest_img(f_name, all_files, acronym,
                                              meas_type_acro)
                    lst.files.append(p)
                    lst.init_filelist()
                    print "Found dark image for ID %s\n" %dark_id
                except:
                    print "Failed to find dark image for ID %s\n" %dark_id
                    failed_ids.append(dark_id)
                    
        return failed_ids
        
    def find_master_darks(self, dark_ids = []):
        """Search master dark image for each dark type
        
        Search a master dark image for all dark image lists that do not
        contain images
        """
        print "\nCHECKING DARK IMAGE LISTS IN DATASET"
        flags = self.camera._fname_access_flags
        if not (flags["filter_id"] and flags["meas_type"]):
            #: it is not possible to separate different image types (on, off, 
            #: dark..) from filename, thus dark or offset images can not be searched
            return []
            
        all_files = self.get_all_filepaths()
        l = self.get_list(self.filters.default_key_on)
        if l.data_available:
            f_name = l.files[int(l.nof/2)]
        else:
            f_name = all_files[int(len(all_files)/2.)]
        failed_ids = [] 
        if not bool(dark_ids):
            dark_ids = self.dark_lists.keys()
        for dark_id in dark_ids:
            lst = self.dark_lists[dark_id]
            if not lst.nof > 0:
                meas_type_acro, acronym = self.lists_access_info[dark_id]
                print ("\nSearching master dark image for\nID: %s\nacronym: %s"
                  "\nmeas_type_acro: %s" %(dark_id, acronym, meas_type_acro))
                try:
                    p = self.find_closest_img(f_name, all_files, acronym,
                                              meas_type_acro)
                    lst.files.append(p)
                    lst.init_filelist()
                    print "Found dark image for ID %s\n" %dark_id
                except:
                    print "Failed to find dark image for ID %s\n" %dark_id
                    failed_ids.append(dark_id)
                    
        return failed_ids
    
    def check_image_access_dark_lists(self):
        """Check whether dark and offset image lists contain at least one img"""
        
        for lst in self.dark_lists.values():
            if not lst.data_available:
                return False
        return True
        
    
    """Helpers"""
    def images_available(self, filter_id):
        """Check if image list has images
        
        :param str filter_id: string (filter) ID of image list
        """
        try: 
            if self.get_list(filter_id).nof > 0:
                return 1
            return 0
        except:
            return 0
    
    def current_image(self, filter_id):
        """Get current image of image list
        
        :param str filter_id: filter ID of image list
        """
        try:
            return self.get_list(filter_id).current_img()
        except:
            return 0
    
    
    def get_list(self, list_id):
        """Get image list for one filter
        
        :param str filter_id: filter ID of image list (e.g. "on")
        
        """
        if not list_id in self.lists_access_info.keys():
            raise KeyError("%s ImgList could not be found..." %list_id)
        info = self.lists_access_info[list_id]
        lst = self._lists_intern[info[0]][info[1]]
        if not lst.nof > 0:
            warn("Image list %s does not contain any images" %list_id)
        return lst
    
    def get_current_img_prep_dict(self, list_id = None):
        """Get the current image preparation settings from one image list
        
        :param str list_id:  ID of image list 
        """
        if list_id is None:
            list_id = self.filters.default_key_on
        return self.get_list(list_id).img_prep
 
    
    def load_images(self):
        """This function loads the current images in all ImageLists in the 
        :mod:`SortedList` object of this :mod:`Dataset`.
        """  
        for lst in self.all_lists():
            lst.load()    
        
    def update_image_prep_settings(self, **settings):
        """Update image preparation settings in all image lists"""
        for list_id, lst in self.img_lists.iteritems():
            print "Checking changes in list %s: " %list_id
            val = lst.update_img_prep_settings(**settings)
            print "list %s updated (0 / 1): %s" %(list_id, val)
    
    def update_times(self, start, stop, reload=False):
        """Update start and stop times of this dataset and reload
        
        :param datetime start: new start time
        :param datetime stop: new stop time
        :param bool reload (True): if True, re-init this dataset object
            (previous information is lost)
        """
        if not all([isinstance(x, datetime) for x in [start, stop]]):
            raise TypeError("Times could not be changed in Dataset, "
                "wrong input type: %s, %s (need datetime)" 
                %(type(start),type(stop)))
                
        self.setup.start = start
        self.setup.stop = stop
        self.setup.check_timestamps()
        
    def duplicate(self):
        """Duplicate Dataset object"""
        print 'Dataset successfully duplicated'
        return deepcopy(self)
    
    """GUI stuff
    """
#==============================================================================
#     def open_in_gui(self):
#         """Open this dataset in GUI application"""
#         try:
#             import pyplis.gui as gui
#             app=QApplication(argv)
#             
#     #win = DispTwoImages.DispTwoImagesWidget(fileListRight=fileList)
#             win = gui.MainApp.MainApp(self)
#             win.show()
#             app.exec_() #run main loop
#         except:
#             print ("Error: could not open pyplis GUI")
#             raise
#==============================================================================
    
    """
    Plotting etc.
    """    
    def show_current_img(self, filter_id, add_forms = False):
        """Plot current image
        
        :param str filter_id: filter ID of image list (e.g. "on")
        """
        ax = self.current_image(filter_id).show_img()
        if add_forms:
            handles = []
            for k, v in self.lines._forms.iteritems():
                l = Line2D([v[0],v[2]],[v[1],v[3]], color = "#00ff00",\
                                                                    label = k)
                handles.append(l)
                ax.add_artist(l)
            for k, v in self.rects._forms.iteritems():
                w, h = v[2] - v[0], v[3] - v[1]
                r = Rectangle((v[0], v[1]), w, h, ec="#00ff00",fc = "none",\
                                                                    label = k)
                ax.add_patch(r)
                handles.append(r)
            ax.legend(handles= handles, loc = 'best', fancybox = True,\
                                framealpha = 0.5, fontsize = 10).draggable()
                                
        return ax
        #ax.draw()
    
    def plot_mean_value(self, filter_id, yerr = 1, rect = None):
        """Plots the pixel mean value of specified filter of the time span
        covered by this dataset
        """
        self.get_list(filter_id).plot_mean_value(yerr = yerr,rect = rect)
        
    def draw_map_2d(self, *args, **kwargs):
        """Wrapper for :func:`draw_map_2d` of ``self.meas_geometry`` object"""
        return self.meas_geometry.draw_map_2d(*args, **kwargs)
        
    def draw_map_3d(self,*args, **kwargs):
        """Wrapper for :func:`draw_map_3d` of ``self.meas_geometry`` object"""
        return self.meas_geometry.draw_map_3d(*args, **kwargs)    
    """
    Decorators
    """    
    @property 
    def camera(self):
        """Return camera base info object"""
        return self.setup.camera
        
    @camera.setter
    def camera(self, val):
        self.setup.camera = val
        
    @property
    def source(self):
        """Get / set current Source"""
        return self.setup.source
        
    @source.setter
    def source(self, val):
        self.setup.source = val
        
    @property
    def cam_id(self):
        """Returns current camera ID"""
        return self.setup.camera.cam_id
        
    @property
    def base_dir(self):
        """Getter / setter of current image base_dir"""
        return self.setup.base_dir
    
    @base_dir.setter
    def base_dir(self, val):
        if exists(val):
            self.setup.base_dir = val
            self.init_image_lists()
    
    @property
    def USE_ALL_FILES(self):
        """Return USE_ALL_FILES boolen from setup"""
        return self.setup.USE_ALL_FILES
    
    @USE_ALL_FILES.setter
    def USE_ALL_FILES(self, val):
        self.setup.USE_ALL_FILES = val
        print ("Option USE_ALL_FILES was updated in Dataset, please call class"
            " method ``init_image_lists`` in order to apply the changes")
    @property
    def USE_ALL_FILE_TYPES(self):
        """Return USE_ALL_FILE_TYPES option from setup"""
        return self.setup.USE_ALL_FILE_TYPES
    
    @USE_ALL_FILE_TYPES.setter
    def USE_ALL_FILE_TYPES(self, val):
        self.setup.USE_ALL_FILE_TYPES = val
        print ("Option USE_ALL_FILE_TYPES was updated in Dataset, please call "
            "class method ``init_image_lists`` in order to apply the changes")
        
    @property
    def INCLUDE_SUB_DIRS(self):
        """Returns boolean sub directory inclusion option"""
        return self.setup.INCLUDE_SUB_DIRS
    
    @INCLUDE_SUB_DIRS.setter
    def INCLUDE_SUB_DIRS(self, val):
        self.setup.INCLUDE_SUB_DIRS = val
        print ("Option INCLUDE_SUB_DIRS was updated in Dataset, please call "
            "class method ``init_image_lists`` in order to apply the changes")
            
    @property
    def start(self):
        """Getter / setter for current start time stamp"""
        return self.setup.start
        
    @start.setter
    def start(self, val):
        self.setup.start = val
        print ("Start time stamp was updated in Dataset, please call "
            "class method ``init_image_lists`` in order to apply the changes")
    
    @property
    def stop(self):
        """Getter / setter for current stop time stamp"""
        return self.setup.stop
    
    @stop.setter
    def stop(self, val):
        self.setup.stop = val
        print ("Stop time stamp was updated in Dataset, please call "
            "class method ``init_image_lists`` in order to apply the changes")
        
    @property
    def file_type(self):
        """Returns current image file type"""
        return self.setup.camera.file_type
        
    @property
    def meas_geometry(self):
        """Returns current measurement geometry"""
        return self.setup.meas_geometry
        
    @property
    def filters(self):
        """Returns the current filter setup"""
        return self.setup.camera.filter_setup
    
    @property
    def filter_acronyms(self):
        """Make a dictionary of filter IDs and corresponding acronyms"""
        acros = {}
        for key, val in self.filters.filters.iteritems():
            #read the acronym from the filter object
            acros[key] = val.acronym 
        return acros

    @property
    def num_of_filters(self):
        """Returns the number of filters in ``self.filters``"""
        return len(self.filters.keys())
    
    @property
    def _fname_access_flags(self):
        return self.camera._fname_access_flags
    
    @property
    def rects(self):
        """Returns rectangle collection"""
        return self.setup.forms.rects
        
    @rects.setter
    def rects(self, name, value):
        """Setter method for rectangle collection stored in ``self.setup``"""
        self.setup.forms.rects[name] = value
    
    @property
    def lines(self):
        """Returns rectangle collection"""
        return self.setup.forms.lines
        
    @lines.setter
    def lines(self, name, value):
        """Setter method for rectangle collection stored in ``self.setup``"""
        self.setup.forms.lines[name] = value
        
    """Magic methods"""
    def __getitem__(self, key):
        """Get one class item
        
        Searches in ``self.__dict__`` and ``self.setup`` and returns item if 
        match found
        
        :param str key: name of item
        """  
        if self.setup.__dict__.has_key(key):
            return self.setup.__dict__[key]
        elif self.__dict__.has_key(key):
            return self.__dict__[key]
        
                
    def __setitem__(self, key, val):
        """Update an item values
        
        Searches in ``self.__dict__`` and ``self.setup`` and overwrites if 
        match found
        
        :param str key: key of item (e.g. base_dir)
        :param val: the replacement
        """
        if self.setup.__dict__.has_key(key):
            self.setup.__dict__[key] = val
        elif self.__dict__.has_key(key):
            self.__dict__[key] = val

    """
    Magic methods
    """    
    def print_list_info(self):
        """Print overview information about image lists"""
        s=("info about image lists in dataset\n-------------------------\n\n" +
            "Scene image lists:\n------------------------\n")
        for lst in self.img_lists.values():
            s += ("ID: %s, type: %s, %s images\n" 
                        %(lst.list_id, lst.list_type, lst.nof))
        s += "\nDark image lists:\n------------------------\n"
        for lst in self.dark_lists.values():
            s += ("ID: %s, type: %s, read_gain: %s,  %s images\n" 
                        %(lst.list_id, lst.list_type, lst.read_gain, lst.nof))
        print s

    """
    THE FOLLOWING STUFF WAS COPIED FROM OLD PLUMEDATA OBJECT
    """
    def connect_meas_geometry(self):
        """Set pointer to current measurement geometry within image lists"""
        if self.meas_geometry is not None:
            for filter_id in self.filters.filters.keys():
                self.get_list(filter_id).set_meas_geometry(self.meas_geometry)
        
    def plot_tau_preview(self, on_id = "on", off_id = "off", pyrlevel = 2):
        """Plot a preview of current tau_on, tau_off and AA images (AA plotted
        twice in 1st row of subplots in 2 diffent value ranges).
        
        :param str on_id: string ID of onband filter ("on")
        :param str off_id: string ID of offband filter ("off")
        :param pyrlevel: provide any integer here to reduce the image 
            sizes using a gaussian pyramide approach (2)
        """
        lists = {}
        tm = {on_id    :   1,
              off_id   :   1}
        fmt = lambda num,pos: '{:.1e}'.format(num)    
        for list_id in [on_id, off_id]:
            try:
                l = self.get_list(list_id)
                lists[list_id] = l
                if not l.bg_model.ready_2_go():
                    print ("Tau preview could not be plotted, bg model is not "
                                            " ready for filter: %s"  %list_id)
                    return 0
                if not l.tau_mode:
                    tm[list_id] = 0
                    l.activate_tau_mode()  
            except:
                print format_exc()
                return 0
        fig, axes = subplots(2, 2, figsize = (16, 10))
        tau_on = lists[on_id].current_img().pyr_down(pyrlevel)
        t_on_str = lists[on_id].current_time_str()
        tau_off = lists[off_id].current_img().pyr_down(pyrlevel)
        t_off_str = lists[off_id].current_time_str()
        aa = tau_on - tau_off #AA image object
        tau_max = max([tau_on.img.max(), tau_off.img.max(), aa.img.max()])
        tau_min = max([tau_on.img.min(), tau_off.img.min(), aa.img.min()])
        #make a color map for the index range
        cmap = shifted_color_map(tau_min, tau_max)

        im = axes[0, 0].imshow(tau_on.img, cmap = cmap,\
                                vmin = tau_min, vmax = tau_max)
        fig.colorbar(im, ax = axes[0, 0], format = FuncFormatter(fmt))
        axes[0,0].set_title("tau on: %s" %t_on_str)
        im=axes[0,1].imshow(tau_off.img, cmap = cmap,\
                                vmin = tau_min, vmax = tau_max)
        fig.colorbar(im, ax = axes[0, 1],format = FuncFormatter(fmt))
        axes[0, 1].set_title("tau off: %s" %t_off_str)
        im=axes[1, 0].imshow(aa.img, cmap = cmap,\
                                vmin = tau_min, vmax = tau_max)
        fig.colorbar(im, ax = axes[1, 0], format = FuncFormatter(fmt))
        axes[1, 0].set_title("AA (vals scaled)")
        cmap = shifted_color_map(aa.img.min(), aa.img.max())
        im = axes[1, 1].imshow(aa.img, cmap = cmap)
        fig.colorbar(im, ax = axes[1, 1], format = FuncFormatter(fmt))
        axes[1, 1].set_title("AA")
        tight_layout()
        for k, v in tm.iteritems():
            lists[k].activate_tau_mode(v)
        return axes