# -*- coding: utf-8 -*-
"""
I/O routines for external data access
-------------------------------------
"""
from dill import load
from os.path import join, basename, exists
from os import listdir, remove

from matplotlib.pyplot import imread
from urllib2 import urlopen
from collections import OrderedDict as od
from traceback import format_exc

def download_test_data(save_path = None):
    """Download piscope test data from
    
    :param save_path: location where path is supposed to be stored
    
    Code for progress bar was "stolen" `here <http://stackoverflow.com/
    questions/11143767/how-to-make-a-download-with>`_ 
    (last access date: 11/01/2017)
    -progress-bar-in-python
    
    """
    from piscope import _LIBDIR
    from zipfile import ZipFile
    from urllib import urlretrieve
    from tempfile import mktemp
    try:
        from progressbar import ProgressBar, Percentage, Bar, RotatingMarker,\
                                        ETA, FileTransferSpeed
    
        widgets = ['Downloading piscope test data: ', Percentage(), ' ',\
                       Bar(marker=RotatingMarker()), ' ',\
                        ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets = widgets)
        def dl_progress(count, block_size, total_size):
            if pbar.maxval is None:
                pbar.maxval = total_size
                pbar.start()
            pbar.update(min(count*block_size, total_size))
            
        prog_ok = True
        
    except:
        print ("Failed to initiate progress bar for test data download, error "
        " msg: %s " %format_exc())
        prog_ok = False
    
    if save_path is None or not exists(save_path):
        save_path = join(_LIBDIR, "data")
        print "save path unspecified"
    else:
        with open(join(_LIBDIR, "data", "_paths.txt"), "a") as f:
            f.write("\n" + save_path  + "\n")
            print ("Adding new path for test data location in "
                    "file _paths.txt: %s" %save_path)
            f.close()
        
    print "installing test data at %s" %save_path
    
    url = 'https://folk.nilu.no/~gliss/piscope_testdata/piscope_etna_testdata.zip'
    
    filename = mktemp('.zip')
    if prog_ok:
        name, hdrs = urlretrieve(url, filename, reporthook = dl_progress)
        pbar.finish()
    else:
        print "Downloading piscope test data... (please wait)"
        name, hdrs = urlretrieve(url, filename)
    thefile = ZipFile(filename)
    print "Extracting data at: %s (this may take a while)" %save_path
    thefile.extractall(save_path)
    thefile.close()
    remove(filename)
    print ("Download successfully finished, deleting temporary data file at: %s"
        %filename)

def load_img_dummy():
    """Load image dummy as numpy array"""
    from piscope import _LIBDIR
    return imread(join(_LIBDIR, "data", "no_images_dummy.png"))

def find_test_data():
    """Searches location of test data folder"""
    from piscope import _LIBDIR
    data_path = join(_LIBDIR, "data")
    folder_name = "piscope_etna_testdata"
    if folder_name in listdir(data_path):
        print "Found test data at default location: %s" %data_path
        return join(data_path, folder_name)
    with open(join(data_path, "_paths.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            p = line.split("\n")[0]
            if exists(p) and folder_name in listdir(p):
                print "Found test data at default location: %s" %p
                f.close()
                return join(p, folder_name)
    raise IOError("piscope test data could not be found, please download"
        "testdata first, using method piscope.inout.download_test_data")

def all_test_data_paths():
    """Return list of all search paths for test data"""
    from piscope import _LIBDIR
    data_path = join(_LIBDIR, "data")
    paths = [data_path]
    with open(join(data_path, "_paths.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            p = line.split("\n")[0].lower()
            if exists(p):
                paths.append(p)
    return paths
    
def set_test_data_path(save_path):
    """Set local path where test data is stored"""
    from piscope import _LIBDIR
    if save_path.lower() in all_test_data_paths():
        print "Path is already in search tree"
        return
    try:
        if not exists(save_path):
            raise IOError("Could not set test data path: specified location "
                "does not exist: %s" %save_path)
        with open(join(_LIBDIR, "data", "_paths.txt"), "a") as f:
            f.write("\n" + save_path  + "\n")
            print ("Adding new path for test data location in "
                    "file _paths.txt: %s" %save_path)
            f.close()
        if not "piscope_etna_testdata" in listdir(save_path):
            print ("WARNING: test data folder (name: piscope_etna_testdata) "
                "could not be  found at specified location, please download "
                "test data, unzip and save at: %s" %save_path)
    except:
        raise
        
        
def get_camera_info(cam_id):
    """Try access camera information from file "cam_info.txt" (package data)
    
    :param str cam_id: string ID of camera (e.g. "ecII")
    
    """
    dat = {}
    if cam_id is None:
        return dat
    from piscope import _LIBDIR
    with open(join(_LIBDIR, "data", "cam_info.txt")) as f:
        filters = []
        darkinfo = []
        found = 0
        for line in f: 
            if "END" in line and found:
                dat["default_filters"] = filters
                dat["dark_info"] = darkinfo
                return dat
            spl = line.split(":")
            if found:
                if not any([line[0] == x for x in["#","\n"]]):
                    spl = line.split(":")
                    k = spl[0].strip()
                    if k == "dark_info":
                        l = [x.strip() for x in spl[1].split("#")[0].split(',')]
                        darkinfo.append(l)
                    elif k == "filter":
                        l = [x.strip() for x in spl[1].split("#")[0].split(',')]
                        filters.append(l)
                    else:
                        dStr = spl[1].split("#")[0].strip()
                        if any([dStr == x for x in ["''", '""']]):
                            dStr = ""
                        dat[k] = dStr
            if spl[0] == "cam_ids":
                if cam_id in [x.strip() for x in spl[1].split("#")[0].split(',')]:
                    found = 1    
    print ("Camera info for cam_id %s could not be found" %cam_id)
    return dat

def get_all_valid_cam_ids():
    """Load all valid camera string ids
    
    Reads info from file cam_info.txt which is part of package data
    """
    from piscope import _LIBDIR
    ids = []
    with open(join(_LIBDIR, "data", "cam_info.txt")) as f:        
        for line in f: 
            spl = line.split(":")
            if spl[0].strip().lower() == "cam_ids":
                ids.extend([x.strip() for x in spl[1].split("#")[0].split(',')])
    return ids

def get_cam_ids():
    """Load all default camera string ids
    
    Reads info from file cam_info.txt which is part of package data
    """
    from piscope import _LIBDIR
    ids = []
    with open(join(_LIBDIR, "data", "cam_info.txt")) as f:        
        for line in f: 
            spl = line.split(":")
            if spl[0].strip().lower() == "cam_id":
                ids.append(spl[1].split("#")[0].strip())
    return ids
    
def get_source_ids():
    """Get all existing source IDs
    
    Reads info from file my_sources.txt which is part of package data
    """
    from piscope import _LIBDIR
    ids = []
    with open(join(_LIBDIR, "data", "my_sources.txt")) as f:        
        for line in f: 
            spl = line.split(":")
            if spl[0].strip().lower() == "name":
                ids.append(spl[1].split("#")[0].strip())
    return ids
    
def get_source_info(source_id, try_online = True):
    """Try access source information from file "my_sources.txt" 
    
    File is part of package data
    
    :param str source_id: string ID of source (e.g. Etna)
    :param bool try_online: if True and local access fails, try to find source 
        ID in online database
    """
    from piscope import _LIBDIR
    dat = od()
    found = 0
    with open(join(_LIBDIR, "data", "my_sources.txt")) as f:        
        for line in f: 
            if "END" in line and found:
                return od([(source_id , dat)])
            spl = line.split(":")
            if found:
                if not any([line[0] == x for x in["#","\n"]]):
                    spl = line.split(":")
                    k = spl[0].strip()
                    dStr = spl[1].split("#")[0].strip()
                    dat[k] = dStr
            if spl[0] == "source_ids":
                if source_id in [x.strip() for x in spl[1].split("#")[0].split(',')]:
                    found = 1 
    print ("Source info for source %s could not be found" %source_id)
    if try_online:
        try:
            return get_source_info_online(source_id)
        except:
            pass
    return od()

def get_source_info_online(source_id):
    """Try to load source info from online database (@ www.ngdc.noaa.gov)
    
    :param str source_id: ID of source    
    """
    name = source_id
    name = name.lower()
    url=("http://www.ngdc.noaa.gov/nndc/struts/results?type_0=Like&query_0=&op"
        "_8=eq&v_8=&type_10=EXACT&query_10=None+Selected&le_2=&ge_3=&le_3=&ge_2"
        "=&op_5=eq&v_5=&op_6=eq&v_6=&op_7=eq&v_7=&t=102557&s=5&d=5")
    print "Trying to access volcano data from URL:"
    print url
    try:
        data = urlopen(url) # it's a file like object and works just like a file
    except:
        raise 
    
    res = od()
    in_row = 0
    in_data = 0
    lc = 0
    col_num = 10
    first_volcano_name = "Abu" #this needs to be identical
    ids = ["name", "country", "region", "lat", "lon", "altitude","type",\
                                                    "status","last_eruption"]
    types = [str, str, str, float, float, float, str, str, str]
    for line in data:
        lc += 1
        if first_volcano_name in line and line.split(">")[1].\
                        split("</td")[0].strip() == first_volcano_name:
            in_data, c = 1, 0
        if in_data:
            if c%col_num == 0 and name in line.lower():
                print "FOUND candidate, line: ", lc
                spl = line.split(">")[1].split("</td")[0].strip().lower()
                if name in spl:
                    print "FOUND MATCH: ", spl
                    in_row, cc = 1, 0
                    cid = spl
                    res[cid] = od()
            if in_row:
                spl = line.split(">")[1].split("</td")[0].strip()
                res[cid][ids[cc]] = types[cc](spl)
                cc += 1
                
            if in_row and cc == 9:
                print "End of data row reached for %s" %cid
                cc, in_row = 0, 0
            c += 1
    
    return res

def get_icon(name, color = None):
    """Try to find icon in lib icon folder
    
    :param str name: name of icon (i.e. filename is <name>.png)
    :param color (None): color of the icon ("r", "k", "g")
    
    Returns icon image filepath if valid
    
    """
    try:
        from piscope import _LIBDIR
    except:
        raise
    subfolders = ["axialis", "myIcons"]
    for subf in subfolders:
        base_path = join(_LIBDIR, "data", "icons", subf) 
        if color is not None:
            base_path = join(base_path, color)
        for file in listdir(base_path):
            fname = basename(file).split(".")[0]
            if fname == name:
                return base_path + file
    print "Failed to load icon at: " + _LIBDIR
    return False
    
def load_setup(fp):
    """Load a setup file which was stored as binary using :mod:`dill`"""
    return load(open(fp, "rb"))