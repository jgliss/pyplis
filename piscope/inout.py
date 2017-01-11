# -*- coding: utf-8 -*-
"""
I/O routines for external data access
-------------------------------------
"""
from dill import load
from os.path import join, basename
from os import listdir

from matplotlib.pyplot import imread
from urllib2 import urlopen
from collections import OrderedDict as od

def load_img_dummy():
    """Load image dummy as numpy array"""
    from piscope import _LIBDIR
    return imread(join(_LIBDIR, "data", "no_images_dummy.png"))

def download_test_data(to_path = None):
    from piscope import _LIBDIR    
    if to_path is None:
        to_path = join()
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